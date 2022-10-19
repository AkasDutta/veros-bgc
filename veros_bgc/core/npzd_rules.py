"""
Collection of rules to be used by the npzd module

Rules should always take at least 3 arguments:
    1. state -  the veros state object
    2. The source tracer (or list of tracers)
    3. The sink tracer (or list of tracers)

Where relevant, other information (such as the ratios in which to divide
source biomass among sink tracers) may also be provided.

This file defines functions which are archetypes of rules - eg: `recycling`
A .yaml file is read during setup to construct concrete rules--see foodweb.y
-- such as recycling detritus to dic or to po4. Each function returns a
dictionary of updates the reelvant tracers. We also define a class Rule that
consists of the function and metadata specifying when and in which grid cells
it is to be called. The creation of concrete rules is caried out by initialising
objects of this class.

ToDo: Make a type check for the tracers supplied for various rules.
"""


from veros import veros_routine
from veros.core.operators import numpy as npx, at, update, update_add, update_multiply
from collections import namedtuple


@veros_routine
def empty_rule(*args):
    """An empty rule for providing structure"""
    return {}


@veros_routine
def primary_production(state, nutrients, plankton, ratio):
    """Primary production: Growth by consumption of light and nutrients"""
    settings = state.settings

    updates = {
        nutrients[i].name: -ratio[i] * plankton.net_primary_production
        for i in range(len(nutrients))
    }
    updates[plankton.name] = plankton.net_primary_production
    return updates


@veros_routine
def recycling(state, plankton, nutrients, ratio):
    """Plankton or detritus is recycled into nutrients

    Parameters
    ----------
    ratio
        Factor for adjusting nutrient contribution. Typically a Redfield ratio.
    """
    settings = state.settings
    updates = {
        nutrients[i].name: ratio[i] * plankton.recycle(state)
        for i in range(len(nutrients))
    }
    updates[plankton.name] = -plankton.recycle(state)

    return updates


@veros_routine
def mortality(state, plankton, remains):
    """All dead matter from plankton is converted to detritus"""
    settings = state.settings
    death = plankton.mortality(state)
    updates = {plankton.name: -death}

    if not plankton.calcite_producing:
        updates[remains.name] = death
    else:
        detritus = remains[0]
        calcite = remains[1]
        updates[detritus.name] = death
        updates[calcite.name] = death * settings.capr * settings.redfield_ratio_CN

        calcite.dprca = updates[calcite.name]

    return updates


@veros_routine
def grazing_cycle(state, prey, grazer, remains, nutrients, ratios):
    vs = state.variables
    settings = state.settings
    foodweb = state.foodweb

    grazed, digested, excreted, sloppy_feeding = grazer.grazing(state, prey)

    updates = {grazer.name: digested, prey.name: grazed}

    if not prey.calcite_producing:
        updates[remains.name] = sloppy_feeding
    else:
        detritus = remains[0]
        calcite = remains[1]
        updates[detritus.name] = sloppy_feeding
        updates[calcite.name] = (
            sloppy_feeding * settings.capr * settings.redfield_ratio_CN
        )
        calcite.dprca = updates[calcite.name]

    for i in range(len(nutrients)):
        updates[nutrients[i].name] = ratios[i] * excreted

    return updates


@veros_routine
def calcite_production(state, dic, alk, calcite):
    """Calcite is produced at a rate similar to detritus, i.e, pieces of calcite-producing
    plankton that fall as detritus are pieces of calcite.

    This calcite resulted from carbon fixation and reqeuires that dssolved carbon is depleted
    to make up for it.

    Intended for use with a smoothing rule
    If explicit tracking of calcite is desired use
    rules for the explicit relationship
    """
    vs = state.variables
    settings = state.settings

    # changes to production of calcite
    dprca = calcite.dprca
    return {dic.name: -dprca, alk.name: -2 * dprca}


@veros_routine
def post_redistribute_calcite(state, calcite, tracers, ratios):
    """Post rule to redistribute produced calcite"""
    vs = state.variables
    settings = state.settings
    foodweb = settings.foodweb

    total_production = (calcite.temp * vs.dzt).sum(axis=2)
    redistributed_production = total_production[:, :, npx.newaxis] * vs.rcak

    updates = {
        tracers[i].name: ratios[i] * redistributed_production
        for i in range(len(tracers))
    }
    return updates


@veros_routine
def pre_reset_calcite(state, calcite):
    """Pre rule to reset calcite production"""
    vs = state.variables
    settings = state.settings

    return {calcite.name: -calcite.temp}


@veros_routine
def co2_surface_flux(state, co2, dic):
    """Pre rule to add or remove DIC from surface layer"""
    from . import atmospherefluxes

    vs = state.variables

    atmospherefluxes.carbon_flux(state)
    flux = vs.cflux * vs.dt_tracer / vs.dzt[-1]
    return {dic.name: flux}  # NOTE we don't have an atmosphere, so this rules is a stub


@veros_routine
def dic_alk_scale(state, dic, alkalinity):
    """Redistribute change in DIC as change in alkalinity"""
    vs = state.variables
    settings = state.settings

    return {
        alkalinity: (dic.temp - dic.data[:, :, :, vs.tau]) / settings.redfield_ratio_CN
    }


@veros_routine
def bottom_remineralization(state, source, sink, scale):
    """Exported material falling through the ocean floor is converted to nutrients

    Note
    ----
    There can be no source, because that is handled by the sinking code

    Parameters
    ----------
    scale
        Factor to convert remineralized material to nutrient. Typically Redfield ratio.
    """
    vs = state.variables
    settings = state.settings
    foodweb = settings.foodweb

    return {sink.name: foodweb.deposits[source.name] * scale}


RuleTemplates = {
    "empty_rule": (empty_rule, []),
    "primary_production": (primary_production, ["nutrients", "plankton", "ratio"]),
    "recycling": (recycling, ["plankton", "nutrients", "ratio"]),
    "mortality": (mortality, ["plankton", "remains"]),
    "grazing_cycle": (
        grazing_cycle,
        ["prey", "grazer", "remains", "nutrients", "ratios"],
    ),
    "calcite_production": (calcite_production, ["plankton", "DIC", "alk", "calcite"]),
    "post_redistribute_calcite": (
        post_redistribute_calcite,
        ["calcite", "tracers", "ratios"],
    ),
    "pre_reset_calcite": (pre_reset_calcite, ["calcite"]),
    "co2_surface_flux": (co2_surface_flux, ["co2", "dic"]),
    "dic_alk_scale": (dic_alk_scale, ["DIC", "alkalinity"]),
    "bottom_remineralization": (bottom_remineralization, ["source", "sink", "scale"]),
}


class Rule:
    def __init__(
        self,
        state,
        name,
        function,
        arguments,
        source,
        sink,
        label=None,
        boundary=None,
        group="Primary",
    ):
        vs = state.variables
        settings = state.settings

        self.name = name
        self.source = source
        self.sink = sink
        self.label = label
        self.group = group
        self.boundary = self._get_boundary(state, boundary)
        self._function = function
        self._arguments = arguments
        self.flag = npx.empty_like(vs.maskT)

    @veros_routine
    def _get_boundary(self, state, boundary_string):
        """Return slice representing boundary

        Parameters
        ----------
        boundary_string
            Identifer for boundary. May take one of the following values:
            SURFACE:       [:, :, -1] only the top layer
            BOTTOM:        bottom_mask as set by veros
            else:          [:, :, :] everything
        """

        vs = state.variables
        settings = state.settings

        if boundary_string == "SURFACE":
            return tuple([slice(None, None, None), slice(None, None, None), -1])

        if boundary_string == "BOTTOM":
            return vs.bottom_mask

        return tuple([slice(None, None, None)] * 3)

    @veros_routine
    def call(self, state, foodweb):

        lst = []
        for arg in RuleTemplates[self._function][1]:
            val = self._arguments[arg]
            if isinstance(val, tuple):
                if val[0] == "s":
                    self._arguments[arg] = val[1](state)
                elif val[0] == "f":
                    self._arguments[arg] = val[1](foodweb)
            lst.append(self._arguments[arg])
        updates = RuleTemplates[self._function][0](state, *lst)
        for key, value in updates.items():
            updates[key] = update_multiply(value, at[...], self.flag)
        return updates
