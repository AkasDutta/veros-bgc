"""
Collection of rules to be used by the npzd module

Rules should always take 3 arguments:
    1. state -  the veros state object
    2. Name of source tracer
    3. Name of sink tracer
"""
# PENDING CHANGE: Dicts of tracers - eg. settings.npzd_export -  are currently calling the associated arrays as settings.npzd_export[tracer].
#                Replace this with settings.npzd_export[tracer].data
# sloppy_feeding, grazing, excretion??


# Plan: Store only rule archetypes here (eg: `recycling`)
#      Create a .yml file with everything else (eg: fields for `recycling_to_no3`)
#      Let the set_foodweb function -  which will be the setup_entrypoint - read this file
#      and construct and register rules therefrom.

# Advantages: Firstly, the question of group rules disappears; common rule groups will now
#      become commonly used .yaml files. Secondly, the distinction between selecting and
#      registering a rule becomes irrelevant, as we don't have every individual rule hard-coded
#      Thirdly, it provides symmetry between the treatment of tracers and rules, i.e, nodes
#      and edges in the foodweb.

from veros import veros_routine
from veros.core.operators import numpy as npx, at, update, update_add, update_multiply
from collections import namedtuple

from . import atmospherefluxes


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
    "mortality": (mortality, ["plankton", "detritus"]),
    "grazing_cycle": (grazing_cycle, ["eaten", "zooplankton"]),
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
    def call(self, state):
        vs = state.variables
        settings = state.settings

        lst = []
        for arg in RuleTemplates[self._function][1]:
            lst.append(self._arguments[arg])
        updates = RuleTemplates[self._function][0](state, *lst)
        for key, value in updates.items():
            updates[key] = update_multiply(value, at[...], self.flag)
        return updates
