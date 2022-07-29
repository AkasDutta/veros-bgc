"""
This file serves as the setup entrypoint for the veros bgc plugin.
It completes the model setup after the setup files set parameter, forcing, topography, etc.
The aim is to complete populating settings, and to initialise all objects we need.
"""

from veros.variables import Variable
from veros.core.operators import numpy as npx, update, at, update_add, update_multiply
import networkx as nx
from veros import veros_routine, veros_kernel, KernelOutput

from collections import namedtuple

from .npzd_tracers import TracerClasses
from .npzd_rules import RuleTemplates, Rule


@veros_routine
def setup_carbon(state, zw):
    vs = state.variables
    settings = state.settings

    # redistribution fraction for calcite at level k
    vs.rcak[:, :, :-1] = (
        -npx.exp(zw[:-1] / settings.dcaco3) + npx.exp(zw[1:] / settings.dcaco3)
    ) / vs.dzt[:-1]
    vs.rcak[:, :, -1] = -(npx.exp(zw[-1] / settings.dcaco3) - 1.0) / vs.dzt[-1]

    # redistribution fraction at bottom
    rcab = npx.empty_like(vs.dic[..., 0])
    rcab[:, :-1] = 1 / vs.dzt[-1]
    rcab[:, :, :-1] = npx.exp(zw[:-1] / settings.dcaco3) / vs.dzt[1:]

    # merge bottom into level k and reset every cell outside ocean
    vs.rcak[vs.bottom_mask] = rcab[vs.bottom_mask]
    vs.rcak = update_multiply(vs.rcak, at[...], vs.maskT)


@veros_routine
def setupNPZD(state):
    vs = state.variables
    settings = state.settings

    vs.bottom_mask[:, :, :] = (
        npx.arange(vs.nz)[npx.newaxis, npx.newaxis, :]
        == (vs.kbot - 1)[:, :, npx.newaxis]
    )

    zw = vs.zw - vs.dzt  # bottom of grid box using dzt because dzw is weird
    dtr_speed = (vs.wd0 + vs.mw * npx.where(-zw < vs.mwz, -zw, vs.mwz)) * vs.maskT

    if settings.enable_npzd:
        if settings.enable_carbon:
            setup_carbon(state, zw)
