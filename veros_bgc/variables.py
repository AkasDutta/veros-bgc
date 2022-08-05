# Add time_dependent=false where relevant

from veros.variables import Variable, T_GRID, T_HOR, YT, TIMESTEPS


VARIABLES = {
    # Variables which are part of the NPZD model
    "bottom_mask": Variable(
        "Bottom mask",
        T_GRID,
        "",
        "Bottom mask",
        dtype="int8",
        active=lambda settings: settings.enable_npzd,
    ),
    "bgc_tracers": Variable(
        "All tracers",
        T_GRID + TIMESTEPS + ("bgc_tracers_idx",),
        "variable",
        "5D array of many 4D tracers; variable size of 5th Dimension",
        write_to_restart=True,
    ),
    "bct": Variable(
        "common temperature factor",
        T_GRID,
        "",
        "common temperature factor determined according to b ** (cT)",
        active=lambda settings: settings.enable_npzd,
    ),
    "swr": Variable(
        "Shortwave radiation",
        T_HOR,
        "W/m^3?",
        "Incoming solar radiation at sea level",
        active=lambda settings: settings.enable_npzd,
    ),
    "rctheta": Variable(
        "Effective vertical coordinate for incoming solar radiation",
        YT,
        "1",
        "Effective vertical coordinate for incoming solar radiation",
        active=lambda settings: settings.enable_npzd,
    ),
    "dayfrac": Variable(
        "Fraction of day with sunlight",
        YT,
        1,
        "Fraction of day with sunlight",
        active=lambda settings: settings.enable_npzd,
    ),
    # What follows is again defined on T_GRID and may be better suited above. Perhaps it needs to be evaluated last?
    "excretion_total": Variable(
        "Total excretion from zooplankton",
        T_GRID,
        "mmol/m^3/s",
        "Zooplankton grazing causes excretion. This variable stores the total excreted amount for all consumed tracers.",
        active=lambda settings: settings.enable_npzd,
    ),
    # Now we list variables for NPZD + Carbon cycle setups.
    # BGC Functions involving the carbon cycle involve both sets of variables
    "atmospheric_co2": Variable(
        "Atmospheric CO2 concentration",
        T_HOR,
        "ppmv",
        "Atmospheric CO2 concentration",
        active=lambda settings: settings.enable_carbon,
    ),
    # Figure out the sign of this term
    "cflux": Variable(
        "DIC flux",
        T_HOR,
        "mmol/m^2/s",
        "Flux of CO2 over the ocean-atmosphere boundary",
        active=lambda settings: settings.enable_carbon,
    ),
    "wind_speed": Variable(
        "Debugging wind speed",
        T_HOR,
        "m/s",
        "Just used for debugging. Please ignore",
        active=lambda settings: settings.enable_carbon,
    ),
    "hSWS": Variable(
        "hSWS",
        T_HOR,
        "[H] in sea water sample",
        active=lambda settings: settings.enable_carbon,
    ),
    "pCO2": Variable(
        "pCO2",
        T_HOR,
        "?ppmv/atm?",
        "Partial CO2 pressure",
        active=lambda settings: settings.enable_carbon,
    ),
    "dpCO2": Variable(
        "dpCO2",
        T_HOR,
        "?ppmv/atm?",
        "Difference in ocean CO2 pressure and atmospheric",
        active=lambda settings: settings.enable_carbon,
    ),
    "co2star": Variable(
        "co2star",
        T_HOR,
        "?ppmv?",
        "Adjusted CO2 in ocean",
        active=lambda settings: settings.enable_carbon,
    ),
    "dco2star": Variable(
        "dco2star",
        T_HOR,
        "?ppmv?",
        "Adjusted CO2 difference",
        active=lambda settings: settings.enable_carbon,
    ),
    "rcak": Variable(
        "Calcite redistribution share",
        T_GRID,
        "1",
        "Calcite is redistributed after production by dissolution varying by depth",
        active=lambda settings: settings.enable_carbon,
    ),
}
