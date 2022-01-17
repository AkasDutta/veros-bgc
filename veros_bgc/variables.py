#Remove output=.. statements
#Add time_dependent=false where relevant

from veros.variables import Variable, T_GRID, T_HOR, YT, TIMESTEPS


VARIABLES = {
    #Variables which are part of the NPZD model
    "bottom_mask": Variable(
        "Bottom mask",
        T_GRID, 
        "",
        "Bottom mask",
        dtype="int8",
        active=lambda settings:settings.enable_npzd
    ),
    "phytoplankton": Variable(
        "Phytoplankton concentration",
        T_GRID + TIMESTEPS,
        "mmol/m^3?",
        "Concentration of phytoplankton in grid box",
         output=True,
         write_to_restart=True,
        active=lambda settings:settings.enable_npzd
    ),
    "zooplankton": Variable(
        "Zooplankton concentration",
        T_GRID + TIMESTEPS,
        "mmol/m^3?",
        "Concentration of zooplankton in grid box,"
        output=True,
        write_to_restart=True,
        active=lambda settings:settings.enable_npzd
    ),
    "detritus": Variable(
        "Detritus concentration",
        T_GRID + TIMESTEPS,
        "mmol/m^3?",
        "Concentration of detritus in grid box",
        output=True,
        write_to_restart=True,
        active=lambda settings:settings.enable_npzd
    ),
    "po4": Variable(
        "Phosphate concentration",
        T_GRID+TIMESTEPS,
        output=True,
        write_to_restart=True,
        active=lambda settings:settings.enable_npzd
    ),
    "swr": Variable(
        "Shortwave radiation",
        T_HOR,
        "W/m^3?",
        "Incoming solar radiation at sea level",
        active=lambda settings:settings.enable_npzd
    ),
    "rctheta": Variable(
        "Effective vertical coordinate for incoming solar radiation",
        YT,
        "1",
        "Effective vertical coordinate for incoming solar radiation",
        active=lambda settings:settings.enable_npzd
    ),
    "dayfrac": Variable(
        "Fraction of day with sunlight",
        YT,
        1
        "Fraction of day with sunlight",
        active=lambda settings:settings.enable_npzd
    ),
    #What follows is again defined on T_GRID and may be better suited above. Perhaps it needs to be evaluated last?
    "excretion_total": Variable(
        "Total excretion from zooplankton",
        T_GRID,
        "mmol/m^3/s",
        "Zooplankton grazing causes excretion. This variable stores the total excreted amount for all consumed tracers.",
        active=lambda settings:settings.enable_npzd
    ),
    
    #Now we list variables for NPZD + Carbon cycle setups.
    #BGC Functions involving the carbon cycle involve both sets of variables
    
    "dic": Variable(
        "Dissolved Inorganic Carbon",
        T_GRID + TIMESTEPS,
        "mmol/m^3",
        "Concentration of inorganic carbon ions and molecules",
        output=True,
        write_to_restart=True,
        active=lambda settings:settings.enable_carbon
    ),
    "alkalinity": Variable(
        "Alkalinity",
        T_GRID + TIMESTEPS,
        "mmol/m^3",
        "Combined bases and acids",
        output=True,
        write_to_restart=True,
        active=lambda settings:settings.enable_carbon
    ),
    "atmospheric_co2": Variable(
        "Atmospheric CO2 concentration",
        T_HOR,
        "ppmv",
        "Atmospheric CO2 concentration",
        active=lambda settings:settings.enable_carbon
    ),
    #Figure out the sign of this term
    "cflux": Variable(
        "DIC flux",
        T_HOR,
        "mmol/m^2/s",
        "Flux of CO2 over the ocean-atmosphere boundary",
        output=True,
        active=lambda settings:settings.enable_carbon
    ),
    "wind_speed": Variable(
        "Debugging wind speed",
        T_HOR,
        "m/s",
        "Just used for debugging. Please ignore",
        output=True,
        active=lambda settings:settings.enable_carbon
    ),
    "hSWS":
    


MAIN_VARIABLES = OrderedDict([])

CONDITIONAL_VARIABLES = OrderedDict([
    ('enable_carbon', OrderedDict([
        ('hSWS', Variable('hSWS', T_HOR, '1',
                          '[H] in Sea water sample', output=True)),
        ('pCO2', Variable('pCO2', T_HOR, '?ppmv/atm?',
                          'Partial CO2 pressure', output=True)),
        ('dpCO2', Variable('dpCO2', T_HOR, '?ppmv/atm?',
                           'Difference in ocean CO2 pressure and atmospheric', output=True)),
        ('co2star', Variable('co2star', T_HOR, '?ppmv?',
                             'Adjusted CO2 in ocean', output=True)),
        ('dco2star', Variable('dco2star', T_HOR, '?ppmv?',
                              'Adjusted CO2 difference', output=True)),
        ('rcak', Variable('Calcite redistribution share', T_GRID, '1',
                          'Calcite is redistributed after production by dissolution varying by depth')),
    ])),
])
