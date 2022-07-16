from veros.settings import Setting


SETTINGS = {
    "enable_npzd": Setting(False, bool, ""),
    "enable_carbon": Setting(False, bool, ""),
    "_bgc_bluprint": Setting((), tuple, "Tuple of tracers and rules in setup"),
    #"foodweb": Setting(FoodWeb([],[]), FoodWeb, "Graph representation with tracers as nodes, rules as edges"),

    # I don't like keeping these in settings, but I can't make them in variables
    # maybe keep selected rules
    #"recycled": Setting({}, dict, "Amount of recycled material [mmol/m^3] for NPZD tracers"),
    #"mortality": Setting({}, dict, "Amount of dead plankton [mmol/m^3] by species"),
    #"net_primary_production": Setting({}, dict, "Primary production for each producing plankton species"),
    #"plankton_growth_functions": Setting({}, dict, "Collection of functions calculating growth for plankton by species"),
    #"limiting_functions": Setting({}, dict, "Collection of functions calculating limits to growth for plankton by species"),
    #"npzd_tracers": Setting({}, dict, "Dictionary whose values point to veros variables for npzd tracers"),



    #"light_attenuation_phytoplankton": Setting(0.047, float, "Light attenuation of phytoplankton"),
    "light_attenuation_water": Setting(0.04, float, "Light attenuation of water [1/m]"),
    "light_attenuation_ice": Setting(5.0, float, "Light attenuation of ice [1/m]"),
    #"remineralization_rate_detritus": Setting(0, float, "Remineralization rate of detritus [1/sec]"),
    "bbio": Setting(0, float, "the b in b ** (c*T)"),
    "cbio": Setting(0, float, "the c in b ** (c*T)"),
    #"maximum_growth_rate_phyto": Setting(0.0, float, "Maximum growth rate parameter for phytoplankton in [1/sec]"),
    #"maximum_grazing_rate": Setting(0, float, "Maximum grazing rate at 0 deg C [1/sec]"),
    #"fast_recycling_rate_phytoplankton": Setting(0, float, "Fast-recycling mortality rate of phytoplankton [1/sec]"),
    "saturation_constant_N": Setting(0.7, float, "Half saturation constant for N uptake [mmol N / m^3]"),
    #"saturation_constant_Z_grazing": Setting(0.15, float, "Half saturation constant for Z grazing [mmol/m^3]"),
    #"specific_mortality_phytoplankton": Setting(0, float, "Specific mortality rate of phytoplankton"),
    #"quadric_mortality_zooplankton": Setting(0, float, "Quadric mortality rate of zooplankton [1/ (mmol N ^2 s)]"),
    #"assimilation_efficiency": Setting(0, float, "Effiency with which ingested prey is converted growth in zooplankton, range: [0,1]"),
    #"zooplankton_growth_efficiency": Setting(0, float, "Zooplankton growth efficiency, range: [0,1]"),
    "wd0": Setting(0 / 86400, float, "Sinking speed of detritus at surface [m/s]"),
    "mwz": Setting(1000, float, "Depth below which sinking speed of detritus remains constant [m]"),
    "mw": Setting(0.02 / 86400, float, "Increase in sinking speed with depth [1/sec]"),
    "dcaco3": Setting(0, float, "Characteristic depth for CaCO3 redistribution"),
    #"zprefP": Setting(1, float, "Zooplankton preference for grazing on Phytoplankton"),
    #"zprefZ": Setting(1, float, "Zooplankton preference for grazing on other zooplankton"),
    #"zprefDet": Setting(1, float, "Zooplankton preference for grazing on detritus"),
    "redfield_ratio_PN": Setting(1. / 16, float, "Refield ratio for P/N"),
    "redfield_ratio_CP": Setting(7.1 * 16, float, "Refield ratio for C/P"),
    "redfield_ratio_ON": Setting(10.6, float, "Redfield ratio for O/N"),
    "redfield_ratio_CN": Setting(7.1, float, "Redfield ratio for C/N"),
    "trcmin": Setting(1e-13, float, "Minimum npzd tracer value"),
    "u1_min": Setting(1e-6, float, "Minimum u1 value for calculating avg J"),
    "zooplankton_max_growth_temp": Setting(20.0, float, "Temperature (C) for which zooplankton growth rate no longer grows with temperature"),
    "capr": Setting(0.022, float, "Carbonate to carbon production ratio"),

    #Setup parameters (initialised in the `set_parameter()` subroutine of a setup file):
    "dt_bio": Setting(0, int, "Timestep for biogeochemistry")
}
