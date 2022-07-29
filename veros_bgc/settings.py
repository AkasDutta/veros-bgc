from veros.settings import Setting

SETTINGS = {
    "enable_npzd": Setting(False, bool, ""),
    "enable_carbon": Setting(False, bool, ""),
    "_bgc_rules_path": Setting("", str, "Filepath for rules"),
    "_bgc_tracers_path": Setting("", str, "Filepath for tracers"),
    "number_of_tracers": Setting(0, int, "Total number of tracers"),
    "light_attenuation_water": Setting(0.04, float, "Light attenuation of water [1/m]"),
    "light_attenuation_ice": Setting(5.0, float, "Light attenuation of ice [1/m]"),
    "bbio": Setting(0, float, "the b in b ** (c*T)"),
    "cbio": Setting(0, float, "the c in b ** (c*T)"),
    "saturation_constant_N": Setting(
        0.7, float, "Half saturation constant for N uptake [mmol N / m^3]"
    ),
    "wd0": Setting(0 / 86400, float, "Sinking speed of detritus at surface [m/s]"),
    "mwz": Setting(
        1000, float, "Depth below which sinking speed of detritus remains constant [m]"
    ),
    "mw": Setting(0.02 / 86400, float, "Increase in sinking speed with depth [1/sec]"),
    "dcaco3": Setting(0, float, "Characteristic depth for CaCO3 redistribution"),
    "redfield_ratio_PN": Setting(1.0 / 16, float, "Refield ratio for P/N"),
    "redfield_ratio_CP": Setting(7.1 * 16, float, "Refield ratio for C/P"),
    "redfield_ratio_ON": Setting(10.6, float, "Redfield ratio for O/N"),
    "redfield_ratio_CN": Setting(7.1, float, "Redfield ratio for C/N"),
    "trcmin": Setting(1e-13, float, "Minimum npzd tracer value"),
    "u1_min": Setting(1e-6, float, "Minimum u1 value for calculating avg J"),
    "zooplankton_max_growth_temp": Setting(
        20.0,
        float,
        "Temperature (C) for which zooplankton growth rate no longer grows with temperature",
    ),
    "capr": Setting(0.022, float, "Carbonate to carbon production ratio"),
    # Setup parameters (initialised in the `set_parameter()` subroutine of a setup file):
    "dt_bio": Setting(0, int, "Timestep for biogeochemistry"),
}
