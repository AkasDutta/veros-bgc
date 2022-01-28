#!/usr/bin/env python

import os
import h5netcdf
import ruamel.yaml as yaml

import veros.tools
from veros import VerosSetup, veros_routine, veros_kernel, KernelOutput, logger
from veros.variables import Variable
from veros.core.operators import numpy as npx, update, 


import veros_bgc

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_FILES = veros.tools.get_assets(
    'global_4deg',
    os.path.join(BASE_PATH, 'assets.yml')
)


class GlobalFourDegreeBGC(VerosSetup):
    """Global 4 degree model with 15 vertical levels and biogeochemistry.

    Reference:
        Steffen Ole Randrup Kristensen. (2019). Extending the Veros climate simulator with biochemistry.
        Model design and AMOC collapse, MSc, 67p.
        `<https://sid.erda.dk/share_redirect/CVvcrowL22/Thesis/SteffenRandrup_MSc_thesis.pdf>`_.

    """
    __veros_plugins__ = (veros_bgc,)

    @veros_routine
    def set_parameter(self, state):
        settings = state.settings
        
        settings.identifier = '4deg'

        settings.nx, settings.ny, settings.nz = 90, 40, 15
        settings.dt_mom = 1800.0
        settings.dt_tracer = 86400.0
        settings.dt_bio = settings.dt_tracer // 4
        settings.runlen = 0.
        
        settings.x_origin = 4.0
        settings.y_origin = -76.0

        settings.trcmin = 0
        settings.enable_npzd = True
        settings.enable_carbon = True

        with open(os.path.join(BASE_PATH, "npzd.yml")) as yaml_file:
            cfg = yaml.safe_load(yaml_file)["npzd"]
            settings.npzd_selected_rules = cfg["selected_rules"]

        settings.remineralization_rate_detritus = 0.09 / 86400
        settings.bbio = 1.038
        settings.cbio = 1.0
        settings.maximum_growth_rate_phyto = 0.23 / 86400
        settings.maximum_grazing_rate = 0.13 / 86400
        settings.fast_recycling_rate_phytoplankton = 0.025 / 86400
        settings.specific_mortality_phytoplankton = 0.035 / 86400
        settings.quadric_mortality_zooplankton = 0.06 / 86400
        settings.zooplankton_growth_efficiency = 0.60
        settings.assimilation_efficiency = 0.5
        settings.wd0 = 2 / 86400
        settings.mwz = 1000
        settings.mw = 0.02 / 86400
        settings.dcaco3 = 2500

        settings.coord_degree = True
        settings.enable_cyclic_x = True

        settings.enable_neutral_diffusion = True
        settings.K_iso_0 = 1000.0
        settings.K_iso_steep = 500.0
        settings.iso_dslope = 0.001
        settings.iso_slopec = 0.001
        settings.enable_skew_diffusion = True

        settings.enable_hor_friction = True
        settings.A_h = (4 * settings.degtom)**3 * 2e-11
        settings.enable_hor_friction_cos_scaling = True
        settings.hor_friction_cosPower = 1

        settings.enable_implicit_vert_friction = True
        settings.enable_tke = True
        settings.c_k = 0.1
        settings.c_eps = 0.7
        settings.alpha_tke = 30.0
        settings.mxl_min = 1e-8
        settings.tke_mxl_choice = 2
        settings.kappaM_min = 2e-4
        settings.kappaH_min = 7e-5  # usually 2e-5
        settings.enable_Prandtl_tke = False
        settings.enable_kappaH_profile = True

        # eke
        settings.K_gm_0 = 1000.0 #Hm
        settings.enable_eke = False
        settings.eke_k_max = 1e4
        settings.eke_c_k = 0.4
        settings.eke_c_eps = 0.5
        settings.eke_cross = 2.
        settings.eke_crhin = 1.0
        settings.eke_lmin = 100.0
        settings.enable_eke_superbee_advection = False
        settings.enable_eke_isopycnal_diffusion = False

        # idemix
        settings.enable_idemix = False
        settings.enable_idemix_hor_diffusion = False
        settings.enable_eke_diss_surfbot = False
        settings.eke_diss_surfbot_frac = 0.2  #  fraction which goes into bottom
        settings.enable_idemix_superbee_advection = False

        settings.eq_of_state_type = 5

        # custom variables
        state.dimensions["nmonths"] = 12
        state.var_meta.update(
            sss_clim=Variable("sss_clim", ("xt", "yt", "nmonths"), "", "", time_dependent=False),
            sst_clim=Variable("sst_clim", ("xt", "yt", "nmonths"), "", "", time_dependent=False),
            qnec=Variable("qnec", ("xt", "yt", "nmonths"), "", "", time_dependent=False),
            qnet=Variable("qnet", ("xt", "yt", "nmonths"), "", "", time_dependent=False),
            taux=Variable("taux", ("xt", "yt", "nmonths"), "", "", time_dependent=False),
            tauy=Variable("tauy", ("xt", "yt", "nmonths"), "", "", time_dependent=False),
        )

    @veros_routine
    def _read_forcing(self, var):
        with h5netcdf.File(DATA_FILES["forcing"], "r") as infile:
            var_obj = infile.variables[var]
            return npx.array(var_obj).T

    @veros_routine
    def set_grid(self, state):
        vs = state.variables
        ddz = npx.array([50., 70., 100., 140., 190., 240., 290., 340.,
                        390., 440., 490., 540., 590., 640., 690.])
        
        vs.dzt = ddz[::-1]
        vs.dxt = 4.0*npx.ones_like(vs.dxt)
        vs.dyt = 4.0*npx.ones_like(vs.dyt)

        idx_global, _ = distributed.get_chunk_slices(vs, ('xt', 'yt', 'zt')) #hm

        with h5netcdf.File(DATA_FILES['corev2'], 'r') as infile:
            vs.swr_initial = infile.variables['SWDN_MOD'][idx_global[::-1]].T #Might move this down

    @veros_routine
    def set_coriolis(self, state):
        vs=state.variables
        settings=state.settigs
        vs.coriolis_t= update(
          vs.coriolis_t, at[...], 2 * settings.omega * npx.sin(vs.yt[npx.newaxis, :] / 180. * settings.pi)

    @veros_routine(dist_safe=False, local_variables=["kbot", "zt"])
    def set_topography(self, state):
        vs = state.variables
        settings = state.settings

        bathymetry_data = self._read_forcing("bathymetry")
        salt_data = self._read_forcing("salinity")[:, :, ::-1]
        land_mask = (vs.zt[npx.newaxis, npx.newaxis, :] <= bathymetry_data[..., npx.newaxis]) | (salt_data == 0.0)

        vs.kbot = update(vs.kbot, at[2:-2, 2:-2], 1 + npx.sum(land_mask.astype("int"), axis=2))

        # set all-land cells
        all_land_mask = (bathymetry_data == 0) | (vs.kbot[2:-2, 2:-2] == settings.nz)
        vs.kbot = update(vs.kbot, at[2:-2, 2:-2], npx.where(all_land_mask, 0, vs.kbot[2:-2, 2:-2]))

    @veros_routine(
      dist_safe=False,
      local_variables=[
        "taux",
        "tauy",
        "qnec", 
        "qnet",
        "sss_clim",
        "sst_clim",
        "temp",
        "salt",
        "area_t",
        "maskT",
        "forc_iw_bottom",
        "forc_iw_surface",
        "zw",
        "phytoplankton",
        "zooplankton",
        "detritus",
        "po4",
        "dic",
        "atmospheric_co2",
        "alkalinity",
        "hSWS"
      ],
    )
    def set_initial_conditions(self, state):
        
        vs = state.variables
        settings = state.settings
        
        # initial conditions for T and S
        
        temp_data = self._read_forcing('temperature')[:, :, ::-1]
        vs.temp = update(
            vs.temp, at[2:-2, 2:-2, :, :2], temp_data[:, :, :, npx.newaxis] * vs.maskT[2:-2, 2:-2, :, npx.newaxis]
        )
        
        salt_data = self._read_forcing('salinity')[:, :, ::-1]
        vs.salt = update(
            vs.salt, at[2:-2, 2:-2, :, :2], salt_data[..., npx.newaxis] * vs.maskT[2:-2, 2:-2, :, npx.newaxis]
        )

        # use Trenberth wind stress from MITgcm instead of ECMWF (also contained in ecmwf_4deg.cdf)
        vs.taux = update(vs.taux, at[2:-2, 2:-2, :], self._read_forcing("tau_x"))
        vs.tauy = update(vs.tauy, at[2:-2, 2:-2, :], self._read_forcing("tau_y"))

        # heat flux
        with h5netcdf.File(DATA_FILES["ecmwf"], "r") as ecmwf_data:
            qnec_var = ecmwf_data.variables["Q3"]
            vs.qnec = update(vs.qnec, at[2:-2, 2:-2, :], npx.array(qnec_var).T)
            vs.qnec = npx.where(vs.qnec <= -1e10, 0.0, vs.qnec)

        q = self._read_forcing("q_net")
        vs.qnet = update(vs.qnet, at[2:-2, 2:-2, :], -q)
        vs.qnet = npx.where(vs.qnet <= -1e10, 0.0, vs.qnet)

        mean_flux = (
            npx.sum(vs.qnet[2:-2, 2:-2, :] * vs.area_t[2:-2, 2:-2, npx.newaxis]) / 12 / npx.sum(vs.area_t[2:-2, 2:-2])
        )
        logger.info(" removing an annual mean heat flux imbalance of %e W/m^2" % mean_flux)
        vs.qnet = (vs.qnet - mean_flux) * vs.maskT[:, :, -1, npx.newaxis]

        # SST and SSS
        vs.sst_clim = update(vs.sst_clim, at[2:-2,2:-2,:], self._read_forcing("sst"))
        vs.sss_clim = update(vs.sss_clim, at[2:-2,2:-2,:], self._read_forcing("sss"))


        if settings.enable_idemix:
            vs.forc_iw_bottom = update(vs.forc_iw_bottom, at[2:-2, 2:-2], 
                                       self._read_forcing("tidal_energy")/settings.rho_0
                                      )            
            vs.forc_iw_surface = update(vs.forc_iw_surface, at[2:-2,2:-2],
                                        self._read_forcing("wind_energy")/settings.rho_0*0.2
                                       )


        if settings.enable_npzd:
            phytoplankton = 0.14 * npx.exp(vs.zw / 100) * vs.maskT
            zooplankton = 0.014 * npx.exp(vs.zw / 100) * vs.maskT
            
            vs.phytoplankton =  update(vs.phytoplankton, at[:, :, :, :], phytoplankton[..., npx.newaxis])
            vs.zooplankton =  update(vs.zooplankton, at[:, :, :, :], zooplankton[..., npx.newaxis])
            vs.detritus =  update(vs.detritus, at[:, :, :, :], 1e-4*vs.maskT[..., npx.newaxis])

            vs.po4 = update(vs.po4, at[:, :, :, :], 2.2)
            vs.po4 = update_multiply(vs.po4, at[...], vs.maskT[..., npx.newaxis]

        if settings.enable_carbon:
            vs.dic = update(vs.dic, at[...], 2300*vs.maskT[..., npx.newaxis])
            vs.atmospheric_co2 = update(vs.atmospheric_co2, at[...], 280)
            vs.alkalinity = update(vs.alkalinity, at[...], 2400*vsmaskT[..., npx.newaxis])
            vs.hSWS = update(vs.hSWS, at[...], 5e-7)

    @veros_routine
    def set_forcing(self, state):
        vs = state.variables
        vs.update(set_forcing_kernel(state))
                                     @veros_kernel
    def set_forcing_kernel(state):
        vs = state.variables
        settings = state.settings

        year_in_seconds = 360 * 86400.0
        (n1, f1), (n2, f2) = veros.tools.get_periodic_interval(vs.time, year_in_seconds, year_in_seconds / 12.0, 12)

        # wind stress
        vs.surface_taux = f1 * vs.taux[:, :, n1] + f2 * vs.taux[:, :, n2]
        vs.surface_tauy = f1 * vs.tauy[:, :, n1] + f2 * vs.tauy[:, :, n2]

        # tke flux
        if settings.enable_tke:
            vs.forc_tke_surface = update(
                vs.forc_tke_surface,
                at[1:-1, 1:-1],
                npx.sqrt(
                    (0.5 * (vs.surface_taux[1:-1, 1:-1] + vs.surface_taux[:-2, 1:-1]) / settings.rho_0) ** 2
                    + (0.5 * (vs.surface_tauy[1:-1, 1:-1] + vs.surface_tauy[1:-1, :-2]) / settings.rho_0) ** 2
                    )
                     ** 1.5,
                )
                                     
        # heat flux : W/m^2 K kg/J m^3/kg = K m/s
        cp_0 = 3991.86795711963
        sst = f1 * vs.sst_clim[:, :, n1] + f2 * vs.sst_clim[:, :, n2]
        qnec = f1 * vs.qnec[:, :, n1] + f2 * vs.qnec[:, :, n2]
        qnet = f1 * vs.qnet[:, :, n1] + f2 * vs.qnet[:, :, n2]
        vs.forc_temp_surface = (
            (qnet + qnec * (sst - vs.temp[:, :, -1, vs.tau])) * vs.maskT[:, :, -1] / cp_0 / settings.rho_0
        )

        # salinity restoring
        t_rest = 30 * 86400.0
        sss = f1 * vs.sss_clim[:, :, n1] + f2 * vs.sss_clim[:, :, n2]
        vs.forc_salt_surface = 1.0 / t_rest * (sss - vs.salt[:, :, -1, vs.tau]) * vs.maskT[:, :, -1] * vs.dzt[-1]

        # apply simple ice mask
        mask = npx.logical_and(vs.temp[:, :, -1, vs.tau] * vs.maskT[:, :, -1] < -1.8, vs.forc_temp_surface < 0.0)
        vs.forc_temp_surface = npx.where(mask, 0.0, vs.forc_temp_surface)
        vs.forc_salt_surface = npx.where(mask, 0.0, vs.forc_salt_surface)
        if vs.enable_npzd:
            # incoming shortwave radiation for plankton production
            vs.swr[2:-2, 2:-2] = (f1 * vs.swr_initial[:, :, n1] + f2 * vs.swr_initial[:, :, n2])
        

        return KernelOutput(
            surface_taux=vs.surface_taux,
            surface_tauy=vs.surface_tauy,
            forc_tke_surface=vs.forc_tke_surface,
            forc_temp_surface=vs.forc_temp_surface,
            forc_salt_surface=vs.forc_salt_surface,
            swr=vs.swr
        )


    @veros_method
    def set_diagnostics(self, vs):
        vs.diagnostics['snapshot'].output_frequency = 360 * 86400.

        vs.diagnostics['overturning'].output_frequency = 360 * 86400.
        vs.diagnostics['overturning'].sampling_frequency = vs.dt_tracer

        vs.diagnostics['energy'].output_frequency = 360 * 86400.
        vs.diagnostics['energy'].sampling_frequency = 86400

        average_vars = [
            'phytoplankton', 'po4', 'zooplankton', 'detritus', 'wind_speed', 'dic', 'alkalinity',
            'temp', 'salt', 'u', 'v', 'surface_tauy', 'surface_taux', 'kappaH', 'psi', 'rho',
            'pCO2', 'dpCO2', 'dco2star', 'cflux', 'atmospheric_co2'
        ]
        vs.diagnostics['averages'].output_variables = average_vars
        vs.diagnostics['averages'].output_frequency = 360 * 86400.
        vs.diagnostics['averages'].sampling_frequency = 86400

        vs.diagnostics['npzd'].output_frequency = 10 * 86400
        vs.diagnostics['npzd'].save_graph = False

    @veros_method
    def after_timestep(self, vs):
        pass


@veros.tools.cli
def run(*args, **kwargs):
    simulation = GlobalFourDegreeBGC(*args, **kwargs)
    simulation.setup()
    simulation.run()


if __name__ == '__main__':
    run()
