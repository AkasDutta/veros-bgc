"""
Contains veros methods for handling bio- and geochemistry
(currently only simple bio)
"""
from collections import namedtuple

from veros import veros_kernel, veros_routine, KernelOutput
from veros import time
from veros.core import diffusion, thermodynamics, utilities
from veros.core import isoneutral
from veros.core.operators import numpy as npx, at, update, update_add, update_multiply
from veros.variables import allocate


@veros_routine
def biogeochemistry(state):
    """Control function for integration of biogeochemistry

    Implements a rule based strategy. Any interaction between tracers should be
    described by registering a rule for the interaction.
    This routine ensures minimum tracer concentrations,
    calculates primary production, mortality, recyclability, vertical export for
    general tracers. Tracers should be registered to be used.
    """
    vs = state.variables
    settings = state.settings
    foodweb = settings.foodweb

    '''Preliminaries'''

    # Number of timesteps to do for bio tracers
    nbio = int(settings.dt_tracer // settings.dt_bio)

    # temporary tracer object to store differences
    for tracer in foodweb.tracers.values():
        #temporary_tracers[i] is of type array (a 3D snapshot) for all tracers i
        tracer.temp = tracer.reset_temp(state)
        tracer.flag = vs.maskT.astype(npx.bool)
    light_attenuators = foodweb.light_attenuators


    '''At the beginning of a tracer timestep, we evaluate all pre_rules'''

    # Pre rules: Changes that need to be applied before running npzd dynamics;
    #            stored as a list

    for pre_rule in settings.npzd_pre_rules:
        updates = pre_rule.call()
        boundary = pre_rule.boundary
        for key, value in updates.items():
            foodweb.tracers[key].temp = update_add(
                    foodweb.tracers[key].temp,
                    at[boundary],
                    value
                )
        
    '''
    Radiative considerations of PAR  
    '''

    # How much plankton is blocking light
    #  -> We actually iterate over all light-attenuating tracers; the name plankton
   
    #plankton_total = sum([plankton.data for plankton in light_attenuation_dict.keys()])\
     #                   * vs.dzt

    # Integrated light_attenuating plankton - starting from top of layer going upwards
    # reverse cumulative sum because our top layer is the last.
    # Needs to be reversed again to reflect direction
    
    integrated_plankton = {}
    for plankton in light_attenuators:
        
        integrated_plankton[plankton.name] = npx.empty_like(plankton.data) 
        integrated_plankton[plankton.name][:,:,:-1] = plankton.data[:,:,1:]
        integrated_plankton[plankton.name][:,:,-1] = 0.0
    
    plankton_total =  sum(plankton.data for plankton in light_attenuators)


    # incoming shortwave radiation at top of layer
    light_extinction = npx.empty_like(plankton_total)
    for plankton in light_attenuators:
        light_extinction = update_add(light_extinction, at[...],
            -plankton.light_attenuation*\
            npx.cumsum(integrated_plankton[plankton.name][:, :, ::-1], axis=2)[:,:,::-1]
        )

    swr = vs.swr[:, :, npx.newaxis] * light_extinction

    # Reduce incoming light where there is ice - as veros doesn't currently
    # have an ice model, we get temperatures below -1.8 and decreasing temperature forcing
    # as recommended by the 4deg model from the setup gallery
    icemask = npx.logical_and(vs.temp[:, :, -1, vs.tau] * vs.maskT[:, :, -1] < -1.8, vs.forc_temp_surface < 0.0)
    swr = update_multiply(swr, at[:,:], 
        npx.exp(-settings.light_attenuation_ice * icemask[:, :, npx.newaxis]))

    # declination and fraction of day with daylight
    # 0.72 is fraction of year at aphelion
    # 0.4 is scaling based on angle of rotation
    declin = npx.sin((npx.mod(vs.time * time.SECONDS_TO_X["years"], 1) - 0.72) * 2.0 * npx.pi) * 0.4
    rctheta = npx.maximum(-1.5, npx.minimum(1.5, npx.radians(vs.yt) - declin))

    # 1.33 is derived from Snells law for the air-sea barrier
    vs.rctheta[:] = settings.light_attenuation_water / npx.sqrt(1.0 - (1.0 - npx.cos(rctheta)**2.0) / 1.33**2)

    # fraction of day with photosynthetically active radiation with a minimum value
    dayfrac = npx.minimum(1.0, -npx.tan(npx.radians(vs.yt)) * npx.tan(declin))
    vs.dayfrac[:] = npx.maximum(1e-12, npx.arccos(npx.maximum(-1.0, dayfrac)) / npx.pi)

    # light at top of grid box
    grid_light = swr * npx.exp(vs.zw[npx.newaxis, npx.newaxis, :]
                              * vs.rctheta[npx.newaxis, :, npx.newaxis])

    # amount of PAR absorbed by water and plankton in each grid cell
    light_attenuation_plankton = sum([plankton.data*plankton.light_attenuation
                                for plankton in light_attenuators]
                                )
    light_attenuation = vs.dzt * settings.light_attenuation_water +\
                        light_attenuation_plankton



    # light-saturated growth and non-saturated growth
    jmax, avej = {}, {}
    for tracer in foodweb.tracers():

        # Calculate light-limited vs unlimited growth
        if hasattr(tracer, "potential_growth"):
            jmax[tracer.name], avej[tracer.name] = tracer.potential_growth(state, grid_light, light_attenuation)

        # Methods for internal use may need an update
        if hasattr(tracer, "update_internal"):
            tracer.update_internal(state)

    # bio loop
    for _ in range(nbio):

        # Plankton is recycled, dying and growing
        # pre compute amounts for use in rules
        # for plankton in vs.plankton_types:
        for tracer in foodweb.tracers.values():

            # Nutrient-limiting growth - if no limit, growth is determined by avej
            u = 1

            # limit maximum growth, usually by nutrient deficiency
            # and calculate primary production from that
            if hasattr(tracer, "potential_growth"):
                # NOTE jmax and avej are NOT updated within bio loop
                for growth_limiting_function in settings.limiting_functions[tracer.name]:
                    u = npx.minimum(u, growth_limiting_function(state, foodweb.tracers))

                tracer.net_primary_production = npx.minimum(avej[tracer.name], 
                                            u*jmax[tracer.name]) * tracer.data
            
            if hasattr(tracer, "grazing"):
                tracer.thetaZ = tracer.reset_thetaZ(state)

            #We want to shift everything to rules
            #Iterate over all rules in foodweb. Evaluate flags of tracer nodes. If flags
            #of all concerned tracers is not above minimum, kill the interaction.

            for rule in foodweb.rules:
                if  isinstance(rule.sink, str):
                    sinks = [rule.sink]
                elif isinstance(rule.sink, list):
                    sinks = rule.sink
                if isinstance(rule.source, str):
                    sources = [rule.source]
                elif isinstance(rule.source, list):
                    sources = [rule.source]
                rule.flag = vs.maskT
                for node in sources+sinks:
                    rule.flag = update_multiply(rule.flag, at[...], foodweb.flags[node])

                

            #Remove all calls of things like net_primary_production, recycling, mortality,
            #grazing to rules. Keep all bgc interactions there rather than cluttering up
            #this space.

            # Calculate concentration leaving cell and entering from above
            if hasattr(tracer, "sinking_speed"):
                # Concentration of exported material is calculated as fraction
                # of total concentration[:, :, ::-1], axis=2) which would have fallen through the bottom
                # of the cell (speed / vs.dzt * vs.dtbio)
                # vs.dtbio is accounted for later
                npzd_export = {}; npzd_import = {}
                npzd_export[tracer.name] = tracer.sinking_speed / vs.dzt * tracer * tracer.flag

                # Import is export from above scaled by the ratio of cell heights
                npzd_import[tracer.name] = npx.empty_like(npzd_export[tracer.name])
                npzd_import[tracer.name] = update(npzd_import[tracer.name], at[:,:,-1], 0)
                npzd_import[tracer.name] = update(npzd_import[tracer.name], at[:,:,:-1],
                                    npzd_export[tracer.name][:, :, 1:] *\
                                    (vs.dzt[1:] / vs.dzt[:-1]))

                # ensure we don't import in cells below bottom
                npzd_import[tracer.name] = update_multiply(npzd_import[tracer.name], at[...], vs.maskT)
                foodweb.deposits[tracer.name] = npzd_export[tracer.name]*vs.bottom_mask

        # Gather all state updates
        npzd_updates = [(rule.call(state), rule.boundary)
                        for rule in settings.npzd_rules]

        # perform updates
        for update, boundary in npzd_updates:
            for key, value in update.items():
                if isinstance(boundary, (tuple, slice)):
                    foodweb.tracers[key].temp = update_add(
                                        foodweb.tracers[key].temp, at[boundary],
                                        value * vs.dt_bio
                                        )
                else:
                    foodweb.tracers[key] = update_add(foodweb.tracers[key], at[...], 
                                        value * vs.dt_bio * boundary
                                        )

        # Import and export between layers
        # for tracer in vs.sinking_speeds:
        for tracer in foodweb.tracers.values():
            if hasattr(tracer, "sinking_speed"):
                tracer.temp = update_add(tracer.temp, at[:,:,:],
                (npzd_import[tracer.name]-npzd_export[tracer.name])*vs.dt_bio
                )
                
        # Prepare temporary tracers for next bio iteration
        for tracer in foodweb.tracers.values():
            tracer.flag = update(tracer.flag, at[:,:,:], 
                npx.logical_and(tracer.flag, (tracer.temp>vs.trcmin))
                )
            tracer.temp = update(tracer.temp, at[:,:,:], 
                utilities.where(state, tracer.flag, tracer.temp, vs.trcmin)
                )
  

    # Post processesing or smoothing rules
    post_results = [(rule.call(state), rule.boundary)
                    for rule in settings.npzd_post_rules]
    post_modified = []  # we only want to reset values, which have actually changed for performance

    for result, boundary in post_results:
        for key, value in result.items():
            foodweb.tracers[key].temp = update_add(foodweb.tracers[key].temp,
                                        at[boundary],
                                        value
                                        )
            post_modified.append(key)

    # Reset before returning
    # using set for unique modifications is faster than resetting all tracers
    for tracer_name in set(post_modified):
        tracer = foodweb.tracers[tracer_name]
        tracer.flag = update(
            tracer.flag,
            at[:,:,:],
            npx.logical_and(tracer.flag, (tracer.temp>vs.trcmin))
        )
        tracer.temp = update(
            tracer.temp,
            at[:,:,:],
            utilities.where(state, tracer.flag, tracer.temp, vs.trcmin)
        )

    # Only return the difference from the current time step. Will be added to timestep taup1
    return {tracer.name: tracer.temp - tracer.data[:, :, :, vs.tau]
            for tracer in foodweb.tracers.values()}


@veros_routine
def npzd(state):
    """
    Main driving function for NPZD functionality

    Computes transport terms and biological activity separately

    \begin{equation}
        \dfrac{\partial C_i}{\partial t} = T + S
    \end{equation}
    """
    vs = state.variables
    settings = state.settings
    foodweb = settings.foodweb
    
    
    if not settings.enable_npzd:
        return

    # TODO: Refactor transportation code to be defined only once and also used by thermodynamics
    # TODO: Dissipation on W-grid if necessary
    
    # common temperature factor determined according to b ** (cT)
    vs.bct = settings.bbio ** (settings.cbio * vs.temp[:, :, :, vs.tau])
    #npzd_changes = biogeochemistry(state)

    """
    For vertical mixing
    """

    a_tri = allocate(state.dimensions, ("xt", "yt", "zt"), include_ghosts=False)
    b_tri = allocate(state.dimensions, ("xt", "yt", "zt"), include_ghosts=False)
    c_tri = allocate(state.dimensions, ("xt", "yt", "zt"), include_ghosts=False)
    d_tri = allocate(state.dimensions, ("xt", "yt", "zt"), include_ghosts=False)
    delta = allocate(state.dimensions, ("xt", "yt", "zt"), include_ghosts=False)

    ks = vs.kbot[2:-2, 2:-2] - 1
    delta[:, :, :-1] = settings.dt_tracer / vs.dzw[npx.newaxis, npx.newaxis, :-1]\
        * vs.kappaH[2:-2, 2:-2, :-1]
    delta[:, :, -1] = 0
    a_tri[:, :, 1:] = -delta[:, :, :-1] / vs.dzt[npx.newaxis, npx.newaxis, 1:]
    b_tri[:, :, 1:] = 1 + (delta[:, :, 1:] + delta[:, :, :-1]) / vs.dzt[npx.newaxis, npx.newaxis, 1:]
    b_tri_edge = 1 + delta / vs.dzt[npx.newaxis, npx.newaxis, :]
    c_tri[:, :, :-1] = -delta[:, :, :-1] / vs.dzt[npx.newaxis, npx.newaxis, :-1]

    for tracer in foodweb.transported_tracers:
        tracer_data = tracer.data
        tr = tracer.name

        """
        Advection of tracers
        """
        settings.npzd_advection_derivatives[tr][:, :, :, vs.tau] = thermodynamics.advect_tracer(
                                                                  state, tracer_data[:, :, :, vs.tau])
                                     

        # Adam-Bashforth timestepping
        tracer_data[:, :, :, vs.taup1] = tracer_data[:, :, :, vs.tau] + settings.dt_tracer \
            * ((1.5 + settings.AB_eps) * settings.npzd_advection_derivatives[tr][:, :, :, vs.tau]
               - (0.5 + settings.AB_eps) * settings.npzd_advection_derivatives[tr][:, :, :, vs.taum1])\
            * vs.maskT

        """
        Diffusion of tracers
        """

        if settings.enable_hor_diffusion:
            horizontal_diffusion_change = npx.zeros_like(tracer_data[:, :, :, 0])
            horizontal_diffusion_change = diffusion.horizontal_diffusion(
                                       state, tracer_data[:, :, :, vs.tau], settings.K_h)

            tracer_data = update_add(tracer_data, at[:,:,:, vs.taup1],
                                     settings.dt_tracer * horizontal_diffusion_change)
            

        if settings.enable_biharmonic_mixing:
            biharmonic_diffusion_change = npx.empty_like(tracer_data[:, :, :, 0])
            biharmonic_diffusion_change = diffusion.biharmonic_diffusion(
                                          state, tracer_data[:, :, :, vs.tau],
                                          npx.sqrt(abs(settings.K_hbi))
                                          )
            
            tracer_data = update_add(tracer_data, at[:, :, :, vs.taup1],
                                    settings.dt_tracer*biharmonic_diffusion_change)

        """
        Restoring zones
        """
        # TODO add restoring zones to general tracers

        """
        Isopycnal diffusion
        """
        if settings.enable_neutral_diffusion:
            dtracer_iso = npx.zeros_like(tracer_data[..., 0])

            tracer_data = isoneutral.diffusion.isoneutral_diffusion_tracer(
                          state, tracer_data, dtracer_iso, iso=True, skew=False
                           )

            if settings.enable_skew_diffusion:
                dtracer_skew = npx.zeros_like(tracer_data[..., 0])
                tracer_data = isoneutral.diffusion.isoneutral_diffusion_tracer(
                              state, tracer_data, dtracer_skew, iso=False, skew=True
                               )
              
        """
        Vertical mixing of tracers
        """
        d_tri[:, :, :] = tracer_data[2:-2, 2:-2, :, vs.taup1]
        # TODO: surface flux?
        # d_tri[:, :, -1] += surface_forcing
        
        land_mask, water_mask, edge_mask = utilities.create_water_masks(ks)
        sol, mask = utilities.solve_implicit(a_tri, b_tri, c_tri, d_tri,
                                       water_mask, edge_mask, b_edge=b_tri_edge
                                      )

        tracer_data = update(
                             tracer_data,
                             at[2:-2, 2:-2, :, vs.taup1],
                             mask*sol + npx.logical_not(mask)*tracer_data[2:-2, 2:-2, :, vs.taup1]
                            )

    # update by biogeochemical changes

    #for tracer, change in npzd_changes.items():
    #    vs.npzd_tracers[tracer][:, :, :, vs.taup1] += change

    # prepare next timestep with minimum tracer values
    for tracer in settings.npzd_tracers.values():
        tracer.data[:, :, :, vs.taup1] = npx.maximum(tracer[:, :, :, vs.taup1], settings.trcmin * vs.maskT)

    for tracer in vs.npzd_tracers.values():
        utilities.enforce_boundaries(tracer.data)