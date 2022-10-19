"""
Classes for npzd tracers
"""
from types import NoneType
from unittest.mock import NonCallableMagicMock
from veros.core.operators import numpy as npx, update, at, update_add, update_multiply

from loguru import logger

from veros import veros_routine


class NPZD_tracer:
    """Class for npzd tracers to store additional information about themselves.

    Note
    ----
    Previously inhenrited from numpy.ndarray to make it work seamlessly with array operations
    No longer possible to do so with the Jax backend.

    Parameters
    ----------
    input_array : :obj:`numpy.ndarray`
        Numpy/JAX array backing data

    name : :obj:`str`
        Identifier for the tracer, which must be unique within a given configuration

    sinking_speed : :obj:`numpy.ndarray`, optional
        Numpy array for how fast the tracer sinks in each cell

    transport : :obj:`bool` = True, optional
        Whether or not to include the tracer in physical transport

    light_attenuation : :obj:`numpy.ndarray`, optional
        Factor for how much light is blocked

    Attributes
    ----------
    name
        Identifier for the tracer, which must be unique within a given configuration

    description
        Description of the tracer represented by the class

    transport
        Whether or not to include the tracer in physical transport

    sinking_speed : :obj:`numpy.ndarray`, optional
        If set: how fast the tracer sinks in each cell

    light_attenuation : :obj:`numpy.ndarray`, optional
        If set: Factor for how much light is blocked
    """

    _types = {
        "name": str,
        "index": int,
        "sinking_speed": npx.ndarray,
        "unit": str,
        "light_attenuation": float,
        "transport": bool,
        "description": str,
    }

    attrs = [key for key in _types.keys()]

    optional_attributes = ["sinking_speed", "light_attenuation"]

    def __new__(
        cls,
        name,
        input_array,
        index=None,
        sinking_speed=None,
        unit=None,
        light_attenuation=None,
        transport=True,
        description=None,
    ):

        obj = super().__new__(cls)

        if sinking_speed is not None:
            obj.sinking_speed = sinking_speed
        if light_attenuation is not None:
            obj.light_attenuation = light_attenuation

        obj.name = name
        obj.index = index
        obj.data = input_array
        obj.transport = transport
        obj.unit = unit
        if description is None:
            obj.description = ""
        else:
            obj.description = description
        obj.temp = npx.zeros_like(obj.data[:, :, :])
        obj.flag = True  # Defaults to true

        return obj

    @veros_routine
    def reset_temp(self, state):
        vs = state.variables
        settings = state.settings

        self.temp = self.data[:, :, :, vs.tau]

    def check_missing(self):
        Missing = ""
        if self.name is None:
            Missing += "name, "
        if self.data is None:
            Missing += "corresponding Variable array, "
        if self.index is None:
            Missing += "BGC_index (this error should be impossible), "
        if self.unit is None:
            Missing += "unit, "
        if self.transport is None:
            Missing += "transport, "
        if self.description is None:
            Missing += "description, "

        if hasattr(self, "light_attenuation") and self.light_attenuation is None:
            delattr(self, "light_attenuation")

        if hasattr(self, "sinking_speed") and self.sinking_speed is None:
            delattr(self, "sinking_speed")

        return Missing

    def check_types(self):
        for att in self.attrs:
            if att not in vars(self).keys():
                if att in self.optional_attributes:
                    break
                else:
                    raise KeyError(f"{att} not recognised.")
            if not isinstance(vars(self)[att], self._types[att]):
                raise TypeError(
                    f"{att} should be of type {self._types[att]}, not {type(vars(self)[att])}"
                )
            else:
                pass

    def isvalid(self):
        # TODO add TypeErrors
        name = self.name
        Missing = self.check_missing()
        if Missing != "":
            raise ValueError(f"{name}: The field(s) {Missing} are missing.")
        self.check_types()


#    def __array_finalize__(self, obj):
#        if obj is None:
#            return

# If we are slicing, obj will have __dir__ therefore we need to set attributes
# on new sliced array
#        if hasattr(obj, "__dir__"):
#            for attribute in (set(dir(obj)) - set(dir(self))):
#                setattr(self, attribute, getattr(obj, attribute))


class Calcite(NPZD_tracer):
    _types = NPZD_tracer._types.copy()
    attrs = [key for key in _types.keys()]
    # Not augmenting _types as dprca is set by the code, not the user.
    def __new__(cls, name, input_array, dprca=0, **kwargs):
        obj = super().__new__(cls, name, input_array, **kwargs)
        obj.dprca = dprca


class Recyclable_tracer(NPZD_tracer):
    """A recyclable tracer

    This would be tracer, which may be a tracer like detritus, which can be recycled

    Parameters
    ----------
    input_array : :obj:`numpy.ndarray`
        Numpy array backing data

    name : :obj:`str`
        Identifier for the tracer, which must be unique within a given configuration

    recycling_rate
        A factor scaling the recycling by the population size

    **kwargs
        All named parameters accepted by super class

    Attributes
    ----------
    recycling_rate
        A factor scaling the recycling by the population size

    + All attributes held by super class
    """

    _types = NPZD_tracer._types.copy()  # Can't use `super()` in cls attributes
    _types["recycling_rate"] = float

    attrs = [key for key in _types.keys()]

    def __new__(cls, name, input_array, recycling_rate=0.0, **kwargs):
        obj = super().__new__(cls, name, input_array)
        obj.recycling_rate = recycling_rate
        return obj

    @veros_routine
    def recycle(self, state):
        """
        Recycling is temperature dependant by :obj:`vs.bct`
        """
        vs = state.variables
        settings = self.settings

        return vs.bct * self.recycling_rate * self.temp

    def check_missing(self):
        Missing = super().check_missing()
        if self.recycling_rate is None:
            Missing += "recycling_rate, "
        return Missing


class Plankton(Recyclable_tracer):
    """Class for plankton object, which is both recyclable and displays mortality

    This class is intended as a base for phytoplankton and zooplankton and not
    as a standalone class

    Note
    ----
    Typically, it would desirable to also set light attenuation


    Parameters
    ----------
    input_array : :obj:`numpy.ndarray`
        Numpy array backing data

    name : :obj:`str`
        Identifier for the tracer, which must be unique within a given configuration

    mortality_rate
        Rate at which the tracer is dying in mortality method

    **kwargs
        All named parameters accepted by super class

    Attributes
    ----------
    mortality_rate
        Rate at which the tracer is dying in mortality method

    + All attributes held by super class
    """

    _types = Recyclable_tracer._types.copy()  # Can't use `super()` in cls attributes
    _types["mortality_rate"] = float
    _types["calcite_producing"] = bool

    attrs = [key for key in _types.keys()]

    def __new__(
        cls, name, input_array, mortality_rate=0.0, calcite_producing=False, **kwargs
    ):
        obj = super().__new__(cls, name, input_array, **kwargs)
        obj.mortality_rate = mortality_rate
        obj.calcite_producing = calcite_producing
        return obj

    @veros_routine
    def mortality(self, state):
        """
        The mortality rate scales linearly with population size
        """
        settings = state.settings
        return self.temp * self.mortality_rate

    def check_missing(self):
        Missing = super().check_missing()
        if self.mortality is None:
            Missing += "mortality, "
        return Missing


class Phytoplankton(Plankton):
    """Phytoplankton also has primary production

    Parameters
    ----------
    input_array : :obj:`numpy.ndarray`
        Numpy array backing data

    name : :obj:`str`
        Identifier for the tracer, which must be unique within a given configuration

    growth_parameter
        Scaling factor for maximum potential growth

    **kwargs
        All named parameters accepted by super class

    Attributes
    ----------
    growth_parameter
        Scaling factor for maximum potential growth

    + All attributes held by super class
    """

    _types = Plankton._types.copy()  # Can't use `super()` in cls attributes
    _types["growth_parameter"] = float

    attrs = [key for key in _types.keys()]

    def __new__(
        cls,
        name,
        input_array,
        growth_parameter=None,
        net_primary_production=None,
        **kwargs,
    ):

        obj = super().__new__(cls, name, input_array, **kwargs)
        obj.growth_parameter = growth_parameter

        return obj

    @veros_routine
    def potential_growth(self, state, grid_light, light_attenuation):
        """Light limited growth, not limited growth"""
        vs = state.variables
        settings = state.settings

        f1 = npx.exp(-light_attenuation)  # available light
        jmax = self.growth_parameter * vs.bct  # maximum growth
        gd = jmax * vs.dayfrac[npx.newaxis, :, npx.newaxis]  # growth in fraction of day
        avej = self._avg_J(
            state, f1, gd, grid_light, light_attenuation
        )  # light limited growth

        return jmax, avej

    @veros_routine
    def _avg_J(self, state, f1, gd, grid_light, light_attenuation):
        """Average light over a triuneral cycle

        Note
        ----
        This calculation is only valid if grid_light / gd < 20
        """
        vs = state.variables
        settings = state.settings

        u1 = npx.maximum(grid_light / gd, settings.u1_min)
        u2 = u1 * f1

        # NOTE: There is an approximation here: u1 < 20
        phi1 = npx.log(u1 + npx.sqrt(1 + u1**2)) - (npx.sqrt(1 + u1**2) - 1) / u1
        phi2 = npx.log(u2 + npx.sqrt(1 + u2**2)) - (npx.sqrt(1 + u2**2) - 1) / u2

        return gd * (phi1 - phi2) / light_attenuation

    def check_missing(self):
        Missing = super().check_missing()
        if self.growth_parameter is None:
            Missing += "growth_parameter, "
        return Missing


class Zooplankton(Plankton):
    """Zooplankton displays quadratic mortality rate but otherwise is similar to ordinary phytoplankton

    Parameters
    ----------
    input_array : :obj:`numpy.ndarray`
        Numpy array backing data

    name : :obj:`str`
        Identifier for the tracer, which must be unique within a given configuration

    max_grazing
        Scaling factor for maximum grazing rate

    grazing_saturation_constant
        Saturation in Michaelis-Menten

    grazing_preferences
        Dictionary of preferences for grazing on other tracers

    assimilation_efficiency
        Fraction of grazed material ingested

    growth_efficiency
        Fraction of ingested material resulting in growth

    maximum_growth_temperature : = 20
        Temperature in Celsius where increasing temperature no longer increases grazing

    **kwargs
        All named parameters accepted by super class

    Attributes
    ----------
    max_grazing
        Scaling factor for maximum grazing rate

    grazing_saturation_constant
        Saturation in Michaelis-Menten

    grazing_preferences
        Dictionary of preferences for grazing on other tracers

    assimilation_efficiency
        Fraction of grazed material ingested

    growth_efficiency
        Fraction of ingested material resulting in growth

    maximum_growth_temperature
        Temperature in Celsius where increasing temperature no longer increases grazing

    + All attributes held by super class
    """

    _types = Plankton._types.copy()  # Can't use `super()` in cls attributes
    _types["max_grazing"] = float
    _types["grazing_saturation_constant"] = float
    _types["grazing_preferences"] = dict
    _types["assimilation_efficiency"] = float
    _types["growth_efficiency"] = float
    _types["maximum_growth_temperature"] = float

    attrs = [key for key in _types.keys()]

    def __new__(
        cls,
        name,
        input_array,
        max_grazing=None,
        grazing_saturation_constant=None,
        grazing_preferences=None,
        assimilation_efficiency=None,
        growth_efficiency=None,
        maximum_growth_temperature=20.0,
        **kwargs,
    ):
        obj = super().__new__(cls, name, input_array, **kwargs)

        obj.max_grazing = max_grazing
        obj.grazing_saturation_constant = grazing_saturation_constant
        obj.grazing_preferences = grazing_preferences
        obj.assimilation_efficiency = assimilation_efficiency
        obj.growth_efficiency = growth_efficiency
        obj.maximum_growth_temperature = maximum_growth_temperature
        obj.thetaZ = 1
        obj._gmax = 0  # should be private

        return obj

    @veros_routine
    def update_internal(self, state):
        """
        Updates internal numbers, which are calculated only from Veros values
        """
        vs = state.variables
        settings = state.settings

        self._gmax = self.max_grazing * settings.bbio ** (
            settings.cbio
            * npx.minimum(self.maximum_growth_temperature, vs.temp[..., vs.tau])
        )

    @veros_routine
    def mortality(self, state):
        """
        Zooplankton is modelled with a quadratic mortality
        """
        return self.mortality_rate * self.temp**2

    @veros_routine
    def reset_thetaZ(self, state):
        settings = state.settings
        tracers = settings.foodweb.tracers

        return (
            sum(
                [
                    pref_score * tracers[preference].temp
                    for preference, pref_score in self.grazing_preferences.items()
                ]
            )
            + settings.saturation_constant_Z_grazing * settings.redfield_ratio_PN
        )

    @veros_routine
    def grazing(self, state, prey):
        """
                Zooplankton grazing on set preys

                Parameters
                ----------
                prey : any tracer that can be grazed upon
        .

                Returns
                -------
                Values for grazing, digestion, excretion and sloppy feeding corresponding to the prey
                species

                Note
                ----
                The result of this method is primarily useful in rules

                Note
                ----
                thetaZ is scaled by settings.redfield_ratio_PN. This may not be desirable in the
                general case
        """
        settings = state.settings

        ingestion = self.grazing_preferences[prey.name] / self.thetaZ

        grazing = self._gmax * ingestion * prey.temp * self.temp

        digestion = self.assimilation_efficiency * grazing

        sloppy_feeding = (1 - self.assimilation_efficiency) * grazing

        excretion = (1 - self.growth_efficiency) * digestion

        return grazing, digestion, excretion, sloppy_feeding

    def check_missing(self):
        Missing = super().check_missing()
        if self.max_grazing is None:
            Missing += "max_grazing, "
        if self.grazing_saturation_constant is None:
            Missing += "grazing_saturation_constant, "
        if self.assimilation_efficiency is None:
            Missing += "assimilation_efficiency, "
        if self.grazing_preferences is None:
            Missing += "grazing_preferences, "
        if self.growth_efficiency is None:
            Missing += "growth_efficiency, "
        if self.maximum_growth_temperature is None:
            Missing += "maximum_growth_temperature, "
        return Missing

    def isvalid(self):
        super().isvalid()
        if self.growth_efficiency < 0 or self.growth_efficiency > 1:
            raise ValueError("Growth efficiency should be in [0,1]")
        if self.assimilation_efficiency < 0 or self.assimilation_efficiency > 1:
            raise ValueError("Assimilation efficiency should be in [0,1]")


TracerClasses = {
    # The key HAS to be the exact name of the class.
    # Bug checks elsewhere are of the form type(tracer).__name__ in TracerClasses.keys()
    # So the name of the class, as a string, has to be exactly the key used in this dict
    "NPZD_tracer": NPZD_tracer,
    "Calcite": Calcite,
    "Recyclable_tracer": Recyclable_tracer,
    "Plankton": Plankton,
    "Phytoplankton": Phytoplankton,
    "Zooplankton": Zooplankton,
}
