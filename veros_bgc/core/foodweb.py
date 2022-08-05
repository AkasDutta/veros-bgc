from veros.variables import Variable
from veros.core.operators import numpy as npx, update, at, update_add, update_multiply
import networkx as nx
from veros import veros_routine, veros_kernel, KernelOutput

from collections import namedtuple
import ruamel.yaml as yaml
from yaml.loader import SafeLoader
from .npzd_tracers import TracerClasses
from .npzd_rules import RuleTemplates, Rule
import functools


def memoize(func):
    """
    Stores the result of func to avoid rework if func is invoked again

    Warning: will provide the same result even if the input to func changes
    """
    func.cache = {}

    @functools.wraps(func)
    def inner(*args):
        if args not in func.cache.keys():
            func.cache[args] = func(*args)

        return func.cache[args]

    return inner


@veros_routine
def parse_tracers(state, dtr_speed):
    vs = state.variables
    settings = state.settings

    path = settings._bgc_tracers_path
    with open(path) as f:
        tracers = yaml.safe_load(f)
    ModelTracers = {}

    for index, tracer in enumerate(tracers):
        Class = tracer["class"]
        Name = tracer["name"]

        if Class not in TracerClasses.keys():
            raise ValueError(f"The tracer class {Class} does not exist.")

        cl = TracerClasses[Class]
        var = vs.bgc_tracers[:, :, :, :, index]
        ModelTracers[Name] = cl(Name, var)
        vars(ModelTracers[Name])[index] = index

        # Check that attributes from bgc_blueprint are present in the tracer class
        UnknownAttributes = ""
        for key in tracer.keys():
            if key not in vars(cl):
                UnknownAttributes += f"{key}, "
        if UnknownAttributes != "":
            raise ValueError(
                f"{UnknownAttributes} is/are unknown attributes of {tracer[Class]}"
            )

        for key in tracer.keys():
            if key == "sinking_speed":
                vars(ModelTracers[Name])[key] = dtr_speed
            else:
                vars(ModelTracers[Name])[key] = tracer["key"]

        ModelTracers[Name].isvalid()
        # `isvalid()` will raise an error if a mandatory attribute is left blank; see
        # `npzd_tracers.py`. This would end the loop.

    # All tracers have been initialised properly
    return ModelTracers


def prefix_parser(arg):
    """Maps strings beginning with "$" to variables bearing those names. Useful when loading
    data from a .yaml file"""
    if isinstance(arg, str):
        #We want to check for strings which should actually be
        #names of objects in our namespace, not strings

        if arg[0] == "$": #prefix for state attributes
            arg = arg[1:]
            parsed = ("s", lambda state: vars(state)[arg])

        elif arg[0] == "^":#prefix for tracers
            arg = arg[1:]
            parsed = ("f", lambda foodweb: foodweb.tracers[arg])
        #See npzd_rules.py, rule.call() for how this is used

    elif isinstance(arg, list):
        #Some arguments take a list of related entities,
        # eg: list of tracers produced during excretion
        parsed = []
        for element in arg:
            parsed.append(prefix_parser(element))

    elif isinstance(arg, dict):
        #Same logic as above, but for dictionaries
        parsed = {}
        for key in arg.keys():
            parsed_value = prefix_parser(arg[key])
            parsed[key] = parsed_value

    else:
        #if we're using a string qua string, not as 
        #the name of an object 
        parsed = arg
    return parsed


@veros_routine
def parse_rules(state):
    vs = state.variables
    settings = state.settings

    path = settings._bgc_rules_path
    with open(path) as f:
        rules = yaml.safe_load(f)
    ModelRules = {}

    for key in rules.keys():
        #The first entry in rules.yml specifies binary criteria
        #for the code to run: for example, the carbon cycle being on
        if key == "criteria":
            criteria = prefix_parser(rules[key])
            for criterion in criteria:
                if criterion is False:
                    raise ValueError(
                        f"The rules used in setup require that {criterion} be\
                        True, but it is set to False."
                    )

        elif isinstance(rules[key], dict):
            rule = rules[key]
            temp_rule = {}

            for att in rule.keys():
                parsed_att = prefix_parser(rule[att])
                temp_rule[att] = parsed_att

            ModelRules[key] = temp_rule

        else:
            raise ValueError(f"{key} is neither a rule nor the criteria.")

        ModelRules = rule_constructor(state, ModelRules)
        return ModelRules


@veros_routine
def rule_constructor(state, InputRules):

    OutputRules = {}

    for index in InputRules.keys():
        rule = InputRules[index]
        if not isinstance(rule["function"], str):
            func = rule["function"]
            raise TypeError(
                f"{func} is not a string. Did you perhaps provide the function\
                object instead of the key mapping to it in RuleTemplates?"
            )
        if rule["function"] not in RuleTemplates.keys():
            func = rule["function"]
            raise ValueError(f"{func} not found. Check RuleTemplates.")

        for el in list(rule["arguments"].keys()):
            func = rule["function"]
            if el not in RuleTemplates["function"][1]:
                raise KeyError(f"{func} does not take the argument {el}")

        if "boundary" not in rule.keys():
            rule["boundary"] = None
        if "label" not in rule.keys():
            rule["label"] = None
        if "group" not in rule.keys():
            rule["group"] = "Primary"

        OutputRules[index] = Rule(
            state,
            index,
            rule["function"],
            rule["arguments"],
            rule["source"],
            rule["sink"],
            rule["label"],
            rule["boundary"],
            rule["group"],
        )
        return OutputRules


@memoize
@veros_routine
def set_foodweb(state, dtr_speed):

    vs = state.variables
    settings = state.settings

    foodweb = nx.MultiDiGraph()

    ModelTracers = parse_tracers(state, dtr_speed)

    ModelRules = parse_rules(state, ModelTracers)

    # Created appropriated tracer objects using parameters . yaml file
    for tracer in ModelTracers:
        foodweb.add_node(tracer)

    for rule in ModelRules:
        # Create nodes not corresponding to tracers
        if rule.source[0] == "~":
            rule.source = rule.source[1:]
            foodweb.add_node("*" + rule.source)
        if rule.sink[0] == "~":
            rule.sink = rule.sink[1:]
            foodweb.add_node("*" + rule.sink)

        # Add to foodweb:
        foodweb.add_edge(rule.sink, rule.source, object=rule)

    foodweb = FoodWeb(foodweb, state)


class FoodWeb(nx.MultiDiGraph):
    def __init__(self):

        self.nodes = super().nodes
        self.edges = super().edges
        self.tracers = {}
        self.transported_tracers = []
        self.npzd_advection_derivatives = {}
        self.light_attenuators = []
        self.flags = {}
        self.deposits = {}
        self.rules = {}
        self.pre_rules = []
        self.post_rules = []
        self.primary_rules = []
        self.limiting_functions = {}
        for node in self.nodes:
            if type(node).__name__ in TracerClasses.keys():
                self.tracers[node.name] = node
                self.flags[node] = node.flag
                if node.transport:
                    self.transported_tracers.append(node)
                    self.npzd_advection_derivatives[node.name] = npx.zeros_like(
                        node.data
                    )
                if node.light_attenuation is not None:
                    self.light_attenuators.append(node)

                self.deposits[node.name] = npx.zeros_like(node.temp)
            else:
                self.flags[node] = 1

        for edge in self.edges:
            self.rules += edge.object
        for rule in self.rules:
            if rule.group == "PRE":
                self.pre_rules.append(rule)
            if rule.group == "PRIMARY":
                self.primary_rules.append(rule)
            if rule.group == "POST":
                self.post_rules.append(rule)


def general_nutrient_limitation(nutrient, saturation_constant):
    """Nutrient limitation form for all nutrients"""
    return nutrient.temp / (saturation_constant + nutrient.temp)


@veros_routine
def phosphate_limitation_phytoplankton(state, tracers):
    """Phytoplankton limit to growth by phosphate limitation"""
    vs = state.variables
    settings = state.settings

    return general_nutrient_limitation(
        tracers["po4"], settings.saturation_constant_N * settings.redfield_ratio_PN
    )


@veros_routine
def get_foodweb(state):
    vs = state.variables
    settings = state.settings
    zw = vs.zw - vs.dzt  # bottom of grid box using dzt because dzw is weird
    dtr_speed = (vs.wd0 + vs.mw * npx.where(-zw < vs.mwz, -zw, vs.mwz)) * vs.maskT
    foodweb = set_foodweb(state, dtr_speed)
    foodweb.limiting_functions["phytoplankton"] = [phosphate_limitation_phytoplankton]
