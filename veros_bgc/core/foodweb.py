from types import NoneType
from veros.variables import Variable
from veros.core.operators import numpy as npx, update, at, update_add, update_multiply
import networkx as nx
import copy
from veros import logger, veros_routine, veros_kernel, KernelOutput

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


@veros_kernel
def parse_tracers(state, dtr_speed):
    vs = state.variables
    settings = state.settings

    path = settings.bgc_tracers_path
    with open(path) as f:
        tracers = list(yaml.load_all(f, yaml.SafeLoader))
    ModelTracers = dict({})

    for index, tracer in enumerate(tracers):
        Class = tracer["class"]
        Name = tracer["name"]

        if Class not in TracerClasses.keys():
            raise ValueError(f"The tracer class {Class} does not exist.")

        cl = TracerClasses[Class]
        var = vs.bgc_tracers[:, :, :, :, index]
        ModelTracers[Name] = cl(Name, var)
        vars(ModelTracers[Name])["index"] = index

        # Check that attributes from bgc_blueprint are present in the tracer class
        UnknownAttributes = ""
        for key in tracer.keys():
            if key == "class":
                continue
            if key not in cl.attrs:
                UnknownAttributes += key + ", "
            KnownAttributes = [term for term in cl.attrs]
        if UnknownAttributes != "":
            raise ValueError(
                f"{UnknownAttributes} is/are unknown attributes of {Class}. {KnownAttributes}"
            )

        for key in tracer.keys():
            if key in ["class", "name", "index"]:
                continue
            elif key == "sinking_speed":
                vars(ModelTracers[Name])[key] = dtr_speed
            else:
                vars(ModelTracers[Name])[key] = tracer[key]

        ModelTracers[Name].isvalid()
        logger.diagnostic(f"Tracer {Name} of class {Class} has been set up.")
        # `isvalid()` will raise an error if a mandatory attribute is left blank; see
        # `npzd_tracers.py`. This would end the loop.

    # All tracers have been initialised properly
    logger.diagnostic("All tracers initialised.")
    return ModelTracers


def prefix_parser(arg):
    """
    Input: string, list/dict containing such strings
    Output: (s/f,lambda fn): lambda from settings/foodweb to settings/tracer
            bearing the name given as input string. Prefix 's'/'f' indicates
            appropriate arg for lambda fn
            Lists, dicts returned with strings switched for such tuples
    Useful when loading data from a .yaml file
    """

    if isinstance(arg, str):
        # We want to check for strings which should actually be
        # names of objects in our namespace, not strings

        if arg[0] == "$":  # prefix for settings attributes
            arg = arg[1:]
            parsed = ("s", lambda settings: vars(settings)[arg])

        elif arg[0] == "^":  # prefix for tracers
            arg = arg[1:]
            parsed = ("f", lambda foodweb: foodweb.tracers[arg])
        else:
            parsed = arg
        # See npzd_rules.py, rule.call() for how this is used

    elif isinstance(arg, list):
        # Some arguments take a list of related entities,
        # eg: list of tracers produced during excretion
        parsed = []
        for element in arg:
            parsed.append(prefix_parser(element))

    elif isinstance(arg, dict):
        # Same logic as above, but for dictionaries
        parsed = {}
        for key in arg.keys():
            parsed_value = prefix_parser(arg[key])
            parsed[key] = parsed_value

    return parsed


@veros_kernel
def parse_rules(state):
    vs = state.variables
    settings = state.settings

    path = settings.bgc_rules_path
    with open(path) as f:
        rules = yaml.load(f, Loader=yaml.SafeLoader)
        if isinstance(rules, NoneType):  # An empty file, used to test only advection
            return {}  # We return no rules
    ModelRules = {}

    for key in rules.keys():
        # The first entry in rules.yml specifies binary criteria
        # for the code to run: for example, the carbon cycle being on

        if key == "criteria":
            criteria = prefix_parser(rules[key])
            for criterion in criteria:
                if not criterion[1](settings):
                    raise ValueError(
                        f"The rules used in setup require that {criterion} be\
                        True, but it is set to False."
                    )
            criteria_names = rules["criteria"]
            logger.diagnostic(f"{criteria_names} parsed")

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
    logger.diagnostic(f"The rules are: {list(ModelRules.keys())}")
    return ModelRules


@veros_kernel
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
            if el not in RuleTemplates[func][1]:
                raise KeyError(f"{func} does not take the argument {el}")

        if "boundary" not in rule.keys():
            rule["boundary"] = None
        if "label" not in rule.keys():
            rule["label"] = None
        if "group" not in rule.keys():
            rule["group"] = "PRIMARY"

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


@veros_kernel
def set_foodweb(state, dtr_speed):

    vs = state.variables
    settings = state.settings

    foodweb = nx.MultiDiGraph()

    ModelTracers = parse_tracers(state, dtr_speed)

    if not isinstance(ModelTracers, dict):
        raise TypeError(f"ModelTracers is {ModelTracers}")

    ModelRules = parse_rules(state)

    # Created appropriated tracer objects using parameters in.yaml file
    for tracer in ModelTracers.values():
        foodweb.add_node(tracer.name, tracer=tracer)
    foodweb.add_node("*Bottom")

    for rule in ModelRules.values():
        # Create nodes not corresponding to tracers
        if isinstance(rule.source, list):
            sources = rule.source
        else:
            sources = [rule.source]
        if isinstance(rule.sink, list):
            sinks = rule.sink
        else:
            sinks = [rule.sink]

        for lst in [sources, sinks]:
            for i, node in enumerate(lst):
                if node[0] == "~":
                    node = node[1:]
                    lst[i] = "*" + node

        # Add edge to foodweb:
        rule_edges = [(source, sink) for source in sources for sink in sinks]
        for edge in rule_edges:
            if edge[0] == sources[0] and edge[1] == sinks[0]:
                desc = rule.label  # The rule label is assigned to the
                # primary source-sink pair
                foodweb.add_edge(
                    edge[0],
                    edge[1],
                    object=rule,
                    label=desc,
                    name=rule.name,
                    key="to_display",
                )
            else:
                desc = rule.label
                foodweb.add_edge(edge[0], edge[1], object=None, label=desc, key="leak")

    foodweb = FoodWeb(foodweb)
    return foodweb


class FoodWeb(nx.MultiDiGraph):
    def __init__(self, foodweb):

        self.nodes = foodweb.nodes
        self.edges = foodweb.edges
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
        for node, tracer in list(self.nodes(data="tracer")):
            if tracer is not None:
                self.tracers[node] = tracer
                self.flags[node] = tracer.flag
                if tracer.transport:
                    self.transported_tracers.append(tracer)
                    self.npzd_advection_derivatives[node] = npx.zeros_like(tracer.data)
                if hasattr(tracer, "light_attenuation"):
                    self.light_attenuators.append(tracer)

                self.deposits[node] = npx.zeros_like(tracer.temp)
            else:
                self.flags[node] = 1

        for a, b, data in list(self.edges.data()):
            if data["object"] is not None:
                self.rules[data["name"]] = data["object"]

        for rule in self.rules.values():
            logger.diagnostic(f"Rule: {rule.name}")
            if rule.group == "PRE":
                self.pre_rules.append(rule)
            if rule.group == "PRIMARY":
                self.primary_rules.append(rule)
            if rule.group == "POST":
                self.post_rules.append(rule)
        logger.diagnostic(
            f"PRE:{self.pre_rules}; PRIMARY: {self.primary_rules};\
             POST: {self.post_rules}"
        )

    def summary(self):
        """
        Removes leaks to simplify display. Eg: Zooplankton
        grazing on phytoplankton takes up most food as biomass,
        but leaks some to detritus, po4, etc. Only show
        Phytoplankton->Zooplankton on the foodweb graph display
        for simplicity.
        """
        display = nx.MultiDiGraph()
        logger.diagnostic(list(self.nodes))
        for node in list(self.nodes):
            display.add_node(node)

        for edge in list(self.edges.data()):
            logger.diagnostic(edge[2])
            if "key" in edge[2].keys() and edge[2]["key"] == "to_display":
                display.add_edge(
                    edge[0],
                    edge[1],
                    object=edge[2]["object"],
                    label=edge[2]["label"],
                    name=edge[2]["name"],
                )

        return display


def general_nutrient_limitation(nutrient, saturation_constant):
    """Nutrient limitation form for all nutrients"""
    return nutrient.temp / (saturation_constant + nutrient.temp)


def phosphate_limitation_phytoplankton(state, tracers):
    """Phytoplankton limit to growth by phosphate limitation"""
    vs = state.variables
    settings = state.settings

    return general_nutrient_limitation(
        tracers["po4"], settings.saturation_constant_N * settings.redfield_ratio_PN
    )


@memoize
def get_foodweb(state):
    vs = state.variables
    settings = state.settings

    zw = vs.zw - vs.dzt  # bottom of grid box using dzt because dzw is weird
    dtr_speed = (
        settings.wd0 + settings.mw * npx.where(-zw < settings.mwz, -zw, settings.mwz)
    ) * vs.maskT
    foodweb = set_foodweb(state, dtr_speed)

    for tracer in foodweb.tracers.values():
        if isinstance(tracer, TracerClasses["Phytoplankton"]):
            foodweb.limiting_functions[tracer.name] = [
                phosphate_limitation_phytoplankton
            ]

    return foodweb
