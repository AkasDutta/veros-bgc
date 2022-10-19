from types import NoneType
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

from loguru import logger


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
            logger.diagnostic(f"Recognised {key} as a rule")
            temp_rule = {}

            for att in rule.keys():
                parsed_att = prefix_parser(rule[att])
                temp_rule[att] = parsed_att

            ModelRules[key] = temp_rule

        else:
            raise ValueError(f"{key} is neither a rule nor the criteria.")

    ModelRules = rule_constructor(state, ModelRules)
    logger.diagnostic(f"ModelRules: {ModelRules.keys()}")
    return ModelRules


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
        foodweb.add_node(tracer)

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
                    if i == 0:
                        node = node.upper()
                    foodweb.add_node("*" + node)
                    lst[i] = "*" + node
        foodweb.add_node("*Bottom")

        # Add edge to foodweb:
        rule_edges = [(source, sink) for source in sources for sink in sinks]
        for edge in rule_edges:
            if edge[0].isupper() and edge[1].isupper():
                desc = rule.label
            else:
                desc = ""
            foodweb.add_edge(edge[0], edge[1], object=rule, label=desc)
        for node in foodweb.nodes:
            if hasattr(node, "sinking_speed"):
                foodweb.add_edge(node, "*Bottom", label="sinking")
                logger.diagnostic(f"{node.name} to *Bottom")

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
        for node in self.nodes:
            if type(node).__name__ in TracerClasses.keys():
                self.tracers[node.name] = node
                self.flags[node] = node.flag
                if node.transport:
                    self.transported_tracers.append(node)
                    self.npzd_advection_derivatives[node.name] = npx.zeros_like(
                        node.data
                    )
                if hasattr(node, "light_attenuation"):
                    self.light_attenuators.append(node)

                self.deposits[node.name] = npx.zeros_like(node.temp)
            else:
                self.flags[node] = 1

        for edge in self.edges:
            logger.diagnostic(edge)
            # self.rules += self.edges[edge]["object"]
        raise AttributeError()

        for rule in self.rules:
            if rule.group == "PRE":
                self.pre_rules.append(rule)
            if rule.group == "PRIMARY":
                self.primary_rules.append(rule)
            if rule.group == "POST":
                self.post_rules.append(rule)

    def summary(self):
        display = nx.MultiDiGraph
        nodes = [node.name for node in self.nodes]
        for node in nodes:
            display.add_node(node)
        edges = []
        for source, sink, data in self.edges(data=True):
            if data["label"] != "":
                display.add_edge(source, sink, obj=data["obj"])
        return display


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


@memoize
def get_foodweb(state):
    vs = state.variables
    settings = state.settings

    zw = vs.zw - vs.dzt  # bottom of grid box using dzt because dzw is weird
    dtr_speed = (
        settings.wd0 + settings.mw * npx.where(-zw < settings.mwz, -zw, settings.mwz)
    ) * vs.maskT
    foodweb = set_foodweb(state, dtr_speed)

    for tracer in foodweb.tracers:
        if isinstance(tracer, TracerClasses["Phytoplankton"]):
            foodweb.limiting_functions[tracer.name] = [
                phosphate_limitation_phytoplankton
            ]

    return foodweb
