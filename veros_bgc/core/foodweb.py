from veros.variables import Variable
from veros.core.operators import numpy as npx, update, at, update_add, update_multiply
import networkx as nx
from veros import veros_routine, veros_kernel, KernelOutput

from collections import namedtuple

from .npzd_tracers import TracerClasses
from .npzd_rules import RuleTemplates, Rule

@veros_routine
def ParseTracers(state, dtr_speed):
    vs = state.variables
    settings = state.settings

    tracers = settings._bgc_blueprint[0]
    ModelTracers = {}

    for tracer in tracers:
        Class = tracer["class"]
        Name  = tracer["name"]


        if Class not in TracerClasses.keys():
            raise ValueError(f"The tracer class {Class} does not exist.")

        if Name not in vars(vs).keys():
            raise ValueError(f"There is no veros_bgc variable {Name} corresponding to\
                 this tracer")


        cl = TracerClasses[Class]
        var = vars(vs)[Name]
        ModelTracers[Name] = cl(Name, var)

        #Check that attributes from bgc_blueprint are present in the tracer class
        UnknownAttributes = ""
        for key in tracer.keys():
            if key not in vars(cl):
                UnknownAttributes += f"{key}, "
        if UnknownAttributes != "":
            raise ValueError(f"{UnknownAttributes} is/are unknown attributes of {tracer[Class]}" )

        for key in tracer.keys():
            if key == "sinking_speed":
                vars(ModelTracers[Name])[key] = dtr_speed
            else:
                vars(ModelTracers[Name])[key] = tracer["key"]
        
        ModelTracers[Name].isvalid()
        #`isvalid()` will raise an error if a mandatory attribute is left blank; see 
        # `npzd_tracers.py`. This would end the loop.

    #All tracers have been initialised properly
    return ModelTracers

def PrefixParser(arg):
    ''' Maps strings beginning with "$" to variables bearing those names. Useful when loading
    data from a .yaml file '''
    if isinstance(arg, str):
        if arg[0] == "$":
            arg = arg[1:]
            parsed = globals()[arg]
    
    elif isinstance(arg, list):
        parsed = []
        for element in arg:
            parsed.append(PrefixParser(element))
    
    elif isinstance(arg, dict):
        parsed = {}
        for key in arg.keys():
            parsed_value = PrefixParser(arg[key])
            parsed[key] = parsed_value

    
    else:
        parsed = arg
    
    return parsed
@veros_routine
def ParseRules(state):
    vs = state.variables
    settings = state.settings

    rules = settings._bgc_blueprint[1]
    ModelRules = {}

    for key in rules.keys():
        if key == "criteria":
            criteria = PrefixParser(rules[key])
            for criterion in criteria:
                if criterion is False:
                    raise ValueError(f"The rules used in setup require that {criterion} be\
                        True, but it is set to False."
                        )
        
        elif isinstance(rules[key], dict):
            rule = rules[key]
            temp_rule = {}

            for att in rule.keys():
                parsed_att = PrefixParser(rule[att])
                temp_rule[att] = parsed_att

            ModelRules[key] = temp_rule

        else:
            raise ValueError(f"{key} is neither a rule nor the criteria.")
        
        ModelRules = RuleConstructor(state, ModelRules)
        return ModelRules
@veros_routine
def RuleConstructor(state, InputRules):

    OutputRules = {}

    for index in InputRules.keys:
        rule = InputRules[index]
        if not isinstance(rule["function"], str):
            func = rule["function"]
            raise TypeError(f"{func} is not a string. Did you perhaps provide the function\
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
            rule["boundary"]=None
        if "label" not in rule.keys():
            rule["label"]=None
        if "group" not in rule.keys():
            rule["group"]="Primary"
        
        OutputRules[index] = Rule(state,
                            index, 
                            rule["function"],
                            rule["arguments"], 
                            rule["source"], 
                            rule["sink"], 
                            rule["label"], 
                            rule["boundary"], 
                            rule["group"]
                        )
        return OutputRules
        

@veros_routine
def set_foodweb(state, dtr_speed):

    vs = state.variables
    settings = state.settings

    if not settings.enable_npzd:
        return

    foodweb = nx.MultiDiGraph()

    ModelTracers = ParseTracers(state, dtr_speed)

    ModelRules = ParseRules(state)


    #Created appropriated tracer objects using parameters . yaml file
    for tracer in ModelTracers:
        foodweb.add_node(tracer)

    for rule in ModelRules:
        #Create nodes not corresponding to tracers
        if rule.source[0] == "~":
            rule.source = rule.source[1:]
            foodweb.add_node("*" + rule.source)
        if rule.sink[0] == "~":
            rule.sink = rule.sink[1:]
            foodweb.add_node("*" + rule.sink)
        
        #Add to foodweb:
        foodweb.add_edge(rule.sink, rule.source, object = rule)
    
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
                    self.npzd_advection_derivatives[node.name] =\
                        npx.zeros_like(node.data)
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
    """ Nutrient limitation form for all nutrients """
    return nutrient.temp / (saturation_constant + nutrient.temp)


@veros_routine
def phosphate_limitation_phytoplankton(state, tracers):
    """ Phytoplankton limit to growth by phosphate limitation """
    vs = state.variables
    settings = state.settings

    return general_nutrient_limitation(tracers["po4"], settings.saturation_constant_N *\
         settings.redfield_ratio_PN)


@veros_routine
def get_foodweb(state):
        vs = state.variables
        settings = state.settings
        zw = vs.zw - vs.dzt  # bottom of grid box using dzt because dzw is weird
        dtr_speed = (vs.wd0 + vs.mw * npx.where(-zw < vs.mwz, -zw, vs.mwz)) \
        * vs.maskT
        foodweb = set_foodweb(state, dtr_speed)
        foodweb.limiting_functions["phytoplankton"] =\
             [phosphate_limitation_phytoplankton]
