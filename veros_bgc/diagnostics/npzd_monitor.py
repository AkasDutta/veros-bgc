from loguru import logger

from veros.diagnostics.base import VerosDiagnostic
from veros.core.operators import numpy as npx, update, at, update_add, update_multiply
from veros import veros_routine

from veros_bgc.core.foodweb import get_foodweb


class NPZDMonitor(VerosDiagnostic):
    """Diagnostic monitoring nutrients and plankton concentrations"""

    name = "npzd"  #:
    output_frequency = None  #: Frequency (in seconds) in which output is written
    restart_attributes = []
    save_graph = False  #: Whether or not to save a graph of the selected dynamics
    graph_attr = {  #: Properties of the graph (graphviz)
        "splines": "ortho",
        "center": "true",
        "nodesep": "0.05",
        "node": "square",
    }

    def __init__(self):
        self.output_variables = []
        self.surface_out = []
        self.bottom_out = []
        self.po4_total = 0
        self.dic_total = 0

    @veros_routine
    def initialize(self, state):
        vs = state.variables
        settings = state.settings
        foodweb =  get_foodweb(state)

        cell_volume = (
            vs.area_t[2:-2, 2:-2, npx.newaxis]
            * vs.dzt[npx.newaxis, npx.newaxis, :]
            * vs.maskT[2:-2, 2:-2, :]
        )
        
        po4_sum = foodweb.tracers["po4"].data[2:-2, 2:-2, :, vs.tau]
        #po4 is present in phyonly setup too

        if settings.enable_npzd: #biological tracers contain some P

            from core.npzd_tracers import Recyclable_tracer

            for tracer in foodweb.tracers:
                if isinstance(tracer, Recyclable_tracer):
                    po4_sum = update_add(
                        po4_sum,
                        at[2:-2,2:-2,:, vs.tau],
                        tracer.data*settings.Redfield_ratio_PN
                    )
        
        self.po4_total = npx.sum(po4_sum * cell_volume)

        if settings.enable_carbon:
            #dic only active in carbon setup

            dic_sum = foodweb.tracers["dic"].data[2:-2, 2:-2, :, vs.tau]
            
            from core.npzd_tracers import Recyclable_tracer

            for tracer in foodweb.tracers: #Bio tracers contain some C
                if isinstance(tracer, Recyclable_tracer):
                    dic_sum = update_add(
                        dic_sum,
                        at[2:-2,2:-2,:, vs.tau],
                        tracer.data*settings.Redfield_ratio_CN
                    )
            
            self.dic_total = npx.sum(dic_sum * cell_volume)

        

    def diagnose(self, state):
        pass

    @veros_routine
    def output(self, state):
        """
        Print NPZD interaction graph
        """
        vs = state.variables
        settings = state.settings

        foodweb = get_foodweb(state)

        # Will update the graph bit later
        if self.save_graph:
            from graphviz import Digraph

            npzd_graph = Digraph("npzd_dynamics", filename="npzd_dynamics.gv")
            label_prefix = "\\tiny "  # should be selectable in settings allows for better exporting to tex
            label_prefix = ""

            # Create a node for all selected tracers
            # Drawing edges also creates nodes, so this just ensures, we se it,
            # when there are no connections to a node
            
            display =  foodweb.summary()
            for node in display.nodes:
                if node[0] != "*":
                    npzd_graph.node(node)
                else:
                    npzd_graph.node(node, shape="square")
            
            for source, sink, data in display.edges(data=True):
                npzd_graph.edge(
                    source,
                    sink,
                    label = label_prefix + data["label"],
                    style = self.style(data["obj"]),
                    lblstyle="sloped, above",
                )


            self.save_graph = False
            npzd_graph.render("npzd_graph", view=False)

        """
        Total phosphorus should be (approximately) constant
        """
        cell_volume = (
            vs.area_t[2:-2, 2:-2, npx.newaxis]
            * vs.dzt[npx.newaxis, npx.newaxis, :]
            * vs.maskT[2:-2, 2:-2, :]
        )
        
        po4_sum = foodweb.tracers["po4"].data[2:-2, 2:-2, :, vs.tau]

        if settings.enable_npzd:

            from core.npzd_tracers import Recyclable_tracer

            for tracer in foodweb.tracers:
                if isinstance(tracer, Recyclable_tracer):
                    po4_sum = update_add(
                        po4_sum,
                        at[2:-2,2:-2,:, vs.tau],
                        tracer.data*settings.Redfield_ratio_PN
                    )

        if settings.enable_carbon:
            dic_sum = foodweb.tracers["dic"].data[2:-2, 2:-2, :, vs.tau]
            
            from core.npzd_tracers import Recyclable_tracer

            for tracer in foodweb.tracers:
                if isinstance(tracer, Recyclable_tracer):
                    dic_sum = update_add(
                        dic_sum,
                        at[2:-2,2:-2,:, vs.tau],
                        tracer.data*settings.Redfield_ratio_CN
                    )

        po4_total = npx.sum(po4_sum * cell_volume)
        logger.diagnostic(
            " total phosphorus: {}, relative change: {}".format(
                po4_total, (po4_total - self.po4_total) / self.po4_total
            )
        )
        self.po4_total = po4_total[...]

        if settings.enable_carbon:
            dic_total = npx.sum(dic_sum * cell_volume)
            logger.diagnostic(
                " total DIC: {}, relative change: {}".format(
                    dic_total, (dic_total - self.dic_total) / self.dic_total
                )
            )
            self.dic_total = dic_total[...]

        #for var in self.output_variables:
            #if var in vs.recycled:
            #    recycled_total = npx.sum(vs.recycled[var][2:-2, 2:-2, :] * cell_volume)
            #else:
            #    recycled_total = 0

            #if var in vs.mortality:
            #    mortality_total = npx.sum(
            #        vs.mortality[var][2:-2, 2:-2, :] * cell_volume
            #    )
            #else:
            #    mortality_total = 0

            #if var in vs.net_primary_production:
            #    npp_total = npx.sum(
            #        vs.net_primary_production[var][2:-2, 2:-2, :] * cell_volume
            #    )
            #else:
            #    npp_total = 0

            #if var in vs.grazing:
            #    grazing_total = npx.sum(vs.grazing[var][2:-2, 2:-2, :] * cell_volume)
            #else:
            #    grazing_total = 0

            #logger.diagnostic(" total recycled {}: {}".format(var, recycled_total))
            #logger.diagnostic(" total mortality {}: {}".format(var, mortality_total))
            #logger.diagnostic(" total npp {}: {}".format(var, npp_total))
            #logger.diagnostic(" total grazed {}: {}".format(var, grazing_total))

        for var in self.surface_out:
            logger.diagnostic(
                " mean {} surface concentration: {} mmol/m^3".format(
                    var, foodweb.tracers[var].data[vs.maskT[:, :, -1]].mean()
                )
            )

        for var in self.bottom_out:
            logger.diagnostic(
                " mean {} bottom concentration: {} mmol/m^3".format(
                    var, foodweb.tracers[var].data[vs.bottom_mask].mean()
                )
            )

    def read_restart(self, vs, infile):
        pass

    def write_restart(self, vs, outfile):
        pass

    def style(self, rule):
        styles = {"PRE": "dotted", "PRIMARY": "solid", "POST": "dashed"}
        return styles[rule.boundary]