
# external imports
import networkx as nx

# internal imports
from .World import *
from .GlobalPlanning import *

class AbstractionOptions:
    def __init__(self):
        self.onlyDirectConnections : bool = True

class GraphAbstraction:
    def __init__(
            self, 
            world : World, 
            gpp : GlobalPathPlanner, 
            opts : AbstractionOptions = None):
        self.graph : nx.DiGraph = None
        self.world: World = world
        self.options : AbstractionOptions = None
        self.setOptions(opts)
        self.abstract(world, gpp)

    def setOptions(self, opts : AbstractionOptions):
        if opts is None:
            self.options = AbstractionOptions()
            return
        if not isinstance(opts, AbstractionOptions):
            raise ValueError("opts must be of type AbstractionOptions")        
        self.options = opts

    def abstract(self, world : World, gpp : GlobalPathPlanner):
        self.graph = nx.MultiDiGraph()
        self.graph.add_nodes_from(world.targets())
        for t1 in world.targets():
            for t2 in world.targets():
                if t1 == t2:
                    continue
                path, time = gpp.planPathToTarget(t1.p(), t2)
                
                # No connection if no path found
                if path is None or time >= np.inf:
                    continue

                # No connection if we choose to drop indirect connections
                if self.options.onlyDirectConnections:
                    if not gpp.isDirectConnection(t1, t2, path):
                        continue

                # otherwise add the edge
                self.graph.add_edge(t1, t2, weight=time, path=path)

    def plotAbstraction(self, ax : plt.Axes = None, **kwargs):
        pos = nx.spring_layout(self.graph)
        for target in self.graph.nodes:
            pos[target] = target.p()
        
        # draw nodes
        node_labels = {}
        target : Target
        for target in self.graph.nodes:
            node_labels[target] = target.name
        nx.draw(self.graph, pos, ax=ax, labels=node_labels, with_labels=True, font_size=10)
        
        # draw edges
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edges(self.graph, pos, ax=ax)

    def simpleGraph(self) -> nx.Graph:
        g = nx.Graph()
        t: Target
        for t in self.graph.nodes:
            g.add_node(t.name)
        for u, v, data in self.graph.edges(data=True):
            g.add_edge(u.name, v.name, weight=data['weight'])
        return g
    