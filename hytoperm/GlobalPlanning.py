
# external imports
import warnings
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Set
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_simulated_annealing

# internal imports
from .World import *
from .DataStructures import Tree, Node, PlotObject


class PlotOptions:
    def __init__(self):
        self.pbp : bool = False                                                 # plot best path
        self.pae : bool = False                                                 # plot all edge lines
        self.par : bool = False                                                 # plot active regions   
        self.psr : bool = False                                                 # plot search regions
        self._ael : PlotObject = PlotObject()                                   # all edge lines
        self._bpl : PlotObject = PlotObject()                                   # best path line
        self._arl : PlotObject = PlotObject()                                   # active region lines
        self._srl : PlotObject = PlotObject()                                   # search region lines
    
    # getters
    def allEdgeLines(self) -> PlotObject:
        return self._ael
    
    def bestPathLines(self) -> PlotObject:
        return self._bpl
    
    def activeRegionObjects(self) -> PlotObject:
        return self._arl
    
    def searchRegionObjects(self) -> PlotObject:
        return self._srl

    def plotAny(self) -> bool:
        return self.pbp or self.pae or self.par or self.psr

    # modifiers
    def toggleAllPlotting(self, bool) -> None:
        self.pbp = bool
        self.pae = bool
        self.par = bool
        self.psr = bool

    def addEdgeLines(self, po : PlotObject) -> None:
        self._ael.add(po)

    def addBestPathLine(self, po : PlotObject) -> None:
        self._bpl.add(po)

    def addActiveRegionObject(self, po : PlotObject) -> None:
        self._arl.add(po)

    def addSearchRegionObject(self, po : PlotObject) -> None:
        self._srl.add(po)


class RRBT:
    def __init__(
            self, 
            world : World,
            initTree : Tree
            ) -> None:
        self._active_regions : List[Region] = []                                # active regions
        self._world : World = None                                              # world
        self._T : Tree = None                                                   # tree
        self._targetDistances : np.ndarray = None                               # target distances

        # caching
        self._rttm : Dict[Region,Set[Tree]] = {}                                # region to node mapper

        # visualization
        self._plot_options = None                                               # a plot options instance

        # initialize
        self.initialize(world, initTree)

    # getters
    def getNodesInRegion(self, r : Region) -> List[Tree]:
        return self._rttm[r]

    def bestTravelRegion(
            self, 
            n0 : Node, 
            nf : Node, 
            regions : Set[Region]
            ) -> Tuple[float, Region]:
        best_tcp = np.inf
        best_region = None
        for region in regions:
            tcp = region.travelCost(n0.p(), nf.p())
            if tcp < best_tcp:
                best_tcp = tcp
                best_region = region
        return best_tcp, best_region
    
    def sampleActiveRegion(self) -> Region:
        idx = np.random.randint(0, len(self._active_regions))
        return self._active_regions[idx]

    def plotOptions(self) -> PlotOptions:
        return self._plot_options

    # modifiers   
    def initialize(self, world : World, initTree : Tree) -> None:
        if not isinstance(world, World):
            raise ValueError("World must be an instance of World")
        
        if not isinstance(initTree, Tree):
            raise ValueError("initTree must be an instance of Tree")
        
        self._world = world
        self._T = initTree
        self.initializeCache()           

    def expandTree(self, max_iter : int = 1000) -> None:
        for i in range(max_iter):
            newNode = self.sample()
            if self.extend(newNode) is None:
                warnings.warn("Could not extend tree. Continuing...")

    def sample(self) -> Node:
        r = self.sampleActiveRegion()
        sampleNodePos = r.randomBoundaryPoint()
        regions = self._world.getRegions(sampleNodePos)
        return Node(sampleNodePos, regions)
        
    def extend(self, node : Node) -> Tree:
        bestCTR = np.inf
        best_parent = None
            
        cost_to_best_parent = np.inf
        region_to_parent = None
        for shared_region in node.regions():
            for parent in self.getNodesInRegion(shared_region):
                cost_to_parent = shared_region.travelCost(
                    node.p(), parent.getData().p()
                )
                cost_to_root = parent.getData().costToRoot() + cost_to_parent
                if cost_to_root < bestCTR:
                    bestCTR = cost_to_root
                    cost_to_best_parent = cost_to_parent
                    best_parent = parent
                    region_to_parent = shared_region

        # Sanity check that we sampled from a feasible region
        if best_parent is None:
            return None

        # connect the node to the tree
        sampleTree = Tree(node)
        self.connect(
            sampleTree, 
            best_parent, 
            cost_to_best_parent, 
            region_to_parent
        )
        
        # update cache
        for r in node.regions():
            self._rttm[r].add(sampleTree)
        self.activateRegionsContaining(node.p())
        
        return sampleTree
    
    def planPath(self, t0) -> Tuple[Tree, float]:
        
        initialRegions = self._world.getRegions(t0)
        initialNode = Node(t0, initialRegions)
        
        initTree = self.extend(initialNode)
        if initTree is None:
            warnings.warn("Could not find a path from initial node to tree. Returning None. Did you run 'expandTree' with a sufficient number of iterates?")
            return None, np.inf
        
        best_cost = initTree.getData().costToRoot()
        return self.extractPath(initTree), best_cost

    def extractPath(self, T : Tree) -> Tree:
        # build queue from leaf to root
        queue = [T]
        active = T
        while active.hasParent():
            active = active.getParent()
            queue.insert(0, active)
        
        # build path from root to leaf
        p = Tree(queue[0].getData())
        for i in range(1, len(queue)):
            qi = queue[i]
            pi = qi.getData()
            ti = Tree(pi)
            self.connect(
                child=ti, 
                parent=p, 
                cost_to_parent=qi.getData().costToParent(), 
                rtp=qi.getData().activeRegionToParent()
            )
            p = ti

        return p

    def connect(
            self, 
            child : Tree, 
            parent : Tree, 
            cost_to_parent : float, 
            rtp : Region
            ) -> None:
        child.setParent(parent, cost_to_parent)
        if rtp is not None:
            child.getData().activate_region_to_parent(rtp)
  
    def rewireInitNode(self) -> None:
        initRegions = self.best_path.getData().regions()
        improved = False
        for initRegion in initRegions:
            for parent in self.getNodesInRegion(initRegion):
                bpp = self.best_path.getData().p()
                rewireCost = initRegion.travelCost(bpp, parent.getData().p())
                if rewireCost + parent.getData().costToRoot() < self.best_cost:
                    self.connect(self.best_path, parent, rewireCost, initRegion)
                    self.best_cost = self.best_path.getData().costToRoot()
                    improved = True
        if improved:
            self.optimizeSwitchingPoints(self.best_path)
            if self.plotOptions().pbp:
                self.plotBestPath()

    def rewire(self, sampleTree : Tree) -> None:
        initRegion = self.best_path.getData().activeRegionToParent()
        bpp = self.best_path.getData().p()
        rewireCost = initRegion.travelCost(bpp, sampleTree.getData().p())
        if rewireCost + sampleTree.getData().costToRoot() < self.best_cost:
            self.connect(self.best_path, sampleTree, rewireCost)
            self.plotBestPath()
    
    def initializeCache(self) -> None:
        for r in self._world.regions():
            self._rttm[r] = set()
        queue = [self._T.getRoot()]
        while queue:
            active = queue.pop(0)
            for r in active.getData().regions():
                self._rttm[r].add(active)
                self.activateRegion(r)
            queue.extend(active.getChildren())

    def clearEdgeLines(self) -> None:
        self.plotOptions().allEdgeLines().remove()

    def activateRegion(self, r : Region) -> None:
        if r not in self._active_regions:
            self._active_regions.append(r)

    def activateRegionsContaining(self, p : np.ndarray) -> None:
        for r in self._world.regions():
            if r in self._active_regions:
                continue
            if r.contains(p):
                self._active_regions.append(r)

    # plotters
    def plotPath(self, ax : plt.Axes = None) -> PlotObject:
        ax = getAxes(ax)
        return self.best_path.plotPathToRoot()
    
    def visualizeActiveRegions(self, ax : plt.Axes = None) -> None:
        ax = getAxes(ax)
        self.plotOptions().activeRegionObjects().remove()
        for r in self._active_regions:
            self.plotOptions().addActiveRegionObject(
                r.fill(ax, color = 'green', alpha = 0.2)
            )

    def visualizeSearchRegions(
            self, 
            p : np.ndarray, 
            regions : Set[Region], 
            ax : plt.Axes = None
            ) -> None:
        ax = getAxes(ax)
        self.plotOptions().searchRegionObjects().remove()
        referencePoint = PlotObject(plt.plot(p[0], p[1], 'gd'))
        self.plotOptions().addSearchRegionObject(referencePoint)
        regionArgs = {'color':'blue','alpha':0.2}
        for r in regions:   
            self.plotOptions().addSearchRegionObject(r.fill(ax, **regionArgs))

    def plotBestPath(self, ax : plt.Axes = None) -> None:
        ax = getAxes(ax)
        self.plotOptions().bestPathLines().remove()
        self.plotOptions().addBestPathLine(
            self.best_path.plotPathToRoot(None, ax, color = 'red', linewidth=2)
        )

    def plotAllEdgeLines(self, T : Tree, ax : plt.Axes = None) -> None:
        ax = getAxes(ax)
        self.plotOptions().allEdgeLines().remove()
        queue = [T.getRoot()]
        while len(queue) > 0:
            n = queue.pop(0)
            for c in n.getChildren():
                queue.append(c)
            
            if n.getParent() is not None:
                el = n.plotPathToParent(
                    ax, color = 'black', linewidth = 1, alpha = 0.5
                )
                self.plotOptions().addEdgeLines(el)


class TSP:
    def __init__(self, targets : List[Target]) -> None:
        self._targets = targets
        self._targetDistances = np.zeros((len(targets), len(targets)))
        self._best_permutation = None
        self._best_distance = np.inf

    # getters       
    def getTargetVisitingSequence(self) -> List[Target]:
        tvs = []
        for p in self._best_permutation:
            tvs.append(self._targets[p])
        return tvs
    
    def targetDistances(self) -> np.ndarray:
        return self._targetDistances
    
    def bestPermutation(self) -> List:
        return self._best_permutation
    
    def bestDistance(self) -> float:
        return self._best_distance

    # setters
    def setTargetDistance(self, i : int, j : int, d : float) -> None:
        self._targetDistances[i,j] = float(d)

    # modifiers
    def removeTargets(self, indices : List[int]) -> None:
        print("Removing {0} invalid targets".format(len(indices)))
        self._targets = []
        for i in range(len(self._targets)):
            if i not in indices:
                self._targets.append(self._targets[i])
        self._targetDistances = np.delete(self._targetDistances,indices,axis=0)
        self._targetDistances = np.delete(self._targetDistances,indices,axis=1)
                
    def computeTSP(self, exact = True) -> Tuple[List, float]:
        if exact:
            p, d = solve_tsp_dynamic_programming(self._targetDistances)
        else:
            p, d = solve_tsp_simulated_annealing(self._targetDistances)
        
        if d < self._best_distance:
            self._best_distance = d
            self._best_permutation = p
        
        return p, d

    # plotters    
    def plotTargetDistances(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        for i in range(self._targetDistances.shape[0]):
            for j in range(self._targetDistances.shape[1]):
                if i == j:
                    continue
                p = self._targets[i].p()
                q = self._targets[j].p()
                delta = -(q - p)/np.linalg.norm(q-p)
                po.add(ax.plot([p[0], q[0]], [p[1],  q[1]], **kwargs))
                
                po.add(ax.quiver(p[0], p[1], delta[0], delta[1], 
                                 pivot='tip', angles='xy', **kwargs)
                )
                
                po.add(ax.annotate(f"{self._targetDistances[i,j]:.2f}", 
                                   ((p[0]+q[0])/2, (p[1]+q[1])/2), 
                                   fontsize=12, 
                                   color='black')
                )
        return po


class GlobalPathPlanner:
    def __init__(self, world : World) -> None:
        self._world = world
        self._tsp : TSP = None
        self._rrbts : Dict[Target, RRBT] = {}
        self._target_paths : Dict[Target, Dict[Target, Tree]]= {}
        self._plot_options = PlotOptions()
        self._have_graph = False

    # getters
    def tsp(self) -> TSP:
        return self._tsp
    
    def targetPath(self, init : Target, goal : Target) -> Tree:
        if init not in self._target_paths:
            warnings.warn("No path exists from {0} to any other target. Running TSP solver".format(init.name))
            self.solveTSP()
            if goal not in self._world.targets():
                warnings.warn("No path exists from {0} to any other target (even after utilizing TSP solver). Returning None".format(init.name, goal.name))
                return None
        if goal not in self._target_paths[init]:
            warnings.warn("No path exists from {0} to {1}. Running TSP solver".format(init.name, goal.name))
            self.solveTSP()
            if goal not in self._world.targets():
                warnings.warn("No path exists from {0} to {1} (even after utilizing TSP solver). Returning None".format(init.name, goal.name))
                return None
        return self._target_paths[init][goal]

    # modifiers
    def planPath(
            self, 
            t0 : np.ndarray, 
            tf : np.ndarray, 
            max_iter = 500
            ) -> Tuple[Tree, float]:
        
        # Switch to local planner if possible
        initialRegions = self._world.getRegions(t0)
        targetRegions = self._world.getRegions(tf)
        for i_reg in initialRegions:
            for t_reg in targetRegions:
                if i_reg == t_reg:
                    return i_reg, i_reg.planPath(t0, tf)
                
        # utilize target RRBT if possible
        for target in self._world.targets():
            if np.linalg.norm(target.p() - tf) < 1e-3:
                return self.planPathToTarget(t0, target, max_iter)
        
        # otherwise, need to build a new RRBT
        root = Tree(Node(tf, targetRegions))
        rrbt = RRBT(self._world, root)
        rrbt._plot_options = self._plot_options
        rrbt.expandTree(max_iter=max_iter)
        return rrbt.planPath(t0)

    def planPathToTarget(
            self,
            init : np.ndarray,
            goal : Target,
            max_iter : int = 1000
            ) -> Tuple[Tree, float]:
        if not goal in self._rrbts:
            targetpos = goal.p()
            root = Tree(Node(targetpos, self._world.getRegions(targetpos)))
            self._rrbts[goal] = RRBT(self._world, root)
            self._rrbts[goal].expandTree(max_iter=max_iter)
        return self._rrbts[goal].planPath(init)

    def solveTSP(self) -> None:
        if self._tsp is None:
            self._tsp = TSP(self._world.targets())
        if not self._have_graph:
            self.generateCompleteGraph()
        self._tsp.computeTSP()

    def generateCompleteGraph(self) -> None:
        for i in range(self._world.nTargets()):
            target_i = self._world.targets()[i]
            self._target_paths[target_i] = {}
            for j in range(self._world.nTargets()):
                if i == j:
                    self._tsp.setTargetDistance(i,j,0)
                    continue
                target_j = self._world.targets()[j]
                plannedPath = self.planPathToTarget(target_i.p(), target_j)
                self._target_paths[target_i][target_j] = plannedPath[0] 
                self._tsp.setTargetDistance(i,j,plannedPath[1])
                print(f"Distance from {i} to {j} is {plannedPath[1]}")
        self._have_graph = True

    # plotters    
    def plotTSPSolution(
            self, 
            ax : plt.Axes = None, 
            annotate = False, 
            **kwargs
            ) -> PlotObject:
        ax = getAxes(ax)
        if self._tsp.bestPermutation() is None:
            print("No TSP solution exists. Please run 'solveTSP' first.")
            return None
        
        if self._tsp.bestPermutation() is None:
            return
        po = PlotObject()
        args = kwargs.copy()
        for i in range(0,len(self._tsp.bestPermutation())):
            currTarget = self._world.targets()[self._tsp.bestPermutation()[i-1]]
            nextTarget = self._world.targets()[self._tsp.bestPermutation()[i]]
            currPath = self._target_paths[currTarget][nextTarget]
            po.add(currPath.plotPathToRoot(ax=ax, plot_direction=True, **args))
            currPath = currPath.getParent()
            
            if annotate:
                po.add(ax.annotate(
                    f"{i}", 
                    (currPath.getData().p()[0], currPath.getData().p()[1]), 
                    fontsize=12, color='black')
                )

        return po
