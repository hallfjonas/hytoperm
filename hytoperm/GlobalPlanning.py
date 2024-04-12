
# external imports
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


class RRT:
    def __init__(
            self, 
            regions : Set[Region], 
            best_cost : float = np.inf
            ) -> None:
        self.regions : Set[Region] = regions                                    # all regions
        self.best_path : Tree = None                                            # path
        self.best_cost : float = best_cost                                      # best cost
        self.active_regions : List[Region] = []                                 # active regions
        self.world = World()                                                    # world
        self._targetDistances : np.ndarray = None                               # target distances
        
        self.world.setRegions(self.regions)                                     # set regions

        # caching
        self._rttm : Dict[Region,Set[Tree]] = {}                                # region to node mapper

        # convergence
        self._max_iter = 1000                                                   # maximum iterations

        # visualization
        self._plot_options = None                                               # a plot options instance

        self._optimize = False
        self._rewire = False                                                    # rewire
        self._cut = False                                                       # cut

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
        idx = np.random.randint(0, len(self.active_regions))
        return self.active_regions[idx]

    def plotOptions(self) -> PlotOptions:
        return self._plot_options

    # modifiers
    def planPath(
            self, 
            t0, 
            tf,
            ax : plt.Axes = None
            ) -> Tuple[Set[Region], List[np.ndarray]]:
        
        targetRegions = self.world.getRegions(tf)
        initialRegions = self.world.getRegions(t0)
        targetNode = Node(tf, targetRegions)
        initialNode = Node(t0, initialRegions)

        if self.plotOptions().plotAny():
            targs = {'color':'orange','marker':'*','markersize':10,'label':'xf'}
            targetNode.plot(ax, **targs)
            iargs = {'color':'green','marker':'o','markersize':10,'label':'x0'}
            initialNode.plot(ax, **iargs)
            
        # initialize tree (either with empty root or with an initial path)
        T = Tree(data = targetNode)
        
        try:
            initTree = self.initializePath(T, initialNode)
        except Exception as e:
            print("No initial path found...")

        if initTree is not None:
            self.best_cost = initTree.getData().costToRoot()
            self.best_path = initTree

            if self.plotOptions().pbp:
                self.plotBestPath(ax)

            self.optimizeSwitchingPoints(initTree)

            T = initTree.getRoot()


        self.initializeRegionToTreeMapper(T)
        self.initializeActiveRegions(initTree)

        # while loop [without branch and cut]
        # 1. draw a random active region
        # 2. draw a random node of the active region:
        # 2.1 if region is initial region, add the target position as a new node
        # 2.2 otherwise, draw random node on the boundary of the current region
        # 3. connect the sampled node with the tree:
        # 3.1 for each node in the neighborhood (region) of the sampled node:
        # 3.1.1 compute travel cost to root going through that parent node
        # 3.2 connect sampled node with parent node minimizing cost to root
        # 4. activate all regions bordering the sampled node

        iter = 0
        while (iter < self._max_iter):
            iter += 1
            
            # 1. sample a random active region
            r = self.sampleActiveRegion()
                
            # 2. draw a random node of the active region:
            sampleNodePos = r.randomBoundaryPoint()
            regions = self.world.getRegions(sampleNodePos)
            sampleNode = Node(sampleNodePos, regions)
            sampleBordersInit = len(initialRegions.intersection(regions)) > 0

            if self.plotOptions().psr:
                self.visualizeSearchRegions(sampleNodePos, regions, ax)

            # 3. connect the sampled node with the tree:
            bestCTR = np.inf
            best_parent = None

            if self._cut:
                T.cut(self.best_cost)
                
            cost_to_best_parent = np.inf
            found_potential_parent = False
            region_to_parent = None
            for shared_region in regions:
                for n in self.getNodesInRegion(shared_region):
                    found_potential_parent = True
                    cost_to_parent = shared_region.travelCost(
                        sampleNodePos, n.getData().p()
                    )
                    cost_to_root = n.getData().costToRoot() + cost_to_parent
                    if cost_to_root < min(bestCTR, np.inf*self.best_cost):
                        bestCTR = cost_to_root
                        cost_to_best_parent = cost_to_parent
                        best_parent = n
                        region_to_parent = shared_region

            # Sanity check that we sampled from a feasible region
            if best_parent is None:
                if not found_potential_parent:
                    T.getRoot().plotTree(ax)
                    self.visualizeActiveRegions(ax)
                    sampleNode.plot(ax,color='purple',marker='o',markersize=10)
                    raise ValueError("No parent found for the sampled node")
                else:
                    continue

            sampleTree = Tree(sampleNode)
            self.connect(
                sampleTree, 
                best_parent, 
                cost_to_best_parent, 
                region_to_parent
            )
            for r in regions:
                self._rttm[r].append(sampleTree)

            # Rewire initial node
            if sampleBordersInit:
                partition_sequence = []
                self.rewireInitNode()
                
            # activate newly entered regions (if any)            
            self.activateRegions(sampleNodePos)

            # Plotting
            if self.plotOptions().pae:
                self.plotOptions().allEdgeLines().add(
                    sampleTree.plotPathToParent(
                        ax, color = 'black', linewidth = 1, alpha = 0.2
                    )
                )
            if self.plotOptions().par:
                self.visualizeActiveRegions(ax)

        if self.best_path is None:
            T.getRoot().plotTree(ax)
            T.getRoot().getData().plot(ax,color='red',marker='o',markersize=10)
            n0 = Node(t0, initialRegions)
            n0.plot(ax, color='green', marker='o', markersize=10)
            raise ValueError("No path found")
               
        return self.best_path, self.best_cost

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

    def optimizeSwitchingPoints(self, T : Tree) -> None:
        if not self._optimize:
            return
        active = T
        init_waypoints : List[Node] = []
        while active is not None:
            init_waypoints.append(active.getData())
            active = active.getParent()

        wps, tcs = ConstantDCPRegion.planPath(init_waypoints)
        if wps is None:
            # failed to obtain a solution
            return
        
        pn = Node(T.getRoot().getData().p(), init_waypoints[-1].regions())
        parent = Tree(pn)
        for i in range(len(wps)-2, -1, -1):
            child = Tree(Node(wps[i], init_waypoints[i].regions()))
            self.connect(
                child, 
                parent, 
                tcs[i], 
                init_waypoints[i].activeRegionToParent()
            )
            parent = child

        if T.getData().costToRoot() > child.getData().costToRoot():
            print("Connecting optimized branch...")
            rootConnector = child.getRoot().getChildren()[0]
            self.connect(
                rootConnector, 
                T.getRoot(), 
                rootConnector.getData().costToParent(), 
                rootConnector.getData().activeRegionToParent()
            )
            self.best_path = child
            self.best_cost = child.getData().costToRoot()

            if self.plotOptions().pbp:
                self.plotBestPath()
        else:
            print("Optimized branch is not better...")
            print("Original cost {0}".format(T.getData().costToRoot()))
            print("Optimized cost {0}".format(child.getData().costToRoot()))
            # T.plotPathToRoot(color='red', annotate_cost=True)
            # child.plotPathToRoot(color='pink', annotate_cost=True)
            
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

    def initializePath(self, T : Tree, initialNode : Node) -> Tree:
        activeT = T.getRoot()
        activeRegions = activeT.getData().regions()
        initialRegions = initialNode.regions()
        while True:

            # if the initial region is entered then we connect and return
            irs = initialRegions.intersection(activeRegions)
            if len(irs) > 0:
                newNode = Node(initialNode.p(), initialRegions)
                newT = Tree(newNode)
                bc, br = self.bestTravelRegion(newNode, activeT.getData(), irs)
                self.connect(newT, activeT, bc, br)
                return newT

            # otherwise we project the current node to the boundary of the 
            # boundary facing the initial node
            for region in activeRegions:
                activeP = activeT.getData().p()
                proj = region.projectToBoundary(activeP, initialNode.p())
                if proj is not None:
                    break

            # append the node to the tree and activate it
            if proj is None:
                raise Exception("Could not project node to boundary")
            newRegions = self.world.getRegions(proj)
            newNode = Node(proj, newRegions)
            newT = Tree(newNode)                                           

            # determine best region to travel through
            intersectingRegions = newRegions.intersection(activeRegions)
            if len(intersectingRegions) == 0:
                raise ValueError("No intersecting regions found.")
            best_tcp, best_region = self.bestTravelRegion(
                newNode, activeT.getData(), intersectingRegions
            )

            if best_tcp >= np.inf:
                Warning("No feasible path continuation found.")
                return 

            self.connect(newT, activeT, best_tcp, best_region)
            activeT = newT
            activeRegions = newRegions
    
    def initializeRegionToTreeMapper(self, init : Tree) -> None:
        for r in self.regions:
            self._rttm[r] = []
        queue = [init]
        while queue:
            active = queue.pop(0)
            for r in active.getData().regions():
                self._rttm[r].append(active)
            queue.extend(active.getChildren())            

    def initializeActiveRegions(self, initTree : Tree) -> None:
        active = initTree
        while active is not None:
            for region in active.getData().regions():
                if region not in self.active_regions:   
                    self.active_regions.append(region)
            active = active.getParent()
        
        if self.plotOptions().par:
            self.visualizeActiveRegions()

    def clearEdgeLines(self) -> None:
        self.plotOptions().allEdgeLines().remove()

    def activateRegions(self, p : np.ndarray) -> None:
        for r in self.regions:
            if r in self.active_regions:
                continue
            if r.contains(p):
                self.active_regions.append(r)


    # plotters
    def plotPath(self, ax : plt.Axes = None) -> PlotObject:
        ax = getAxes(ax)
        return self.best_path.plotPathToRoot()
    
    def visualizeActiveRegions(self, ax : plt.Axes = None) -> None:
        ax = getAxes(ax)
        self.plotOptions().activeRegionObjects().remove()
        for r in self.active_regions:
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
        self._target_paths : Dict[Target, Dict[Target, Tree]]= {}
        self._plot_options = PlotOptions()

    # getters
    def tsp(self) -> TSP:
        return self._tsp
    
    def targetPaths(self) -> Dict[Target, Dict[Target, Tree]]:
        return self._target_paths

    # modifiers
    def planPath(
            self, 
            t0 : np.ndarray, 
            tf : np.ndarray, 
            max_iter = 500, 
            ax : plt.Axes = None
            ) -> Tuple[Set[Region], List[np.ndarray]]:
        ax = getAxes(ax)        
        initialRegions = self._world.getRegions(t0)
        targetRegions = self._world.getRegions(tf)

        # Switch to local planner if possible
        for i_reg in initialRegions:
            for t_reg in targetRegions:
                if i_reg == t_reg:
                    return i_reg, i_reg.planPath(t0, tf)
        
        # Otherwise use global planner
        rrt = RRT(self._world.regions())
        rrt._plot_options = self._plot_options
        rrt._max_iter = max_iter
        return rrt.planPath(t0, tf, ax)

    def removeUnreachableTargets(self) -> None:
        remove_targets = []
        reachable = {}
        
        for j in range(self._world.nTargets()):
            reachable[j] = False
        
        for i in range(self._world.nTargets()):
            for j in range(self._world.nTargets()):
                if i == j:
                    continue
                if self._tsp.targetDistances()[i,j] < np.inf:
                    reachable[j] = True
                    break

        for j in range(self._world.nTargets()):
            if not reachable[j]:
                remove_targets.append(j)

        self._tsp.removeTargets(remove_targets)

    def removeUnescapableTargets(self) -> None:
        remove_targets = []
        for i in range(self._world.nTargets()):
            i_escapable = False
            for j in range(self._world.nTargets()):
                if i == j:
                    continue
                if self._tsp.targetDistances()[i,j] < np.inf:
                    i_escapable = True
                    break
            if not i_escapable:
                remove_targets.append(i)
        self._tsp.removeTargets(remove_targets)

    def solveTSP(self) -> None:

        if self._tsp is None:
            self._tsp = TSP(self._world.targets())

        for i in range(self._world.nTargets()):
            target_i = self._world.targets()[i]
            self._target_paths[target_i] = {}
            for j in range(self._world.nTargets()):
                if i == j:
                    self._tsp.setTargetDistance(i,j,0)
                    continue
                target_j = self._world.targets()[j]
                plannedPath = self.planPath(target_i.p(), target_j.p())
                self._target_paths[target_i][target_j] = plannedPath[0] 
                self._tsp.setTargetDistance(i,j,plannedPath[1])
                print(f"Distance from {i} to {j} is {plannedPath[1]}")
        
        self._tsp.computeTSP()

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
