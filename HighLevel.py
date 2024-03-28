
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Set
from World import *
from DataStructures import Tree, Node, PlotObject
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_simulated_annealing

class PlotOptions:
    def __init__(self):
        self._pbp : bool = True                        # plot best path
        self._pae : bool = True                        # plot all edge lines
        self._par : bool = True                        # plot active regions   
        self._psr : bool = False                       # plot search regions
        self._ael : PlotObject = PlotObject()   # all edge lines
        self._bpl : PlotObject = PlotObject()   # best path line
        self._arl : PlotObject = PlotObject()   # active region lines
        self._srl : PlotObject = PlotObject()   # search region lines

    def pbp(self) -> bool:
        return self._pbp
    
    def pae(self) -> bool:
        return self._pae
    
    def par(self) -> bool:
        return self._par
    
    def psr(self) -> bool:
        return self._psr
    
    def ael(self) -> PlotObject:
        return self._ael
    
    def bpl(self) -> PlotObject:
        return self._bpl
    
    def arl(self) -> PlotObject:
        return self._arl
    
    def srl(self) -> PlotObject:
        return self._srl

    def plot_any(self) -> bool:
        return self._pbp or self._pae or self._par or self._psr

    def toggle_all_plotting(self, bool) -> None:
        self._pbp = bool
        self._pae = bool
        self._par = bool
        self._psr = bool


class RRT:
    def __init__(self, regions : Set[Region], best_cost : float = np.inf) -> None:
        self.regions : Set[Region] = regions        # all regions
        self.best_path : Tree = None                # path
        self.best_cost : float = best_cost          # best cost
        self.active_regions = []                    # active regions
        self.world = World()                        # world
        self._target_distances : np.ndarray = None   # target distances
        
        self.world.SetRegions(self.regions)         # set regions

        # caching
        self._rttm : Dict[Region,Set[Tree]] = {}    # region to node mapper

        # convergence
        self._max_iter = 1000                       # maximum iterations

        # visualization
        self._plot_options = None                   # a plot options instance

        self._optimize = False
        self._rewire = False                # rewire
        self._cut = False                   # cut

    def planPath(self, t0, tf, ax : plt.Axes = None) -> Tuple[Set[Region], List[np.ndarray]]:
        
        targetRegions = self.world.GetRegions(tf)
        initialRegions = self.world.GetRegions(t0)
        targetNode = Node(tf, targetRegions)
        initialNode = Node(t0, initialRegions)

        if self.plot_options().plot_any():
            targetNode.Plot(ax, '', color='orange', marker='*', markersize=10, label='xf')
            initialNode.Plot(ax, '', color='green', marker='o', markersize=10, label='x0')
            
        # initialize tree (either with empty root or with an initial path)
        T = Tree(data = targetNode)
        initTree = self.InitializePath(T, initialNode)
        if initTree is not None:
            assert(isinstance(initTree, Tree))
            self.best_cost = initTree.getData().costToRoot()
            self.best_path = initTree

            if self.plot_options()._pbp:
                self.PlotBestPath(ax)

            self.OptimizeSwitchingPoints(initTree)

            T = initTree.getRoot()


        self.InitializeRegionToTreeMapper(T)
        self.InitializeActiveRegions(initTree)

        # while loop [without branch and cut]
        # 1. draw a random active region
        # 2. draw a random node of the active region:
        # 2.1 if region is the initial region, add the target position as a new node
        # 2.2 otherwise, draw a random node on the boundary of the current region
        # 3. connect the sampled node with the tree:
        # 3.1 for each node in the neighborhood (region) of the sampled node:
        # 3.1.1 compute the travel cost from root to sampled node through that parent
        # 3.2 connect sampled node with the parent node that minimizes the travel cost to root
        # 4. if sampled node is on boundary, activate the region(s) bordering the sampled node

        # Potential branch and cut extensions:
        # 3.1.1 if travel cost from the parent to root exceeds best cost, cut the node
        # 3.2.1.1 if the active region is now empty, deactivate the region 

        iter = 0
        while (iter < self._max_iter):
            iter += 1
            
            # 1. sample a random active region
            r = self.SampleActiveRegion()
                
            # 2. draw a random node of the active region:
            sampleNodePos = r.RandomBoundaryPoint()
            regions = self.world.GetRegions(sampleNodePos)
            sampleNode = Node(sampleNodePos, regions)
            sampleBordersInit = len(initialRegions.intersection(regions)) > 0

            if self.plot_options().psr():
                self.VisualizeSearchRegions(sampleNodePos, regions, ax)

            # 3. connect the sampled node with the tree:
            best_cost_to_root = np.inf
            best_parent = None

            if self._cut:
                T.cut(self.best_cost)
                
            cost_to_best_parent = np.inf
            found_potential_parent = False
            region_to_parent = None
            for shared_region in regions:
                for n in self.GetNodesInRegion(shared_region):
                    found_potential_parent = True
                    cost_to_parent = shared_region.TravelCost(sampleNodePos, n.getData().p())
                    cost_to_root = n.getData().costToRoot() + cost_to_parent
                    if cost_to_root < min(best_cost_to_root, np.inf*self.best_cost):
                        best_cost_to_root = cost_to_root
                        cost_to_best_parent = cost_to_parent
                        best_parent = n
                        region_to_parent = shared_region

            # Sanity check that we sampled from a feasible region
            if best_parent is None:
                if not found_potential_parent:
                    T.getRoot().PlotTree(ax)
                    self.VisualizeActiveRegions(ax)
                    sampleNode.Plot(ax, color='purple', marker='o', markersize=10)
                    raise ValueError("No parent found for the sampled node")
                else:
                    continue

            sampleTree = Tree(sampleNode)
            self.Connect(sampleTree, best_parent, cost_to_best_parent, region_to_parent)
            for r in regions:
                self._rttm[r].append(sampleTree)

            # Rewire initial node
            if sampleBordersInit:
                partition_sequence = []
                # self.Rewire(sampleTree)
                self.RewireInitNode()
                # self.RewireBestPath()

            # activate newly entered regions (if any)            
            self.ActivateRegions(sampleNodePos)

            # Plotting
            if self.plot_options().pae():
                self.plot_options().ael().add(sampleTree.plotPathToParent(ax, color = 'black', linewidth = 1, alpha = 0.2))
            if self.plot_options().par():
                self.VisualizeActiveRegions(ax)

        if self.best_path is None:
            T.getRoot().PlotTree(ax)
            T.getRoot().getData().Plot(ax, color='red', marker='o', markersize=10)
            n0 = Node(t0, initialRegions)
            n0.Plot(ax, color='green', marker='o', markersize=10)
            raise ValueError("No path found")
               
        return self.best_path, self.best_cost

    def Connect(self, child : Tree, parent : Tree, cost_to_parent : float, rtp : Region) -> None:
        child.setParent(parent, cost_to_parent)
        if rtp is not None:
            child.getData().activate_region_to_parent(rtp)

    def OptimizeSwitchingPoints(self, T : Tree) -> None:
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
        
        parent = Tree(Node(T.getRoot().getData().p(), init_waypoints[-1].regions()))
        for i in range(len(wps)-2, -1, -1):
            child = Tree(Node(wps[i], init_waypoints[i].regions()))
            self.Connect(child, parent, tcs[i], init_waypoints[i].active_region_to_parent())
            parent = child

        if T.getData().costToRoot() > child.getData().costToRoot():
            print("Connecting optimized branch...")
            rootConnector = child.getRoot().getChildren()[0]
            assert(isinstance(rootConnector, Tree))
            self.Connect(rootConnector, T.getRoot(), rootConnector.getData().costToParent(), rootConnector.getData().active_region_to_parent())
            self.best_path = child
            self.best_cost = child.getData().costToRoot()

            if self.plot_options().pbp():
                self.PlotBestPath()
        else:
            print("Optimized branch is not better...")
            print("Original cost {0}".format(T.getData().costToRoot()))
            print("Optimized cost {0}".format(child.getData().costToRoot()))
            # T.plotPathToRoot(color='red', annotate_cost=True)
            # child.plotPathToRoot(color='pink', annotate_cost=True)


    def RewireNode(self, node : Tree, region : Region) -> None:
        for sampleTree in self.GetNodesInRegion(region):
            rewire_cost = region.TravelCost(sampleTree.getData().p(), sampleTree.getParent().getData().p())
            if rewire_cost + sampleTree.getData().costToRoot() < self.best_cost:
                self.Connect(sampleTree, sampleTree.getParent(), rewire_cost)
                self.best_cost = sampleTree.getData().costToRoot()
                self.PlotBestPath()
                print("Rewired")

    def RewireBestPath(self):
        
        improved = False
        
        # queue up the best path
        activeTree = self.best_path
        queue : List[Tree] = []
        while activeTree is not None:
            queue.append(activeTree)
            activeTree = activeTree.getParent()

        # go down the tree and rewire according to best cost to go
        while queue:
            activeTree = queue.pop()
            r = queue.pop().getData().regions()[0]
            
            # rewire to best parent    
            for sampleTree in self.GetNodesInRegion(r):
                assert(isinstance(sampleTree, Tree))
                rewire_cost = r.TravelCost(activeTree.getData().p(), sampleTree.getData().p())
                if rewire_cost + sampleTree.getData().costToRoot() < activeTree.getData().costToRoot():
                    self.Connect(activeTree, sampleTree, rewire_cost)
                    improved = True
                    if self.plot_options().pbp():
                        self.PlotBestPath()

        if improved:
            self.best_cost = self.best_path.getData().costToRoot()
            
    def RewireInitNode(self) -> None:
        initRegions = self.best_path.getData().regions()
        improved = False
        for initRegion in initRegions:
            for parent in self.GetNodesInRegion(initRegion):
                rewire_cost = initRegion.TravelCost(self.best_path.getData().p(), parent.getData().p())
                if rewire_cost + parent.getData().costToRoot() < self.best_cost:
                    self.Connect(self.best_path, parent, rewire_cost, initRegion)
                    self.best_cost = self.best_path.getData().costToRoot()
                    improved = True
                    print("Rewired")
        if improved:
            self.OptimizeSwitchingPoints(self.best_path)
            if self.plot_options().pbp():
                self.PlotBestPath()

    def Rewire(self, sampleTree : Tree) -> None:
        initRegion = self.best_path.getData().active_region_to_parent()
        rewire_cost = initRegion.TravelCost(self.best_path.getData().p(), sampleTree.getData().p())
        if rewire_cost + sampleTree.getData().costToRoot() < self.best_cost:
            self.Connect(self.best_path, sampleTree, rewire_cost)
            self.PlotBestPath()
            print("Rewired")

    def InitializePath(self, T : Tree, initialNode : Node) -> Tree:
        
        activeT = T.getRoot()
        assert(len(activeT.getData().regions()) == 1)
        assert(len(initialNode.regions()) == 1)
        activeRegions = activeT.getData().regions()
        initialRegions = initialNode.regions()
        while True:

            # if the initial region is entered then we connect and return
            irs = initialRegions.intersection(activeRegions)
            if len(irs) > 0:
                newNode = Node(initialNode.p(), initialRegions)
                newT = Tree(newNode)
                best_tcp, best_region = self.BestTravelRegion(newNode, activeT.getData(), irs)
                self.Connect(newT, activeT, best_tcp, best_region)
                return newT

            # otherwise we project the current node to the boundary of the boundary facing the initial node
            for region in activeRegions:
                proj = region.ProjectToBoundary(activeT.getData().p(), initialNode.p())
                if proj is not None:
                    break

            # append the node to the tree and activate it
            assert(proj is not None)
            newRegions = self.world.GetRegions(proj)
            newNode = Node(proj, newRegions)
            newT = Tree(newNode)                                           

            # determine best region to travel through
            intersectingRegions = newRegions.intersection(activeRegions)
            assert(len(intersectingRegions) > 0)
            best_tcp, best_region = self.BestTravelRegion(newNode, activeT.getData(), intersectingRegions)
            self.Connect(newT, activeT, best_tcp, best_region)
            activeT = newT
            activeRegions = newRegions
    
    def InitializeRegionToTreeMapper(self, init : Tree) -> None:
        for r in self.regions:
            self._rttm[r] = []
        queue = [init]
        while queue:
            active = queue.pop(0)
            for r in active.getData().regions():
                self._rttm[r].append(active)
            queue.extend(active.getChildren())            

    def InitializeActiveRegions(self, initTree : Tree) -> None:
        active = initTree
        while active is not None:
            for region in active.getData().regions():
                if region not in self.active_regions:   
                    self.active_regions.append(region)
            active = active.getParent()
        
        if self.plot_options().par():
            self.VisualizeActiveRegions()

    def ClearEdgeLines(self) -> None:
        self.plot_options().ael().remove()

    def ActivateRegions(self, p : np.ndarray) -> None:
        for r in self.regions:
            if r in self.active_regions:
                continue
            if r.Contains(p):
                self.active_regions.append(r)

    # getters
    def GetNodesInRegion(self, r : Region) -> List[Tree]:
        return self._rttm[r]

    def BestTravelRegion(self, n0 : Node, nf : Node, regions : Set[Region]) -> Tuple[float, Region]:
        best_tcp = np.inf
        best_region = None
        for region in regions:
            tcp = region.TravelCost(n0.p(), nf.p())
            if tcp < best_tcp:
                best_tcp = tcp
                best_region = region
        return best_tcp, best_region
    
    def SampleActiveRegion(self) -> Region:
        return self.active_regions[np.random.randint(0, len(self.active_regions))]

    def plot_options(self) -> PlotOptions:
        return self._plot_options

    # plotters
    def PlotPath(self, ax : plt.Axes = plt) -> PlotObject:
        return self.best_path.plotPathToRoot()
    
    def VisualizeActiveRegions(self, ax : plt.Axes = plt) -> None:
        self.plot_options().arl().remove()
        for r in self.active_regions:
            assert(isinstance(r, Region))
            self.plot_options().arl().add(r.Fill(ax, color = 'green', alpha = 0.2))

    def VisualizeSearchRegions(self, p : np.ndarray, regions : Set[Region], ax : plt.Axes = plt) -> None:
        self.plot_options().srl().remove()
        self.plot_options().srl().add(plt.plot(p[0], p[1], 'gd'))
        for r in regions:   
            self.plot_options().srl().add(r.Fill(ax, color = 'blue', alpha = 0.2))

    def PlotBestPath(self, ax : plt.Axes = plt) -> None:
        self.plot_options().bpl().remove()
        self.plot_options().bpl().add(self.best_path.plotPathToRoot(None, ax, color = 'red', linewidth=2))

    def PlotAllEdgeLines(self, T : Tree, ax : plt.Axes = plt) -> None:
        self.plot_options().ael().remove()
        queue = [T.getRoot()]
        while len(queue) > 0:
            n = queue.pop(0)
            for c in n.getChildren():
                queue.append(c)
            
            if n.getParent() is not None:
                self.plot_options().ael().add(n.plotPathToParent(ax, color = 'black', linewidth = 1, alpha = 0.5))


class TSP:
    def __init__(self, targets : List[Target]) -> None:
        self._targets = targets
        self._target_distances = np.zeros((len(targets), len(targets)))
        self._best_permutation = None
        self._best_distance = np.inf

    # getters       
    def getTargetVisitingSequence(self) -> List[Target]:
        tvs = []
        for p in self._best_permutation:
            tvs.append(self._targets[p])
        return tvs
    
    def targetDistances(self) -> np.ndarray:
        return self._target_distances

    # setters
    def setTargetDistance(self, i : int, j : int, d : float) -> None:
        self._target_distances[i,j] = float(d)

    # modifiers
    def removeTargets(self, indices : List[int]) -> None:
        print("Removing {0} invalid targets".format(len(indices)))
        self._targets = []
        for i in range(len(self._targets)):
            if i not in indices:
                self._targets.append(self._targets[i])
        self._target_distances = np.delete(self._target_distances, indices, axis=0)
        self._target_distances = np.delete(self._target_distances, indices, axis=1)
                
    def computeTSP(self, exact = True):        
        if exact:
            permutation, distance = solve_tsp_dynamic_programming(self._target_distances)
        else:
            permutation, distance = solve_tsp_simulated_annealing(self._target_distances)
        
        if distance < self._best_distance:
            self._best_distance = distance
            self._best_permutation = permutation
        
        return permutation, distance

    # plotters    
    def plotTargetDistances(self, ax : plt.Axes = plt, style='', **kwargs) -> PlotObject:
        po = PlotObject()
        for i in range(self._target_distances.shape[0]):
            for j in range(self._target_distances.shape[1]):
                if i == j:
                    continue
                p = self.targets[i].p()
                q = self.targets[j].p()
                delta = -(q - p)/np.linalg.norm(q-p)
                po.add(ax.plot([p[0], q[0]], [p[1],  q[1]], **kwargs))
                po.add(ax.quiver(p[0], p[1], delta[0], delta[1], pivot='tip', angles='xy', **kwargs))
                po.add(ax.annotate(f"{self._target_distances[i,j]:.2f}", ((p[0]+q[0])/2, (p[1]+q[1])/2), fontsize=12, color='black'))
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
            ax : plt.Axes = plt
        ) -> Tuple[Set[Region], List[np.ndarray]]:
        
        initialRegions = self._world.GetRegions(t0)
        targetRegions = self._world.GetRegions(tf)

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
        
        for j in range(self._world.NT()):
            reachable[j] = False
        
        for i in range(self._world.NT()):
            for j in range(self._world.NT()):
                if i == j:
                    continue
                if self._tsp.targetDistances()[i,j] < np.inf:
                    reachable[j] = True
                    break

        for j in range(self._world.NT()):
            if not reachable[j]:
                remove_targets.append(j)

        self._tsp.removeTargets(remove_targets)

    def removeUnescapableTargets(self) -> None:
        remove_targets = []
        for i in range(self._world.NT()):
            i_escapable = False
            for j in range(self._world.NT()):
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

        for i in range(self._world.NT()):
            target_i = self._world.targets()[i]
            self.target_paths[target_i] = {}
            for j in range(self._world.NT()):
                if i == j:
                    self._tsp.setTargetDistance(i,j,0)
                    continue
                target_j = self._world.targets()[j]
                plannedPath = self.planPath(target_i.p(), target_j.p())
                self.target_paths[target_i][target_j] = plannedPath[0] 
                self._tsp.setTargetDistance(i,j,plannedPath[1])
                print(f"Distance from {i} to {j} is {plannedPath[1]}")
        
        self._tsp.computeTSP()

    # plotters    
    def plotTSPSolution(
            self, 
            ax : plt.Axes = plt, 
            annotate=False, 
            **kwargs
        ) -> PlotObject:
        if self.tsp._best_permutation is None:
            print("No TSP solution exists. Please run 'solveTSP' first.")
            return None
        
        if self.tsp._best_permutation is None:
            return
        po = PlotObject()
        args = kwargs.copy()
        for i in range(0,len(self.tsp._best_permutation)):
            currTarget = self._world.targets()[self.tsp._best_permutation[i-1]]
            nextTarget = self._world.targets()[self.tsp._best_permutation[i]]
            currPath = self.target_paths[currTarget][nextTarget]
            assert(isinstance(currPath, Tree))
            po.add(currPath.plotPathToRoot(ax=ax, plot_direction=True, **args))
            currPath = currPath.getParent()
            
            if annotate:
                po.add(ax.annotate(f"{i}", (currPath.getData().p()[0], currPath.getData().p()[1]), fontsize=12, color='black'))

        return po
