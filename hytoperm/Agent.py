      
# external imports
import warnings

# internal imports
from .World import *
from .Dynamics import *
from .GlobalPlanning import *
from .Optimization import *
from .Sensor import *
from .PlotAttributes import PlotAttributes
from .Decomposition import *
_plotAttr = PlotAttributes()

'''
Agent: the main agent class
'''
class Agent:
    def __init__(
            self, 
            world : World, 
            sensor : Sensor,
            gpp : GlobalPathPlanner = None,
            name : str = ""
            ) -> None:
        self.name = name
        self._world : World = world                                             # world instance    
        self._sensor : Sensor = sensor                                          # utilized sensor        
        self._gpp : GlobalPathPlanner = gpp                                     # global path planner
        self._tvs : List[Target] = []                                           # target visiting sequence
        self._K: int = None                                                     # length of the visiting sequence
        self.decomposition : Decomposition = None                               # the trajectory decomposition    
        
        # optimization statistics
        self._kkt_residuals : Dict[int, float] = {}                             # map a target visit index to a KKT residual
        self._iteration_stats: IterationStats = IterationStats()                # iteration statistics        
        self._kkt_violations : List[np.ndarray] = []                            # KKT residuals (per steady state cycle)
           
    def setTargetVisitingSequence(self, tvs : List[Target]) -> None:
        self._tvs = tvs
        self._K = len(self._tvs)
        if isinstance(self.decomposition, Decomposition):
            self.decomposition.setTargetVisitingSequence(tvs)

    def computeVisitingSequence(self) -> None:
        self.gpp().solveTSP()
        tvs = self.gpp().tsp().getTargetVisitingSequence()
        self.setTargetVisitingSequence(tvs)
    
    def simulateCycle(self) -> None:
        self.decomposition.initialize()        
        self.decomposition._cycle.simulate()
        
    def optimizeCycle(self, op: OptimizationParameters) -> None:
        self.initializeDecomposition()
        it = 0
        self._iteration_stats = IterationStats()
        while True:
            self.decomposition.updateParameters(
                op,
                self._iteration_stats
            )
            self.printIteration(self._iteration_stats)

            if it > op.optimization_iters:
                print("Maximum number of iterations reached...")
                break

            it += 1
    
    def initializeDecomposition(self, **kwargs) -> None:
        """
        Initialize the decomposition.

        Args:
            **kwargs: keyword arguments for the decomposition.
        """
        self.decomposition = Decomposition(
            world = self.world(),
            sensor = self.sensor(),
            gpp = self.gpp(),
            tvs = self._tvs,
            **kwargs
        )
        self.decomposition.initialize()

    # Getters
    def gpp(self) -> GlobalPathPlanner:
        return self._gpp   

    def world(self) -> World:
        return self._world

    def sensor(self) -> Sensor:
        return self._sensor
    
    # Plotters
    def plotMSE(
            self, 
            ax : plt.Axes = None, 
            add_labels = False, 
            **kwargs
            ) -> PlotObject:
        ax = getAxes(ax)
        return self.decomposition._cycle.plotMSE(add_labels=add_labels, ax=ax, **kwargs)
    
    def plotControls(
            self, 
            ax : plt.Axes = None, 
            add_monitoring_labels = False, 
            **kwargs
            ) -> PlotObject:
        ax = getAxes(ax)
        return self.decomposition._cycle.plotControls(
            ax, 
            add_monitoring_labels=add_monitoring_labels, 
            **kwargs
            )

    def plotCycle(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        self.decomposition._cycle.plot(ax, **kwargs)
       
        for ms in self.decomposition._cycle._monitoringSegments:
            p = ms.pTrajectory.x[:,-2]
            q = ms.pTrajectory.x[:,-1]
            po.add(self.plotArrow(p, q, ax, **kwargs))

        return po

    def plotArrow(self, p : np.array, q : np.array, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject(ax.arrow(
            p[0], 
            p[1], 
            q[0] - p[0], 
            q[1] - p[1], 
            width=0, 
            head_width = 0.02, 
            head_length=0.033, 
            overhang=0.3, 
            length_includes_head=True,**kwargs
            ))
        return po
        
    def plotGlobalCosts(
            self, 
            ax : plt.Axes = None, 
            **kwargs
            ) -> PlotObject:
        ax = getAxes(ax)
        return PlotObject(ax.plot(self._iteration_stats.global_costs, **kwargs))
    
    def plotGlobalGradientNorms(
            self, 
            ax : plt.Axes = None, 
            **kwargs
            ) -> PlotObject:
        ax = getAxes(ax)
        return PlotObject(ax.plot(self._kkt_violations, **kwargs))
        
    def plotGlobalGradients(
            self, 
            ax : plt.Axes = None, 
            **kwargs
            ) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        Nk = len(self._iteration_stats.global_gradients)
        tc = _plotAttr.target_colors
        for i in range(self._K):
            eka = extendKeywordArgs(
                {'color': tc[-int(self._tvs[i].name)+1]}, 
                **kwargs
            )
            dJ_di = [self._iteration_stats.global_gradients[k][i] for k in range(Nk)]
            po.add(ax.plot(dJ_di, **eka))
        return po
    
    def plotTauVals(
            self, 
            ax : plt.Axes = None, 
            add_lower_bounds = True, 
            **kwargs
            ) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        tv = np.array(self._iteration_stats.tau_values)
        for i in range(tv.shape[1]):
            eka = extendKeywordArgs(
                {'color': _plotAttr.target_colors[-int(self._tvs[i].name)+1]}, 
                **kwargs
                )
            po.add(ax.plot(tv[:,i], **eka))

            if add_lower_bounds:
                eka = extendKeywordArgs(
                    {'alpha' : 0.75, 'linestyle' : '--'}, 
                    **eka
                    )
        return po

    def plotKKTViolations(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        NK = len(self._kkt_violations)
        tc = _plotAttr.target_colors
        for i in range(self._K):
            eka = extendKeywordArgs(
                {'color': tc[-int(self._tvs[i].name)+1]}, 
                **kwargs
            )
            po.add(PlotObject(
                ax.plot([self._kkt_violations[k][i] for k in range(NK)], 
                        **eka)))
        return po

    def plotAlphas(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        return PlotObject(ax.plot(self._iteration_stats.alphas, **kwargs))

    def plotSensorQuality(
            self, 
            grid_size = 0.005, 
            ax : plt.Axes = None, 
            **kwargs
            ) -> PlotObject:
        ax = getAxes(ax)
        X, Y, Z = self._world.getMeshgrid(dx=grid_size, dy=grid_size)
        sensor = self.sensor()
        for i in range(X.shape[0]):
            for j in range(Y.shape[1]):
                p = np.array((X[i,j], Y[i,j]))
                sensor.setPosition(p)
                for target in self._world.targets():
                    region = target.region()
                    if region.contains(p):
                        Z[i,j] = sensor.getSensingQuality(target=target)
        sqAttr = _plotAttr.sensor_quality.getAttributes()
        eka = extendKeywordArgs(sqAttr, **kwargs)
        cf = ax.contourf(X, Y, Z, **eka)
        # plt.colorbar(res)
        return PlotObject(cf)
    
    def addSteadyStateLines(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        cumsum = np.cumsum(self._iteration_stats.steady_state_iterations)
        po = PlotObject()
        for i in range(len(cumsum)):
            po.add(ax.axvline(cumsum[i], **kwargs))
        return po

    # printers
    def printHeader(self) -> None:
        print("----|-----------|-----------|-----------|--------|--------")
        print(" it | avrg cost | grad. nrm | step size | it std | is std ")
        print("----|-----------|-----------|-----------|--------|--------")
              
    def printIteration(self, stats: IterationStats) -> None:
        if stats.iterate % 10 == 0:
            self.printHeader()
        
        if len(stats.global_costs) == 0:
            return

        print("{:3d} | {:9.2e} | {:9.2e} | {:9.2e} | {:6d} | {:>6s}".format(
            stats.iterate, 
            stats.global_costs[-1], 
            stats.global_gradient_norms[-1],
            stats.alphas[-1], 
            stats.steady_state_iterations[-1],
            'T' if stats.is_steady_state[-1] else 'F'
        ))
