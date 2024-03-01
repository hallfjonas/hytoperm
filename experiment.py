
from World import *
import matplotlib.pyplot as plt
from matplotlib import figure
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np

class Experiment:

    def __init__(self, domain : Domain = Domain()) -> None:
        self._vc = []                   # Voronoi centers
        self._voronoi = None            # Voronoi object
        self._world : World = World()   # world object
        self._domain = domain           # domain object
        
    def AddRandomVoronoiPoints(self, M) -> None:
        self._vc = []
        self._M = M
        for i in range(M):
            self._vc.append(np.array((np.random.uniform(self._domain.xmin(), self._domain.xmax()), np.random.uniform(self._domain.ymin(), self._domain.ymax()))))

    def GeneratePartitioning(self) -> None:
        self._voronoi = Voronoi(self._vc)
        regions = []
        for i in range(self._M):
            g = {}
            b = {}
            for j in range(self._M):
                if i == j:
                    continue

                a = self._vc[j] - self._vc[i]
                a = a / np.linalg.norm(a)
                g[j] = a 
                b[j] = a @ (self._vc[i] + self._vc[j]) / 2
            dyn = ConstantDynamics(2,0,0,np.random.uniform(-0.5,0.5,2))
            regions.append(ConstantDCPRegion(g,b,self._vc[i], domain=self._domain, dynamics=dyn))      
        self._world.SetRegions(regions)
    
    def AddRandomAgent(self) -> None:
        sensor = Sensor()
        for target in self._world.targets():
            ranint = np.random.randint(0, 6)
            if ranint == 0:
                sensor.setSensingQualityFunction(target, ConstantSensingQualityFunction())
            elif ranint <= 1:
                sensor.setSensingQualityFunction(target, SinusoidalSensingQualityFunction(c1=np.random.uniform(3,20),c2=np.random.uniform(3,20)))
            else:
                sensor.setSensingQualityFunction(target, GaussianSensingQualityFunction())
        self._world.AddAgent(Agent(np.array([0.5,0.5]), sensor=sensor))

    def AddRandomTargets(self, fraction=0.5) -> None:
        for region in self._world.regions():
            assert(isinstance(region, CPRegion))
            if np.random.uniform(0, 1) < fraction:
                pos = region.p()
                distToBoundary = region.DistToBoundary(pos)
                target = Target(pos, region, np.array([1.0]))
                self.AddTarget(target)

    def AddTarget(self, target : Target) -> None:
        assert(isinstance(target, Target))
        self._world.AddTarget(target)
        self._M += 1

    def voronoi(self) -> Voronoi:
        return self._voronoi

    def PlotWorld(self, ax : plt.Axes = None) -> PlotObject:
        if ax is None:
            ax = plt.gca()
        ax.set_xlim(self._domain.xmin()*1.1, self._domain.xmax()*1.1)
        ax.set_ylim(self._domain.ymin()*1.1, self._domain.ymax()*1.1)
        return self._world.PlotMissionSpace(ax)

    