
# external imports
from __future__ import annotations
import pickle
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import numpy as np
import math

# internal imports
from .World import *
from .Agent import *


class Experiment:
    def __init__(self, name : str = "", domain : Domain = Domain()) -> None:
        self._vc = []                                                           # Voronoi centers
        self._voronoi = None                                                    # Voronoi object
        self._world : World = World()                                           # world object
        self._agent : Agent = None                                              # agent object
        self._domain = domain                                                   # domain object
        self._name = name                                                       # name of the experiment
        
    def randomRegion(self) -> Region:
        idx = np.random.randint(0, self._world.nRegions())
        return self._world.regions()[idx]

    def addRandomVoronoiPoints(self, M) -> None:
        self._vc = []
        self._M = M
        xvals = np.random.uniform(self._domain.xmin(), self._domain.xmax(), M)
        yvals = np.random.uniform(self._domain.ymin(), self._domain.ymax(), M)
        self._vc = np.array(list(zip(xvals, yvals)))

    def generatePartitioning(self) -> None:
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
        self._world.setRegions(regions)
    
    def assignRandomAgent(self) -> None:
        sensor = Sensor()
        for target in self._world.targets():
            if target.name == '3':
                sensor.setgetQualityFunction(target, SinusoidalgetQualityFunction(c1=np.random.uniform(3,20),c2=np.random.uniform(3,20)))
            else:
                sensor.setgetQualityFunction(target, GaussiangetQualityFunction())

            sensor.setNoiseMatrix(target, np.eye(1))
            sensor.setMeasurementMatrix(target, np.eye(1))
        self._agent = Agent(self._world, sensor=sensor)

    def addRandomTargets(self, n : int = None, fraction = 0.5) -> None:
        target_counter = 0
        if n is None:
            assert(fraction is not None)
            n = self._world.nRegions() * fraction
        n = math.floor(n)
        for region in self._world.regions():
            if target_counter >= n:
                break
           
            pos = region.p()
            distToBoundary = region.distToBoundary(pos)
            if distToBoundary < 0.005:
                continue
            phi0 = np.array([1.0])
            Q = np.array([0.8])
            A = np.array([0.0])
            target = Target(pos=pos, region=region, phi0=phi0, Q=Q, A=A)
            target.name = str(target_counter+1)
            self.addTarget(target)
            target_counter += 1

    def addTarget(self, target : Target) -> None:
        assert(isinstance(target, Target))
        self._world.addTarget(target)
        self._M += 1

    def voronoi(self) -> Voronoi:
        return self._voronoi

    def plotWorld(
            self, 
            with_sensor_quality=False, 
            add_target_labels=True, 
            fill_empty_regions=True
            ) -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots()
        ax.set_aspect('equal', 'box')
        fig.tight_layout()
        plt.axis('off')
        plt.ion()
        plt.show()
        fig.tight_layout()
    
        ax.set_xlim(self._domain.xmin()*1.01, self._domain.xmax()*1.01)
        ax.set_ylim(self._domain.ymin()*1.01, self._domain.ymax()*1.01)
        self._world.plotMissionSpace(
            ax=ax, 
            add_target_labels=add_target_labels, 
            fill_empty_regions=fill_empty_regions
            )

        if with_sensor_quality:
            self._agent.plotSensorQuality(ax=ax)

        return fig, ax

    def zoomToTargetRegion(self, ax : plt.Axes, name : str):
        target = self._world.getTarget(name)
        region = target.region()
        xrange = [np.inf, -np.inf]
        yrange = [np.inf, -np.inf]
        i = 0
        while i < 100:
            i += 1
            p = region.randomBoundaryPoint()
            xrange[0] = min(xrange[0], p[0])
            xrange[1] = max(xrange[1], p[0])
            yrange[0] = min(yrange[0], p[1])
            yrange[1] = max(yrange[1], p[1])

        ax.set_xlim(xrange[0] - 0.01, xrange[1] + 0.01)
        ax.set_ylim(yrange[0] - 0.01, yrange[1] + 0.01)

    def serialize(self, filename : str) -> None:
        plt.close()
        with open(filename, "wb") as f:
            pickle.dump(self, f)
    
    # static methods
    @staticmethod
    def deserialize(fileame : str):
        with open(fileame, "rb") as f:
            return pickle.load(f)
    
    @staticmethod
    def generate(n_sets=15, fraction=0.5, seed=784) -> Experiment:
        if seed is not None:
            np.random.seed(seed)
        ex = Experiment()
        ex.addRandomVoronoiPoints(n_sets)
        ex.generatePartitioning()
        ex.addRandomTargets(fraction=fraction)
        ex.assignRandomAgent()
        return ex
