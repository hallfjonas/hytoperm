
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
        self._world : World = World()                                           # world object
        self._agents : List[Agent] = []                                         # agent object
        self._homogeneous_agents = False                                        # all agents have the same sensor model
        self._domain = domain                                                   # domain object
        self._name = name                                                       # name of the experiment

    # getters
    def world(self) -> World:
        return self._world

    def agent(self, idx : int = 0) -> Agent:
        if idx >= len(self._agents) or idx < 0:
            raise IndexError("Agent index out of bounds.")
        return self._agents[idx]
        
    def agents(self) -> List[Agent]:
        return self._agents

    def randomRegion(self) -> Region:
        idx = np.random.randint(0, self._world.nRegions())
        return self._world.regions()[idx]

    def nAgents(self) -> int:
        return len(self._agents)
    
    def nTargets(self) -> int:
        return self._world.nTargets()

    # modifiers
    def generatePartitioning(self, **kwargs) -> None:
        pass
    
    def addAgent(
            self, 
            gpp : GlobalPathPlanner = None,
            sensor : Sensor = None,
            name : str = ""
            ) -> None:
        agent = Agent(self._world, sensor=sensor, gpp=gpp, name=name)
        self._agents.append(agent)

    def addAgentHomogeneousSensor(
                self, 
                gpp : GlobalPathPlanner = None, 
                name : str = ""
                ) -> None:
        """
        Build a homogeneous sensor and add an agent with this sensor.
        """
        sensor = HomogeneousSensor()
        sensor.setTargetQualityFunction(GaussianQualityFunction())
        sensor.setNoiseMatrix(np.eye(1))
        sensor.setMeasurementMatrix(np.eye(1))
        self.addAgent(gpp=gpp, sensor=sensor, name=name)

    def addAgentHeterogeneousSensor(
                self, 
                gpp : GlobalPathPlanner = None, 
                name : str = ""
                ) -> None:
        """
        Build a sample heterogeneous sensor and add an agent with this sensor.
        """
        sensor = HeterogeneousSensor()
        for target in self._world.targets():
            if target.name == '3':
                sensor.setTargetQualityFunction(
                    SinusoidalQualityFunction(
                        c1=np.random.uniform(3,20),
                        c2=np.random.uniform(3,20)
                        ),
                    target=target 
                    )
            else:
                sensor.setTargetQualityFunction(
                    GaussianQualityFunction(),
                    target=target 
                    )

            sensor.setNoiseMatrix(np.eye(1), target=target)
            sensor.setMeasurementMatrix(np.eye(1), target=target)
        self.addAgent(gpp=gpp, sensor=sensor, name=name)

    def addTarget(self, target : Target) -> None:
        if not isinstance(target, Target):
            raise ValueError("Argument must be of type Target.")
        self._world.addTarget(target)

    # plotters
    def plotWorld(
            self, 
            with_sensor_quality=False, 
            add_target_labels=True, 
            fill_empty_regions=True,
            plot_partition=True,
            plot_targets=True,
            plot_vector_field=True,
            plot_domain=False,
            ax=None
            ) -> Tuple[plt.Figure, plt.Axes]:
        ax = getAxes(ax)        
        ax.set_aspect('equal', 'box')
        ax.axis('off')
        ax.set_xlim(self._domain.xmin()*1.01, self._domain.xmax()*1.01)
        ax.set_ylim(self._domain.ymin()*1.01, self._domain.ymax()*1.01)

        if with_sensor_quality and len(self._agents) > 0:
            if len(self._agents) == 1 or self._homogeneous_agents:
                self.agent(0).plotSensorQuality(ax=ax)
            else:
                warnings.warn("Adding the sensor quality tot he world plot is only supported for a single agent.")

        self._world.plotMissionSpace(
            ax=ax, 
            add_target_labels=add_target_labels, 
            fill_empty_regions=fill_empty_regions,
            plot_partition=plot_partition,
            plot_targets=plot_targets,
            plot_vector_field=plot_vector_field,
            plot_domain=plot_domain
            )

        return plt.gcf(), ax

    def zoomToTargetRegion(self, ax : plt.Axes, name : str):
        target = self._world.getTargetByName(name)
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
        if not str.endswith(filename, ".pkl") and not str.endswith(filename, ".pickle"):
            raise ValueError("File must be have a pickle file extension (.pkl or .pickle).")
        plt.close()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(self, f)
    
    # static methods
    @staticmethod
    def deserialize(filename : str):
        if not str.endswith(filename, ".pkl") and not str.endswith(filename, ".pickle"):
            raise ValueError("File must be have a pickle file extension (.pkl or .pickle).")
        if not os.path.exists(filename):
            return None
        with open(filename, "rb") as f:
            return pickle.load(f)
    
    @staticmethod
    def generate(
            seed=None, 
            domain=Domain(),
            spherical=False,
            **kwargs
            ) -> Experiment:
        '''
        generate: Generate a random experiment.
        
        Parameters:
        seed: int (default = None)
            Seed for the random number generator.
        domain: Domain (default = Domain())
            Specify a domain to bound the mission space.
        **kwargs: dict
            Special keyword arguments for the specific experiment type.
        '''
        raise NotImplementedError("You tried to generate an abstract experiment. Please call the generate method of a specific experiment type.")

    def addAgents(
            self, 
            n_agents: int,
            gpp: GlobalPathPlanner = None,
            sensor: Sensor = None
            ) -> None:
        
        for i in range(n_agents):
            ex.addRandomAgent(gpp=gpp, sensor=sensor, name=str(i))
            if homogeneous_agents:
                sensor = ex.agent().sensor()

    def getFraction(self, **kwargs):
        return kwargs.get('fraction', 0.33)

    def getNTargets(self, **kwargs):
        if 'n_targets' in kwargs:
            return kwargs.get('n_targets')
        
        fraction = self.getFraction(**kwargs)
        if 'n_sets' in kwargs:
            return math.floor(kwargs.get('n_sets') * fraction)
        
        raise ValueError("Number of targets must be specified. Do this either directly by passing argument 'n_targets', or indirectly from the equation n_targets = fraction * n_sets.")

    def getNSets(self, **kwargs):
        if 'n_sets' in kwargs:
            return kwargs.get('n_sets')
        
        fraction = self.getFraction(**kwargs)
        if 'n_targets' in kwargs:
            return math.ceil(kwargs.get('n_targets') / fraction)
        
        raise ValueError("Number of sets must be specified. Do this either directly by passing argument 'n_sets', or indirectly from the equation n_targets = fraction * n_sets.")

class VoronoiExperiment(Experiment):
    def __init__(self, name : str = "", domain : Domain = Domain()) -> None:
        self._vc = []                                                           # Voronoi centers
        self._voronoi = None                                                    # Voronoi object
        super().__init__(name=name, domain=domain)

    def generate(
            seed=None, 
            domain=Domain(),
            **kwargs
            ) -> VoronoiExperiment:
        '''
        generate: Generate a random Voronoi-based experiment.
        
        Special keyword arguments:
        n_targets: int
            Number of targets. If not specified, it is computed via the equation 
            n_targets = fraction * n_sets.
        n_sets: int
            Number of sets for the partition. If not specified, it is computed 
            via the equation n_targets = fraction * n_sets.
        fraction: float (default = 0.33)
            Fraction of regions that will contain targets.
        min_dist: float (default = 0.1)
            Minimum distance between Voronoi points.
        n_obstacles: int
            Number of regions that will be obstacles.
        '''
        if seed is not None:
            np.random.seed(seed)
        
        print(f"Generating voronoi experiment with seed = {seed}.")

        ex = VoronoiExperiment(domain=domain)
        ex.addRandomVoronoiPoints(
            ex.getNSets(**kwargs), 
            min_dist=kwargs.get('min_dist', 0.1)
        )
        ex.generatePartitioning(**kwargs)
        ex.addRandomTargets(n=ex.getNTargets(**kwargs))
        gpp = RRBTGlobalPlanner(ex.world())
        ex.addAgentHomogeneousSensor(gpp=gpp)            
        return ex

    def voronoi(self) -> Voronoi:
        return self._voronoi
    
    def addRandomVoronoiPoints(self, M : int, min_dist=0.0) -> None:
        self._vc = []
        counter = 0
        
        if (M < 0):
            raise ValueError("Number of Voronoi points must be a nonnegative number.")
        
        while len(self._vc) < M:

            # prevent infinite loop
            counter += 1
            if counter > 10000 * M:
                raise RuntimeError(
                    "Could not generate enough Voronoi points. " + 
                    "Try decreasing the minimum distance between points."
                )

            # sample new point
            x = np.random.uniform(self._domain.xmin(), self._domain.xmax())
            y = np.random.uniform(self._domain.ymin(), self._domain.ymax())
            p = np.array([x, y])
            
            # always add first point
            if len(self._vc) == 0:
                self._vc.append(p)
                continue
            
            # check if point is too close to existing points
            dist = np.linalg.norm(self._vc - p, axis=1)
            if np.min(dist) < min_dist:
                continue

            self._vc.append(p)

        self._vc = np.array(self._vc)

    def generatePartitioning(self, **kwargs) -> None:
        """
        Generate a partitioning of the domain using the Voronoi centers.

        Special keyword arguments:
        n_obstacles: int (default = 0)
            Number of regions that will be obstacles.
        """
        n_obstacles = kwargs.get('n_obstacles', 0)        
        if self._vc.shape[0] > 1:
            self._voronoi = Voronoi(self._vc)

        regions = []
        for i in range(len(self._vc)):
            g = {}
            b = {}
            for j in range(len(self._vc)):
                if i == j:
                    continue

                a = self._vc[j] - self._vc[i]
                a = a / np.linalg.norm(a)
                g[j] = a 
                b[j] = a @ (self._vc[i] + self._vc[j]) / 2
            
            if i < len(self._vc) - n_obstacles:
                dyn = ConstantDynamics(2,0,0,np.random.uniform(-0.5,0.5,2))
                regions.append(ConstantDCPRegion(
                    g,
                    b,
                    self._vc[i], 
                    domain=self._domain, 
                    dynamics=dyn)
                    )      
            else:
                regions.append(ObstacleCPRegion(
                    g,
                    b,
                    self._vc[i], 
                    domain=self._domain)
                    )
        self._world.setRegions(regions)
    
    def addRandomTargets(
            self, 
            n : int = None, 
            fraction : float = 0.5,
            min_dist_to_boundary : float = 0.005
            ) -> None:
        target_counter = 0
        if fraction < 0 or float(fraction) > 1:
            raise ValueError("Fraction must be in [0,1].")
        if n is None:
            if fraction is None:
                raise ValueError("Either n or fraction must be specified.")
            n = self._world.nRegions() * fraction
        n = math.floor(n)

        if n > self._world.nRegions() - self._world.nObstacles():
            raise ValueError("Number of targets exceeds number of regions.")

        for region in self._world.regions():
            if target_counter >= n:
                break
           
            if region.isObstacle():
                continue

            cntr = 0
            while True:
                pos = region.randomPoint()
                if region.distToBoundary(pos) > min_dist_to_boundary:
                    break
                cntr += 1
                if cntr > 1000:
                    raise Exception("Could not add target. Try decreasing minimum distance to boundary.")
            phi0 = np.array([1.0])
            Q = np.array([0.8])
            A = np.array([0.001])
            target = Target(pos=pos, region=region, phi0=phi0, Q=Q, A=A)
            target.name = str(target_counter+1)
            self.addTarget(target)
            target_counter += 1

        if target_counter < n:
            raise Exception("Could not add all targets.")

class SphericalExperiment(Experiment):
    def __init__(self, name : str = "", domain : Domain = Domain()) -> None:
        self._spheres: List[SphericalRegion] = []
        super().__init__(name=name, domain=domain)

    def generate(
            seed=None, 
            domain=Domain(),
            **kwargs
            ) -> Experiment:
        '''
        generate: Generate a random experiment with spherical regions.
        
        Special keyword arguments:
        n_targets: int
            Number of target locations.
        radius: float (default = None)
            Maximum radius of the targets. If None, the radius is randomized.
        min_dist: float
            Minimum distance between target regions.
        '''
        if seed is not None:
            np.random.seed(seed)

        print(f"Generating spherical experiment with seed = {seed}.")

        ex = SphericalExperiment(domain=domain)
        n_targets = ex.getNTargets(**kwargs)
        radius = kwargs.get('radius', None)
        min_dist = kwargs.get('min_dist', 0.1)
        ex.addRandomSpheres(
            n_targets,
            min_dist=min_dist, 
            max_radius=radius if radius is not None else np.inf,
            min_radius=radius if radius is not None else min_dist
        )
        ex.generatePartitioning(**kwargs)
        ex.addCenteredTargets()
        gpp = NormBasedGlobalPlanner(ex.world())
        ex.addAgentHomogeneousSensor(gpp=gpp)
        return ex

    def addRandomSpheres(self, M: int, min_radius=0.0, max_radius=np.inf, min_dist=0.0) -> None:
        if M is None or M < 0:
            raise ValueError("Number of target locations must be a nonnegative number.")
        
        dx = self._domain.xmax() - self._domain.xmin()
        dy = self._domain.ymax() - self._domain.ymin()

        targets: List[Target] = []
        k = 0
        while len(targets) < M:
            
            k += 1
            if k > 10000 * M:
                raise RuntimeError("Could not generate enough target locations. Try decreasing the minimum or maximum radius.")

            x = np.random.uniform(self._domain.xmin(), self._domain.xmax())
            y = np.random.uniform(self._domain.ymin(), self._domain.ymax())
            
            r_m = min(max_radius, x - self._domain.xmin(), self._domain.xmax() - x, y - self._domain.ymin(), self._domain.ymax() - y)
            
            if r_m < min_radius:
                continue
            
            rad = np.random.uniform(min_radius, r_m)
            
            r = SphericalRegion(np.array([x, y]), rad)
            intersects = False
            for target in targets:
                if r.intersects(target.region(), tol=min_dist):
                    intersects = True
                    break
            if intersects:
                continue
            targets.append(r)
        self._spheres = targets

    def generatePartitioning(self, **kwargs) -> None:
        self._world.setRegions(self._spheres)

    def addRandomTargets(self) -> None:
        for region in self._world.regions():
            pos = region.randomPoint()
            phi0 = np.array([1.0])
            Q = np.array([0.8])
            A = np.array([0.001])
            target = Target(pos=pos, region=region, phi0=phi0, Q=Q, A=A)
            target.name = str(len(self._world.targets())+1)
            self.addTarget(target)

    def addCenteredTargets(self) -> None:
        for region in self._world.regions():
            pos = region.p()
            phi0 = np.array([1.0])
            Q = np.array([0.8])
            A = np.array([0.001])
            target = Target(pos=pos, region=region, phi0=phi0, Q=Q, A=A)
            target.name = str(len(self._world.targets())+1)
            self.addTarget(target)
