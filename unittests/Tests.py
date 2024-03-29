
# external imports
import matplotlib.pyplot as plt
import numpy as np  
from scipy.spatial import Voronoi, voronoi_plot_2d
from typing import Tuple

# internal imports
from Experiment import Experiment
from GlobalPlanning import GlobalPathPlanner
from World import * 
from Agent import *
from experiments.experiment_small import plot_results

def run(name, exec, **kwargs):
    print("Running test " + name + " ...")
    exec(**kwargs)
    print("Test " + name + " passed ...")

def shift_time():
    traj = Trajectory(np.array([[-10,20,0.2,0.01,-1.0]]), np.array([0,1,2,3,4]))
    traj.shiftTime(1)
    assert(np.all(traj.t == np.array([1,2,3,4,5])))                 

def test_random_boundary_point(n_sets = 10):
    ex = Experiment()
    assert(isinstance(ex, Experiment))
    ex.AddRandomVoronoiPoints(n_sets=10)
    ex.AddRandomTargets()
    ex.GeneratePartitioning()
    ex.PlotWorld()
    p = ex._world.target(0).region().RandomBoundaryPoint()
    plt.plot(p[0][0], p[0][1], 'gd')
    plt.show()

def test_voronoi(M = 10):
    points = []
    for i in range(M):
        points.append(np.array((np.random.uniform(0, 1), np.random.uniform(0, 1))))
        
    vor = Voronoi(points)
    voronoi_plot_2d(vor)
    plt.show()

    return

def test_plot_world(n_sets=10):
    ex = Experiment.generate(n_sets=n_sets)
    fig, ax = ex.PlotWorld()
    return ex  

def test_dist_to_boundary(n_sets=10):
    ex = Experiment.generate(n_sets=n_sets)
    fig, ax = ex.PlotWorld()
    ex._world.plotDistToBoundary(ax)
    return ex

def test_travel_cost():
    ex = Experiment.generate()
    fig, ax = ex.PlotWorld()
    ex._world.plotTravelCostPerRegion(ax)
    return ex
    
def test_rrt(max_iter = 100, n_sets=20, plot = True) -> GlobalPathPlanner:
    ex = Experiment.generate(n_sets=n_sets)
    assert(isinstance(ex, Experiment))
    gpp = GlobalPathPlanner(ex._world)
    gpp._plot_options.toggleAllPlotting(plot)
    gpp._plot_options._par = False
    gpp._plot_options._psr = False
    fig, ax = ex.PlotWorld()
    gpp.planPath(ex._world.target(1).p(), ex._world.target(9).p(), max_iter, ax)
    ex._world.plotTravelCostPerRegion(ax)
    plt.tight_layout()
    return gpp

def test_tsp(n_sets=20, plot = False) -> GlobalPathPlanner:
    ex = Experiment.generate(n_sets=n_sets, fraction=0.2)
    assert(isinstance(ex, Experiment))
    gpp = GlobalPathPlanner(ex._world)
    gpp._plot_options.toggleAllPlotting(plot)
    fig, ax = ex.PlotWorld()
    gpp.solveTSP()
    # gpp.tsp.plotTargetDistances(ax)
    gpp.plotTSPSolution(ax, color='red', linewidth=2)
    po = ex._agent.plotSensorQuality(ax)
    gpp.plotTSPSolution(ax, color='red', linewidth=2)
    return gpp

def test_local_controller(n_sets=20) -> Experiment:
    ex = Experiment.generate(n_sets=n_sets)
    assert(isinstance(ex, Experiment))
    
    target = ex._world.targets()[0]
    sensor = ex._agent.sensor()

    phi = SwitchingPoint(target.region().RandomBoundaryPoint())
    psi = SwitchingPoint(target.region().RandomBoundaryPoint())
    tf = 10
    Omega0 = {}
    for target in ex._world.targets():
        Omega0[target] = np.eye(1)
    lmp = SwitchingParameters(phi=phi,psi=psi,tf=tf,Omega0=Omega0)
    mc = MonitoringController(target, sensor)

    mc.buildOptimalMonitoringSolver(target, sensor)
    tp, tmse, tomega, tu, cost = mc.optimalMonitoringControl(lmp)

    ex, ax = ex.PlotWorld(with_sensor_quality=False, savefig=False)
    tp.plotStateVsState(0,1, ax)
    return ex

def test_cycle(n_sets=20) -> Experiment:
    ex = Experiment.generate(n_sets=n_sets)
    assert(isinstance(ex, Experiment))
    
    target = ex._world.targets()[0]
    sensor = ex._agent.sensor()

    ex._agent.computeVisitingSequence()
    ex._agent.initializeCycle()
    ex._agent._cycle.simulate()
    plot_results(ex, wsq=False, savefig=False, leave_open=True)
    return ex

def test_bilevel_optimization(n_sets=20) -> Experiment:
    ex = Experiment.generate(n_sets=n_sets)
    assert(isinstance(ex, Experiment))
    
    target = ex._world.targets()[0]
    sensor = ex._agent.sensor()

    ex._agent.computeVisitingSequence()
    ex._agent.optimizeCycle()

    plot_results(ex, wsq=True, savefig=False, leave_open=True)

    return ex