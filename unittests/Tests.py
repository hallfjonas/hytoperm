from experiment import Experiment
from HighLevel import GlobalPathPlanner
from World import * 
import matplotlib.pyplot as plt
import numpy as np  
from scipy.spatial import Voronoi, voronoi_plot_2d
from typing import Tuple
from Plotters import export
from Agent import *

def run(name, exec, **kwargs):
    print("Running test " + name + " ...")
    exec(**kwargs)
    print("Test " + name + " passed ...")

def empty_plot_figure():
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.axis('off')
    plt.ion()
    plt.show()
    fig.tight_layout()
    return fig, ax

# Nice seeds for 10 sets and 0.5 fraction: 235
def generate_partitioning(n_sets=10,fraction=0.5,seed=235) -> Experiment:
    if seed is not None:
        np.random.seed(seed)
    ex = Experiment()
    ex.AddRandomVoronoiPoints(n_sets)
    ex.GeneratePartitioning()
    ex.AddRandomTargets(fraction=fraction)
    ex.AssignRandomAgent()
    return ex

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

def plot_world(ex : Experiment, with_sensor_quality=False, savefig = False) -> Tuple[Experiment, plt.Axes]:
    fig, ax = empty_plot_figure()
    ex.PlotWorld(ax)

    if savefig:
        export('.sample_mission_space')

    if with_sensor_quality:
        po = plot_sensor_quality_per_region(ex,ax)
        if savefig:
            export('.sample_mission_space_w_quality')

    return fig, ax

def test_plot_world(n_sets=10):
    ex = generate_partitioning(n_sets=n_sets)
    fig, ax = plot_world(ex)
    return ex

def get_meshgrid(ex : Experiment):
    dx = 0.005; dy = 0.005
    x = np.arange(ex._world.domain().xmin()-0.5*dx, ex._world.domain().xmax()+0.5*dx, dx)
    y = np.arange(ex._world.domain().ymin()-0.5*dy, ex._world.domain().ymax()+0.5*dy, dy)
    X, Y = np.meshgrid(x, y)
    Z = np.nan*np.ones(X.shape)
    return X, Y, Z

def plot_dist_to_boundary(ex : Experiment, ax : plt.Axes = plt):
    X, Y, Z = get_meshgrid(ex)
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            p = np.array((X[i,j], Y[i,j]))
            for region in ex._world.regions():
                if region.Contains(p):
                    Z[i,j] = region.DistToBoundary(p)
    ax.contourf(X, Y, Z, antialiased=True, alpha=0.5)

def plot_travel_cost_per_region(ex : Experiment, ax : plt.Axes = plt):
    X, Y, Z = get_meshgrid(ex)
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            p = np.array((X[i,j], Y[i,j]))
            for region in ex._world.regions():
                assert(isinstance(region, CPRegion))
                if region.Contains(p):
                    Z[i,j] = region.TravelCost(p, region.p())
                    if Z[i,j] == np.inf:
                        Z[i,j] = -region.TravelCost(region.p(), p)
    cf = ax.contourf(X, Y, Z, antialiased=True, alpha=0.5)
    # plt.colorbar(cf)    

def plot_sensor_quality_per_region(ex : Experiment, ax : plt.Axes = plt) -> PlotObject:
    X, Y, Z = get_meshgrid(ex)
    sensor = ex._agent.sensor()
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            p = np.array((X[i,j], Y[i,j]))
            sensor.setPosition(p)
            for target in ex._world.targets():
                assert(isinstance(target, Target))
                region = target.region()
                if region.Contains(p):
                    Z[i,j] = sensor.getSensingQuality(target)
    return PlotObject(ax.contourf(X, Y, Z, antialiased=True, alpha=0.5))
    # plt.colorbar(cf)

def test_dist_to_boundary(n_sets=10):
    ex = generate_partitioning(n_sets=n_sets)
    fig, ax = plot_world(ex)
    plot_dist_to_boundary(ex, ax)
    return ex

def test_travel_cost():
    ex = generate_partitioning()
    fig, ax = plot_world(ex)
    plot_travel_cost_per_region(ex, ax)
    return ex
    
def test_rrt(max_iter = 1000, n_sets=20, plot = True) -> GlobalPathPlanner:
    ex = generate_partitioning(n_sets=n_sets)
    assert(isinstance(ex, Experiment))
    gpp = GlobalPathPlanner(ex._world)
    gpp._plot_options.toggle_all_plotting(plot)
    gpp._plot_options._par = False
    gpp._plot_options._psr = False
    fig, ax = plot_world(ex)
    gpp.PlanPath(ex._world.target(1).p(), ex._world.target(9).p(), max_iter, ax)
    export('.sample_rrt')
    plot_travel_cost_per_region(ex, ax)
    plt.tight_layout()
    export('.sample_rrt_w_travel_cost')
    return gpp

def test_tsp(n_sets=20, plot = False) -> GlobalPathPlanner:
    ex = generate_partitioning(n_sets=n_sets)
    assert(isinstance(ex, Experiment))
    gpp = GlobalPathPlanner(ex._world)
    gpp._plot_options.toggle_all_plotting(plot)
    fig, ax = empty_plot_figure()
    ex.PlotWorld(ax)
    gpp.SolveTSP(ax)
    # gpp.tsp.PlotTargetDistances(ax)
    gpp.PlotTSPSolution(ax, color='red', linewidth=2)
    export('.sample_tsp')
    po = plot_sensor_quality_per_region(ex, ax)
    gpp.PlotTSPSolution(ax, color='red', linewidth=2)
    export('.sample_tsp_w_sensing_quality')
    return gpp

def test_local_controller(n_sets=20) -> Experiment:
    ex = generate_partitioning(n_sets=n_sets)
    assert(isinstance(ex, Experiment))
    
    target = ex._world.targets()[0]
    sensor = ex._agent.sensor()

    phi = SwitchingPoint(target.region().RandomBoundaryPoint())
    psi = SwitchingPoint(target.region().RandomBoundaryPoint())
    tf = 10
    Omega0 = np.eye(1)
    lmp = MonitoringParameters(phi=phi,psi=psi,tf=tf,Omega0=Omega0)
    mc = MonitoringController(target, sensor)

    mc.buildOptimalMonitoringSolver(target, sensor)
    tp, tmse, tu = mc.optimalMonitoringControl(lmp)

    ex, ax = plot_world(ex, with_sensor_quality=False, savefig=False)
    tp.plotStateVsState(0,1, ax)
    return ex
