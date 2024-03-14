from Experiment import Experiment
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

def shift_time():
    traj = Trajectory(np.array([[-10,20,0.2,0.01,-1.0]]), np.array([0,1,2,3,4]))
    traj.shiftTime(1)
    assert(np.all(traj.t == np.array([1,2,3,4,5])))                 

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
        po = ex._agent.plotSensorQuality(ax)
        if savefig:
            export('.sample_mission_space_w_quality')

    return fig, ax

def plot_results(ex : Experiment, wsq = True, savefig = False, leave_open = True):
    fig, ax = plot_world(ex, with_sensor_quality=wsq, savefig=False)
    ex._agent.plotCycle(ax)
    if savefig:
        export(ex._name + '_cycle')

    fig2, ax2 = plt.subplots()
    ex._agent.plotMSE(ax2, add_labels=True)
    ax2.legend()
    if savefig:
        export(ex._name + '_mse')

    fig3, ax3 = plt.subplots()
    ex._agent.plotControls(ax3)
    if savefig:
        export(ex._name + '_controls')

    fig4, ax4 = plt.subplots(4,1)
    gca : plt.Axes = ax4[0]
    ggn : plt.Axes = ax4[1]
    tva : plt.Axes = ax4[2]
    kkt : plt.Axes = ax4[3]
    ex._agent.plotGlobalCosts(ax4[0])
    gca.set_title('Global Costs')
    ex._agent.plotGlobalGradientNorms(ax4[1])
    ggn.set_title('Global Gradient Norms')
    ex._agent.plotTauVals(tva)
    tva.set_title('Tau Values')
    ex._agent.plotKKTViolations(kkt)
    kkt.set_title('KKT Violations')

    if leave_open:
        plt.ioff()
        plt.show()

def test_plot_world(n_sets=10):
    ex = generate_partitioning(n_sets=n_sets)
    fig, ax = plot_world(ex)
    return ex  

def test_dist_to_boundary(n_sets=10):
    ex = generate_partitioning(n_sets=n_sets)
    fig, ax = plot_world(ex)
    ex._world.plotDistToBoundary(ax)
    return ex

def test_travel_cost():
    ex = generate_partitioning()
    fig, ax = plot_world(ex)
    ex._world.plotTravelCostPerRegion(ax)
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
    ex._world.plotTravelCostPerRegion(ex, ax)
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
    po = ex._agent.plotSensorQuality(ex, ax)
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
    Omega0 = {}
    for target in ex._world.targets():
        Omega0[target] = np.eye(1)
    lmp = SwitchingParameters(phi=phi,psi=psi,tf=tf,Omega0=Omega0)
    mc = MonitoringController(target, sensor)

    mc.buildOptimalMonitoringSolver(target, sensor)
    tp, tmse, tomega, tu, cost = mc.optimalMonitoringControl(lmp)

    ex, ax = plot_world(ex, with_sensor_quality=False, savefig=False)
    tp.plotStateVsState(0,1, ax)
    return ex

def test_cycle(n_sets=20) -> Experiment:
    ex = generate_partitioning(n_sets=n_sets)
    assert(isinstance(ex, Experiment))
    
    target = ex._world.targets()[0]
    sensor = ex._agent.sensor()

    ex._agent.computeVisitingSequence()
    ex._agent.initializeCycle()
    ex._agent._cycle.simulate()
    plot_results(ex, wsq=False, savefig=False, leave_open=True)
    return ex

def test_bilevel_optimization(n_sets=20) -> Experiment:
    ex = generate_partitioning(n_sets=n_sets)
    assert(isinstance(ex, Experiment))
    
    target = ex._world.targets()[0]
    sensor = ex._agent.sensor()

    ex._agent.computeVisitingSequence()
    ex._agent.optimizeCycle()

    plot_results(ex, wsq=True, savefig=False, leave_open=True)

    return ex