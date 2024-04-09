
# external imports
import os
from matplotlib.ticker import MaxNLocator

# internal imports
from hytoperm import *
from experiments.large.config import *


##################################
## NO NEED TO MAKE CHANGES HERE ##
##################################
exp_name = NAME
exp_dir = os.path.join("experiments", exp_name)
exp_filename = "exp"
exp_hl_filename = exp_filename + "_hl"
exp_init_filename = exp_filename + "_init" 
exp_res_filename = exp_filename + "_res"
pickle_extension = ".pickle"

exp_file = os.path.join(exp_dir, exp_filename + pickle_extension)
exp_hl_file = os.path.join(exp_dir, exp_hl_filename + pickle_extension)
exp_init_file = os.path.join(exp_dir, exp_init_filename + pickle_extension)
exp_res_file = os.path.join(exp_dir, exp_res_filename + pickle_extension)

# adapt the optimization parameters
op = OptimizationParameters()
op.kkt_tolerance = 1e-1
op.sim_to_steady_state_tol = 1e-1
op.optimization_iters = 100
op.steady_state_iters = 1
op.beta = 0.95

if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

# Generate experiment
def load_experiment() -> Experiment:
    if not os.path.exists(exp_file):
        ex = Experiment.generate(n_sets=NSETS, fraction=FRACTION, seed=SEED)
        ex._name = exp_filename
        if not isinstance(ex, Experiment)
        ex.serialize(exp_file)
    else:
        ex : Experiment = Experiment.deserialize(exp_file)
    return ex

# High level solution
def load_high_level_solution(ex : Experiment) -> Experiment:
    if not os.path.exists(exp_hl_file):
        target = ex._world.targets()[0]
        sensor = ex._agent.sensor()
        ex._agent.computeVisitingSequence()
        ex.serialize(exp_hl_file)
    else:
        ex : Experiment = Experiment.deserialize(exp_hl_file)
    return ex

# Initial (steady state) cycle
def load_initial_cycle(ex : Experiment) -> Experiment:
    if not os.path.exists(exp_init_file):
        ex._agent.initializeCycle()
        ex._agent.simulateToSteadyState()
        ex.serialize(exp_init_file)
    else:
        ex : Experiment = Experiment.deserialize(exp_init_file)
    return ex

# Optimize cycle
def load_optimized_cycle(ex : Experiment) -> Experiment:
    if not os.path.exists(exp_res_file):
        ex._agent.op = op
        ex._agent.optimizeCycle()
        ex.serialize(exp_res_file)
    else:
        ex : Experiment = Experiment.deserialize(exp_res_file)
    return ex

# Create plots for the TSP, initial and optimized cycle
def tsp_vs_init_vs_opti(leave_open=False):
    
    hl = load_high_level_solution(load_experiment())
    init = load_initial_cycle(hl)
    res = load_optimized_cycle(hl)

    fig, ax = hl.plotWorld(with_sensor_quality=True)

    po = PlotObject()
    po.add(hl._agent.gpp().plotTSPSolution(ax=ax, annotate=False, color='red', linewidth=2.5, alpha=0.5))
    po._objs[0].set_label('$\mathrm{TSP}$')
    exporter.export("tsp", fig=fig)

    po.add(init._agent.plotCycle(ax=ax, color='yellow', alpha=0.75, linewidth=2.25, label='$\mathrm{initial~cycle}$', linestyle='--'))
    exporter.export("tsp_vs_init", fig=fig)

    po.add(res._agent.plotCycle(ax=ax, label='$\mathrm{optimal~cycle}$'))
    ax.legend(loc='lower left', framealpha=1)
    exporter.export("tsp_vs_init_vs_opt", fig=fig)

    if leave_open:
        plt.ioff()
        plt.show()

# Initial MSE vs Optimal MSE
def mse_inti_vs_opti(leave_open=False):
    hl = load_high_level_solution(load_experiment())
    init = load_initial_cycle(hl)
    res = load_optimized_cycle(hl)

    startTimeInit = init._agent._cycle.getStartTime()
    startTimeRes = res._agent._cycle.getStartTime()
    init._agent._cycle.shiftTime(-startTimeInit)
    res._agent._cycle.shiftTime(-startTimeRes)

    fig, ax = plt.subplots()
    
    po = init._agent.plotMSE(ax, add_labels=True, linestyle='--', alpha=0.7)
    po = res._agent.plotMSE(ax, add_labels=False)
    ax.set_ylabel('$tr(\Omega_i(t))$')
    ax.set_ylim(bottom=0)
    ax.set_xlabel('$t~[s]$')
    plt.legend()
    exporter.export('mse_init_vs_opti', fig)

    init._agent._cycle.shiftTime(startTimeInit)
    res._agent._cycle.shiftTime(startTimeRes)

    if leave_open:
        plt.ioff()
        plt.show()

# Create optimization plots
def plotResults(ex : Experiment, wsq = True, savefig = True, leave_open = False):
    
    # shift cycle to relative time
    startTime = ex._agent._cycle.getStartTime()
    ex._agent._cycle.shiftTime(-startTime)

    world_plot(ex, with_sensor_quality=wsq, savefig=savefig)
    mse_plot(ex, savefig=savefig)
    controls_plot(ex, savefig=savefig)
    optimization_plot(ex, savefig=savefig)

    # Shift back to absolute time
    ex._agent._cycle.shiftTime(startTime)

    if leave_open:
        plt.ioff()
        plt.show()

def world_plot(ex : Experiment, with_sensor_quality = True, savefig = True):
    
    fig, ax = ex.plotWorld(with_sensor_quality=with_sensor_quality)
    ex._agent.plotCycle(ax)
    if savefig:
        exporter.HEIGHT = exporter.WIDTH
        exporter.export('cycle', fig)

def mse_plot(ex : Experiment, savefig = True):
    fig2, ax2 = plt.subplots()
    ex._agent.plotMSE(ax2, add_labels=True, linewidth=2)
    ax2.set_ylabel('$tr(\Omega_i(t))$')
    ax2.set_ylim(top=1.3*ax2.get_ylim()[1])
    ax2.set_xlim(0, ex._agent._cycle.getDuration())
    ax2.set_xlabel('$t~[s]$')
    plt.legend(loc="upper right", ncol=len(ex._world.targets()))
    if savefig:
        exporter.HEIGHT = exporter.WIDTH/3.0
        exporter.export('mse', fig2)

def controls_plot(ex : Experiment, savefig = True):
    fig, ax = plt.subplots()
    ex._agent.plotControls(ax)
    ax.set_ylim(-1.1, 1.9)
    ax.set_ylabel('$\mathrm{control~input}$')
    ax.set_xlim(0, ex._agent._cycle.getDuration())
    ax.set_xlabel('$t~[s]$')
    plt.legend(loc="upper right", ncol=3)
    if savefig:
        exporter.HEIGHT = exporter.WIDTH/3.0
        exporter.export('controls', fig)

def optimization_plot(ex : Experiment, savefig = True):
    fig4, ax4 = plt.subplots(3,1, sharex=True)
    
    # global cost plot
    gca : plt.Axes = ax4[0]
    ex._agent.plotGlobalCosts(ax4[0], linewidth=2)
    gca.set_ylabel('$J(\\tau_k)$')

    ggn : plt.Axes = ax4[1]
    ex._agent.plotGlobalGradientNorms(ax4[1], linewidth=2)
    ggn.set_ylabel('$\\| \\nabla_\\tau J(\\tau_k) \\|_\\infty$')
    ggn.set_yscale('log')
    
    # plot the tau values
    tva : plt.Axes = ax4[2]
    ex._agent.plotTauVals(tva, linewidth=2)
    tva.set_ylabel('$ \\tau_k $')
    tva.set_xlabel('$\mathrm{outer~iteration~}k$')
    tva.set_xlim(0, len(ex._agent._tau_vals)-1)
    tva.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # plot the KKT violations
    # kkt : plt.Axes = ax4[3]
    # ex._agent.plotKKTViolations(kkt)
    # kkt.set_yscale('symlog') 
    # kkt.set_ylabel('KKT res.')
    # kkt.set_xlabel('iteration')
    if savefig:
        exporter.HEIGHT = exporter.WIDTH*0.9
        exporter.export('optimization', fig4)

def rrt_plot(savefig = True):
    ex = load_high_level_solution(load_experiment())
    fig, ax = ex.plotWorld(add_target_labels=False)
    target0 = ex._world.target(0)
    target1 = ex._world.target(1)
    po = ex._agent.gpp().target_paths[target1][target0].getRoot().plotTree(ax=ax, color='black', linewidth=1, alpha=0.2)
    po = ex._agent.gpp().target_paths[target1][target0].plotPathToRoot(ax=ax, color='red', linewidth=2, alpha=1)
    if savefig:
        exporter.export('rrt', fig)

if __name__ == '__main__':
    
    if True or not os.path.exists(exp_file):
        ex = Experiment.generate(n_sets=NSETS, fraction=FRACTION, seed=SEED)
        ex._name = exp_filename
        assert(isinstance(ex, Experiment))
        ex.serialize(exp_file)
    else:
        ex : Experiment = Experiment.deserialize(exp_file)

    fig, ax = ex.plotWorld(fill_empty_regions=False)
    rrt = RRT(ex._world.regions())
    rrt._plot_options = PlotOptions()
    rrt._plot_options.toggleAllPlotting(False)
    r, p = rrt.planPath(ex._world.targets()[0].p(), ex._world.targets()[1].p())

    rrt.best_path.getRoot().plotTree(ax=ax, color='black', linewidth=1, alpha=0.2)
    rrt.best_path.plotPathToRoot(ax=ax, color='red', linewidth=2, alpha=1)
    rrt.best_path.getData().plot(ax=ax, color='green', marker='o', markersize=10)
    rrt.best_path.getRoot().getData().plot(ax=ax, color='yellow', marker='*', markersize=10)
    exporter.HEIGHT = exporter.WIDTH*0.8
    plt.show()
    exporter.export('rrt', fig)