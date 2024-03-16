
import os
from Experiment import *
from unittests import *

# plt.rcParams['text.usetex'] = True

exp_name = "large"
out_dir = os.path.join("/home/jonas/PhD/papers/CDC2024/figures/experiments", exp_name)

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

# visualize the results immediately
generate_plots = False

if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

# Generate experiment
def load_experiment() -> Experiment:
    if not os.path.exists(exp_file):
        ex = generate_experiment(n_sets=20, fraction=0.5, seed=987)
        ex._name = exp_filename
        assert(isinstance(ex, Experiment))
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

# Initial cycle
def load_initial_cycle(ex : Experiment) -> Experiment:
    if not os.path.exists(exp_init_file):
        ex._agent.initializeCycle()
        ex._agent._cycle.simulate()
        ex.serialize(exp_init_file)
    else:
        ex : Experiment = Experiment.deserialize(exp_init_file)
    return ex

# Optimize cycle
def load_optimized_cycle(ex : Experiment) -> Experiment:
    if not os.path.exists(exp_res_file):
        ex._agent._op = op
        ex._agent._op._beta = 0.95
        ex._agent._op._kkt_tolerance = 1e-3
        ex._agent._op._sim_to_steady_state_tol = 1e-1
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

    fig, ax = plot_world(hl, with_sensor_quality=True)

    po = PlotObject()
    po.add(hl._agent.gpp().PlotTSPSolution(ax=ax, annotate=False, color='red', linewidth=2.5, alpha=0.5))
    po._objs[0].set_label('$\mathrm{TSP}$')
    export(directory=out_dir, name="tsp")

    po.add(init._agent.plotCycle(ax=ax, color='yellow', alpha=0.75, linewidth=2.25, label='$\mathrm{initial~cycle}$', linestyle='--'))
    export(directory=out_dir, name="tsp_vs_init")

    po.add(res._agent.plotCycle(ax=ax, label='$\mathrm{optimal~cycle}$'))
    ax.legend(loc='center left')
    export(directory=out_dir, name="tsp_vs_init_vs_opt")

    if leave_open:
        plt.ioff()
        plt.show()

# Initial MSE vs Optimal MSE
def mse_vs_opti(leave_open=False):
    hl = load_high_level_solution(load_experiment())
    init = load_initial_cycle(hl)
    res = load_optimized_cycle(hl)

    startTimeInit = init._agent._cycle.getStartTime()
    startTimeRes = res._agent._cycle.getStartTime()
    init._agent._cycle.shiftTime(-startTimeInit)
    res._agent._cycle.shiftTime(-startTimeRes)

    fig, ax = plt.subplots()
    init._agent.plotMSE(ax, add_labels=True)

    plotAttributes.target_colors = plotAttributes.target_colors[0:-3]
    res._agent.plotMSE(ax, add_labels=False)
    ax.set_ylabel('$tr(\Omega_i(t))$')
    ax.set_ylim(bottom=0)
    # ax.set_xlim(0, init._agent._cycle.getDuration())
    ax.set_xlabel('$t~[s]$')
    # plt.legend(loc="lower center", ncol=len(init._world.targets()))
    export(directory=out_dir, name='mse_vs_opti')

    init._agent._cycle.shiftTime(startTimeInit)
    res._agent._cycle.shiftTime(startTimeRes)

    if leave_open:
        plt.ioff()
        plt.show()

# Create optimization plots
def plot_results(ex : Experiment, wsq = True, savefig = True, leave_open = False):
    
    # shift cycle to relative time
    startTime = ex._agent._cycle.getStartTime()
    ex._agent._cycle.shiftTime(-startTime)

    fig, ax = plot_world(ex, with_sensor_quality=wsq, savefig=False)
    ex._agent.plotCycle(ax)
    if savefig:
        export(directory=out_dir, name='cycle')

    fig2, ax2 = plt.subplots()
    ex._agent.plotMSE(ax2, add_labels=True)
    ax2.set_ylabel('$tr(\Omega_i(t))$')
    ax2.set_ylim(bottom=0)
    ax2.set_xlim(0, ex._agent._cycle.getDuration())
    ax2.set_xlabel('$t~[s]$')
    plt.legend(loc="lower center", ncol=len(ex._world.targets()))
    if savefig:
        export(directory=out_dir, name='mse')

    fig3, ax3 = plt.subplots()
    ex._agent.plotControls(ax3)
    ax3.set_ylim(-1.3, 1.3)
    ax3.set_ylabel('$\mathrm{controls}$')
    ax3.set_xlim(0, ex._agent._cycle.getDuration())
    ax3.set_xlabel('$t~[s]$')
    plt.legend(loc="upper center", ncol=3)
    if savefig:
        export(directory=out_dir, name='controls')

    fig4, ax4 = plt.subplots(3,1, sharex=True)
    gca : plt.Axes = ax4[0]
    ggn : plt.Axes = ax4[1]
    tva : plt.Axes = ax4[2]
    # kkt : plt.Axes = ax4[3]
    ex._agent.plotGlobalCosts(ax4[0])
    gca.set_ylabel('$\mathrm{cycle cost}$')
    ex._agent.plotGlobalGradientNorms(ax4[1])
    ggn.set_ylabel('$\mathrm{grad. norm}$')
    ex._agent.plotTauVals(tva, linewidth=2)
    tva.set_ylabel('$\tau$')
    # ex._agent.plotKKTViolations(kkt)
    # kkt.set_yscale('symlog')
    # kkt.set_ylabel('KKT res.')
    # kkt.set_xlabel('iteration')
    if savefig:
        export(directory=out_dir, name='optimization')

    # Shift back to absolute time
    ex._agent._cycle.shiftTime(startTime)

    if leave_open:
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    plot_results(load_optimized_cycle(load_high_level_solution(load_experiment())))
    tsp_vs_init_vs_opti()
