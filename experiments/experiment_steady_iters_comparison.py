
# external imports
import os
from matplotlib.ticker import MaxNLocator

# internal imports
from src.Experiment import *
from experiments.large_comparison.config import *


##################################
## NO NEED TO MAKE CHANGES HERE ##
##################################
exp_name = NAME
exp_dir = os.path.join("experiments", exp_name)
exp_filename = "exp"
exp_hl_filename = exp_filename + "_hl"
exp_steady_filename = exp_filename + "_steady"
exp_non_steady_filename = exp_filename + "_non_steady"
pickle_extension = ".pickle"

exp_hl_file = os.path.join(exp_dir, exp_filename + pickle_extension)
exp_res_steady_file = os.path.join(exp_dir, exp_steady_filename + pickle_extension)
exp_res_non_steady_filename = os.path.join(exp_dir, exp_non_steady_filename + pickle_extension)

if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

# Generate experiment
def load_hl_sol() -> Experiment:
    if not os.path.exists(exp_hl_file):
        ex = Experiment.generate(n_sets=NSETS, fraction=FRACTION, seed=SEED)
        ex._name = exp_filename
        ex._agent.computeVisitingSequence()
        ex.serialize(exp_hl_file)
    else:
        ex : Experiment = Experiment.deserialize(exp_hl_file)
    return ex

# Steady state
def load_solution(filename, op : OptimizationParameters) -> Experiment:
    if not os.path.exists(filename):
        ex = load_hl_sol()
        ex._agent.op = op.copy()
        ex._agent.optimizeCycle()
        ex.serialize(filename)
    else:
        ex : Experiment = Experiment.deserialize(filename)
    return ex

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
    for i in range(len(ex._agent._isSteadyState)):
        if ex._agent._isSteadyState[i]:
            gca.axvspan(i, i+1, color='green', alpha=0.2)
        else:
            gca.axvspan(i, i+1, color='red', alpha=0.2)

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

def steady_vs_non(steady : Experiment, non_steady : Experiment, savefig = True):
    fig4, ax4 = plt.subplots(2,1, sharex=True)

    xticks_steady = np.cumsum(steady._agent._steady_state_iters)
    xticks_non_steady = np.cumsum(non_steady._agent._steady_state_iters)

    # global cost plot
    gca : plt.Axes = ax4[0]
    gca.plot([i+1 for i in range(len(steady._agent._global_costs))], steady._agent._global_costs, linewidth=2, color='orange', label='$\mathrm{with~steady~state~iterations}$')
    gca.plot([i+1 for i in range(len(non_steady._agent._global_costs))], non_steady._agent._global_costs, linewidth=2, color='blue', label='$\mathrm{no~steady~state~iterations}$')
    gca.set_ylabel('$J$')
    # gca.legend(loc='lower right')
    
    # gradient plot
    # ggn : plt.Axes = ax4[1]
    # ggn.step(xticks_steady, steady._agent._global_gradient_norms, linewidth=2, color='orange')
    # ggn.step(xticks_non_steady, non_steady._agent._global_gradient_norms, linewidth=2, color='blue')
    # ggn.set_ylabel('$\\| \\nabla_\\tau J \\|_\\infty$')
    # ggn.set_yscale('log')
    # ggn.set_yticks([0.1, 1.0])
    # ggn.set_yticklabels(['$0.1$', '$1.0$'])
        
    # plot the tau values
    tva : plt.Axes = ax4[1]
    xtau_steady = np.concatenate(([1], xticks_steady))
    ytau_steady = np.concatenate(([steady._agent._tau_vals[0][0]], np.array(steady._agent._tau_vals)[:,0]))
    tva.step(xtau_steady, ytau_steady, linewidth=2, color='orange')
    tva.step(xticks_non_steady, np.array(non_steady._agent._tau_vals)[:,0], linewidth=2, color='blue')
    tva.set_ylabel('$ \\tau_1$')
    tva.set_xlabel('$\mathrm{cycle~}$')
    tva.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # add steady state lines
    steady._agent.addSteadyStateLines(ax4[0], alpha=0.1, color='black')
    steady._agent.addSteadyStateLines(ax4[1], alpha=0.1, color='black')
    
    if savefig:
        exporter.HEIGHT = exporter.WIDTH*0.8*2/3
        exporter.export('optimization_steady_vs_non_steady', fig4)

if __name__ == '__main__':
    op.steady_state_iters = 1
    op.sim_to_steady_state_tol = 1e-1
    non_steady = load_solution(exp_res_non_steady_filename, op)

    op.steady_state_iters = 1000
    steady = load_solution(exp_res_steady_file, op)

    # compare cycle
    fig, ax = steady.plotWorld(steady, with_sensor_quality=True, add_target_labels=False)
    steady._agent.plotCycle(ax, linestyle='-', color='orange', linewidth=2, label='$\mathrm{with~steady~state~iterations}$')
    non_steady._agent.plotCycle(ax, linestyle='--', color='blue', linewidth=2, label='$\mathrm{no~steady~state~iterations}$')
    exporter.export('world_steady_vs_non_steady', fig)

    # mse comparison plot
    fig2, ax2 = plt.subplots()
    steady._agent._cycle.shiftTime(-steady._agent._cycle._cycle_start)
    steady._agent.plotMSE(ax2, add_labels=True, linewidth=2)
    non_steady._agent._cycle.shiftTime(-non_steady._agent._cycle._cycle_start)
    non_steady._agent.plotMSE(ax2, add_labels=True, linewidth=2, linestyle='--')
    exporter.HEIGHT = exporter.WIDTH*0.33
    exporter.export('mse_steady_vs_non_steady', fig2)

    # optimization comparison plot
    steady_vs_non(steady, non_steady)
    