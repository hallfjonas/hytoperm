
from hytoperm.Experiment import *
from hytoperm.BTO import BTO
from export import exporter
import matplotlib.pyplot as plt  

##############
## plotting ##
##############
def create_plots(ex: Experiment, bto: BTO):
    
    # set exporter dir
    exporter.DIR = os.path.join(exporter.DIR, ex._name)
    
    ag = ex.agent()

    # Plot final trajectory
    fig_cycle, ax_cycle = plt.subplots()
    ex.plotWorld(ax=ax_cycle, with_sensor_quality=True)
    exporter.export("world", fig = fig_cycle)
    ag.plotCycle(ax=ax_cycle, linestyle='-', label="optimized")
    exporter.export("optimized_trajectory", fig = fig_cycle)

    fig, ax_mse = plt.subplots()
    ag.plotMSE(ax=ax_mse, linestyle='-')
    exporter.export("optimized_mse", fig = plt.gcf())

    # plot cost 
    fig, ax_cost = plt.subplots()
    ax_cost.plot(bto._stats.global_costs)
    ax_cost.set_xlabel("iteration")
    ax_cost.set_ylabel("cost")
    exporter.export("cost", fig = fig)

        
    # plot tau/phi/psi vals
    tau_values = np.array(bto._stats.tau_values)
    fig_tau, ax_tau = plt.subplots(); plt.xlabel("iteration"); ax_tau.set_ylabel("$\\tau$")
    for k, target in enumerate(ag._tvs):
        ax_tau.plot(tau_values[:,k], label=f"$\\tau_{target.name}$")
    ax_tau.legend()
    exporter.export("tau_vals", fig = fig_tau)

    phi_values = np.array(bto._stats.phi_values)
    fig_phi, ax_phi = plt.subplots(); plt.xlabel("iteration"); ax_phi.set_ylabel("$\\varphi$")
    for k, target in enumerate(ag._tvs):
        ax_phi.plot(phi_values[:,k], label=f"$\\varphi_{target.name}$")
    ax_phi.legend()
    exporter.export("phi_vals", fig = fig_phi)

    psi_values = np.array(bto._stats.psi_values)
    fig_psi, ax_psi = plt.subplots(); plt.xlabel("iteration"); ax_psi.set_ylabel("$\\psi$")
    for k, target in enumerate(ag._tvs):
        ax_psi.plot(psi_values[:,k], label=f"$\\psi_{target.name}$")
    ax_psi.legend()
    exporter.export("psi_vals", fig = fig_psi)

    # plot global gradient norms
    fig, ax_grad = plt.subplots()
    ax_grad.plot(bto._stats.global_gradient_norms)
    ax_grad.set_xlabel("iteration")
    ax_grad.set_ylabel("gradient norm")
    ax_grad.set_yscale('log')
    exporter.export("global_gradient_norms", fig = fig)

    # plot steady state difference of Omega0
    fig, ax_grad = plt.subplots()
    ax_grad.plot(bto._stats.steady_state_violations)
    ax_grad.set_xlabel("iteration")
    ax_grad.set_ylabel("steady state violations")
    ax_grad.set_yscale('log')
    exporter.export("steady_state_violations", fig = fig)
