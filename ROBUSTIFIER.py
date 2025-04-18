
from hytoperm.Decomposition import *
from experiments.Spherical.small_homogeneous import ex, op
import numpy as np
import time
import pickle

# sensor
ex.agent().initializeDecomposition()
sensor = ex.agent().sensor()

# initialize agent
# experiment settings
num_exp = 100
target_idx = 2
N = 200
mc = MonitoringController(ex.world().target(target_idx), sensor, N = N)

# Randomized tests for a single monitoring segment
noise = 1e-3
o_range =   [ 3.52978373,  3.52978373]
phi_range = [-1.10801198, -1.10801198]
psi_range = [-2.09687591, -2.09687591]
tau_range = [ 1.40909349-noise,  1.40909349+noise]

# stats
Omegas = []
Phis = []
Psis = []
Taus = []
tau_facts = []
tau_relaxs = []
min_dts = []
cpu_times = []
solved = []

# params
r = ex.world().target(target_idx).region()
params = LocalMonitoringParameters(r=r, a_phi=np.zeros(2), a_psi=np.zeros(2), tf=0.0)

def next_params():
    Phis.append(np.random.uniform(phi_range[0], phi_range[1]))
    Psis.append(np.random.uniform(psi_range[0], psi_range[1]))
    Omegas.append(np.random.uniform(o_range[0], o_range[1]))
    a_phi = r.getBoundaryPoint(alpha=Phis[-1])
    a_psi = r.getBoundaryPoint(alpha=Psis[-1])
    min_dts.append(r.travelCost(a_phi, a_psi))
    Taus.append(np.random.uniform(tau_range[0], tau_range[1]))
        
    params.fromVector([Taus[-1], Phis[-1], Psis[-1]])
    params._Omega0 = Omegas[-1]

    mc.solver.params = cad.vertcat(*[
        params.getStartPoint().p(), 
        params.getEndPoint().p(), 
        np.array(params.getDuration()), 
        np.matrix(Omegas[-1]).getA1()
    ])

for i in range(num_exp):
    next_params()

    # solve
    start = time.time()
    sol = mc.solver.solve()
    elapsed = time.time() - start

    # save results
    solved.append(mc.solver.solver.stats()['success'])
    cpu_times.append(elapsed)

    # print update
    print(f"Experiment {i}/{num_exp}\t succeeded {sum(solved)} ({sum(solved)/(i+1) * 100}%)")

pickle.dump(
    {
        'Omegas': Omegas,
        'Phis': Phis,
        'Psis': Psis,
        'Taus': Taus,
        'tau_facts': tau_facts,
        'tau_relaxs': tau_relaxs,
        'min_dts': min_dts,
        'cpu_times': cpu_times,
        'solved': solved
    }, open('results.pkl', 'wb')
)

# load results
with open('results.pkl', 'rb') as f:
    data = pickle.load(f)
    Omegas = data['Omegas']
    Phis = data['Phis']
    Psis = data['Psis']
    Taus = data['Taus']
    tau_facts = data['tau_facts']
    min_dts = data['min_dts']
    cpu_times = data['cpu_times']
    solved = data['solved']

# plot results
import matplotlib.pyplot as plt
for i in range(len(Phis)):
    color = 'green' if solved[i] else 'red'
    marker = 's'
    #plt.scatter(Phis[i], tau_facts[i], color=color, marker=marker, s=10, alpha=0.66)
    #plt.scatter(Phis[i], Taus[i] - min_dts[i], color=color, marker=marker, s=10, alpha=0.66)
    #plt.scatter(Phis[i], Omegas[i], color=color, marker=marker, s=10, alpha=0.66)
    plt.scatter(Phis[i], Taus[i], color=color, marker=marker, s=10, alpha=0.66)
    plt.xlabel('phi')
# plt.plot(Phis, min_dts, 'bx', label='min_dt')
plt.yscale('log')
plt.legend()

plt.show()