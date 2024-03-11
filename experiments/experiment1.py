
import os
from Experiment import *
from unittests import *

exp_file = "experiments/ex1.p"

if not os.path.exists(exp_file):
    ex = generate_partitioning(n_sets=10, seed=754)
    assert(isinstance(ex, Experiment))

    target = ex._world.targets()[0]
    sensor = ex._agent.sensor()

    ex._agent.computeVisitingSequence()
    ex._agent.initializeCycle()
    ex.serialize(exp_file)
else:
    ex : Experiment = Experiment.deserialize(exp_file)

fig, ax = plot_world(ex, with_sensor_quality=True, savefig=False)
ex._agent.initializeCycle()
ex._agent.simulateToSteadyState(maxIter=2)
ex._agent.plotControls()
# ex._agent.gpp().PlotTSPSolution(ax, color='red', linewidth=2)
ex._agent.plotSwitchingSegments(ax, color='green')
ex._agent.plotMonitoringSegments(ax, color='blue')
ex._agent.plotSwitchingPoints(ax, marker='o', color='red', markersize=5)

fig2, ax2 = plt.subplots()
ex._agent.plotMSE(ax2)
ax2.legend()

print("DONE!")