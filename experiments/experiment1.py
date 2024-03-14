
import os
from Experiment import *
from unittests import *

exp_name = "ex1"
exp_dir = "experiments"
exp_filename = os.path.join(exp_dir, exp_name)
pickle_extension = ".pickle"
exp_file = exp_filename + pickle_extension
res_file = exp_filename + "_results" + pickle_extension

if not os.path.exists(exp_file):
    ex = generate_partitioning(n_sets=10, seed=187)
    ex._name = exp_filename
    assert(isinstance(ex, Experiment))

    target = ex._world.targets()[0]
    sensor = ex._agent.sensor()

    ex._agent.computeVisitingSequence()
    plt.close()
    ex.serialize(exp_file)
else:
    ex : Experiment = Experiment.deserialize(exp_file)

# plot_world(ex, with_sensor_quality=True)
ex._agent.optimizeCycle(maxIter=100)

plt.close()
ex.serialize(res_file)

plot_results(ex, wsq=True, savefig=False)
print("DONE!")
