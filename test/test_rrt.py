
# external imports
import unittest

# internal imports
from hytoperm import *


class TestRRT(unittest.TestCase):
    
    def testRRT(self):
        max_iter = 100; n_sets = 20; plot = True
        ex = Experiment.generate(n_sets=n_sets)
        assert(isinstance(ex, Experiment))
        gpp = GlobalPathPlanner(ex._world)
        gpp._plot_options.toggleAllPlotting(plot)
        gpp._plot_options._par = False
        gpp._plot_options._psr = False
        fig, ax = ex.plotWorld()
        gpp.planPath(
            ex._world.target(1).p(), 
            ex._world.target(9).p(), 
            max_iter, 
            ax
            )
        ex._world.plotTravelCostPerRegion(ax)


if __name__ == "__main__":
    unittest.main()
