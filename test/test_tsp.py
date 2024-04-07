
# external imports
import unittest

# internal imports
from hytoperm import *


class TesttestTSP(unittest.TestCase):
    
    def testTSP(self, n_sets=20, plot = False):
        ex = Experiment.generate(n_sets=n_sets, fraction=0.2)
        assert(isinstance(ex, Experiment))
        gpp = GlobalPathPlanner(ex._world)
        gpp._plot_options.toggleAllPlotting(plot)
        fig, ax = ex.plotWorld()
        gpp.solveTSP()
        gpp.plotTSPSolution(ax, color='red', linewidth=2)
        po = ex._agent.plotSensorQuality(grid_size=0.05, ax=ax)
        gpp.plotTSPSolution(ax, color='red', linewidth=2)


if __name__ == "__main__":
    unittest.main()
