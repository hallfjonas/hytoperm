
# external imports
import unittest

# internal imports
from hytoperm import *


class TestGlobalPlanner(unittest.TestCase):
    
    def testRRBT(self):
        niter = 1000; n_sets = 20
        ex = VoronoiExperiment.generate(n_sets=n_sets)
        assert(isinstance(ex, Experiment))
        gpp: RRBTGlobalPlanner = ex.agent().gpp()
        gpp.rrbt_iter = niter
        fig, ax = ex.plotWorld()

        t0 = ex._world.targets()[0].p()
        tf = ex._world.targets()[-1].p()
        path, time = gpp.planPath(t0, tf)

        self.assertTrue(isinstance(path, Tree))
        self.assertTrue(isinstance(time, float))
        self.assertTrue(np.allclose(t0,path.getData().p()))
        self.assertTrue(np.allclose(tf,path.getRoot().getData().p()))
        
        ex._world.plotTravelCostPerRegion(ax)
        
    def testTSP(self):
        n_targets=10; plot = False
        ex = SphericalExperiment.generate(n_targets=n_targets)
        assert(isinstance(ex, Experiment))
        gpp = NormBasedGlobalPlanner(ex._world)
        gpp._plot_options.toggleAllPlotting(plot)
        fig, ax = ex.plotWorld()
        gpp.solveTSP()
        gpp.plotTSPSolution(ax, color='red', linewidth=2)
        po = ex.agent().plotSensorQuality(grid_size=0.05, ax=ax)
        gpp.plotTSPSolution(ax, color='red', linewidth=2)


if __name__ == "__main__":
    unittest.main()
