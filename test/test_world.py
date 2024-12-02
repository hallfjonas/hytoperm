
# external imports
import unittest

# internal imports
from hytoperm import *

pass_settings = {
    "n_sets": [1, 5, 10, 100],
    "fraction": [0, 0.5, 1],
    "seed": [None, 234],
    "min_dist": [0.0]
}

fail_settings = {
    "n_sets": [ 10],
    "fraction": [-1, 1],
    "min_dist": [2]
}


class TestWorld(unittest.TestCase):
    
    def testTravelCost(self):
        ex = VoronoiExperiment.generate(n_targets=5, fraction=0.5)
        fig, ax = ex.plotWorld()
        ex._world.plotTravelCostPerRegion(ax)

    def testDistToBoundary(self):
        n_sets = 10
        ex = VoronoiExperiment.generate(n_sets=n_sets, fraction=0.5)
        fig, ax = ex.plotWorld()
        ex._world.plotdistToBoundary(ax)

    def testRandomBoundaryPoint(self):
        ex = VoronoiExperiment()
        assert(isinstance(ex, Experiment))
        ex.addRandomVoronoiPoints(10)
        ex.generatePartitioning()
        ex.addRandomTargets()
        fig, ax = ex.plotWorld()
        region = ex._world.target(0).region()
        p = region.randomBoundaryPoint()
        Node(p, set([region])).plot(ax, color='yellow', marker='d')

    def testRandomPoint(self):
        ex = VoronoiExperiment()
        assert(isinstance(ex, Experiment))
        ex.addRandomVoronoiPoints(10)
        ex.generatePartitioning(n_obstacles=1)
        
        plt.ion()
        fig, ax = ex.plotWorld()
        plt.show()
        for region in ex.world().regions():
            p = region.randomPoint()
            po = region.fill()
            pn = Node(p, set([region])).plot(ax, color='yellow', marker='d')
            self.assertTrue(region.contains(p))
            po.remove()
           
            regs = ex.world().getRegions(p)
            reg = ex.world().getRegion(p)

            po_region = region.plot(color='red')
            po_reg = reg.plot(color='blue', linestyle='--')
            po_regs = PlotObject()
            for r in regs:
                po_regs.add(r.plot(color='green', linestyle='-.'))

            self.assertTrue(reg in regs)
            self.assertTrue(region in regs)

            po_region.remove()
            po_reg.remove()
            po_regs.remove()
            pn.remove()
        plt.ioff()
        plt.close()

    def testWorldGenerationPass(self):
        for n_sets in pass_settings["n_sets"]:
            for fraction in pass_settings["fraction"]:
                for seed in pass_settings["seed"]:
                    for min_dist in pass_settings["min_dist"]:
                        ex = VoronoiExperiment.generate(
                            n_sets=n_sets, 
                            fraction=fraction, 
                            seed=seed, 
                            min_dist=min_dist
                        )
                        self.assertIsInstance(ex, Experiment)

    def testWorldGenerationFail(self):
        for n_sets in fail_settings["n_sets"]:
            for fraction in fail_settings["fraction"]:
                for min_dist in fail_settings["min_dist"]:
                    try:
                        VoronoiExperiment.generate(
                            n_sets=n_sets, 
                            fraction=fraction, 
                            seed=None, 
                            min_dist=min_dist
                        )
                        self.fail("Expected ValueError.")
                    except RuntimeError:
                        pass

    def testPlotWorld(self):
        n_sets = 10
        ex = SphericalExperiment.generate(n_sets=n_sets, fraction=0.5)
        fig, ax = ex.plotWorld()
        ex.world().plotDomain(ax)   


if __name__ == "__main__":
    unittest.main()
