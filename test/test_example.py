
# external imports
import unittest
import matplotlib.pyplot as plt

# internal imports
from hytoperm import *


class TestExample(unittest.TestCase):

    def testVoronoiExample(self):
        seed = np.random.randint(0, 10000)
        ex = VoronoiExperiment.generate(
            seed=seed,
            n_sets=10, 
            fraction=0.5
            )
        fig, ax = ex.plotWorld()
        ex.agent().plotSensorQuality()
        ex.agent().computeVisitingSequence()
        op = OptimizationParameters()
        op.optimization_iters = 3
        
        print(f"Selected seed: {seed}.")

        ex.agent().optimizeCycle(op=op)
        ex.agent().plotCycle()


    def testSphericalExample(self):
        ex = SphericalExperiment.generate(n_targets=5)
        fig, ax = ex.plotWorld()
        ex.agent().plotSensorQuality()
        ex.agent().computeVisitingSequence()
        op = OptimizationParameters()
        op.optimization_iters = 3
        ex.agent().optimizeCycle(op)
        ex.agent().plotCycle()


if __name__ == "__main__":
    unittest.main()
