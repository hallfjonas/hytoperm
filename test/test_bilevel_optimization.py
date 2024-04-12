
# external imports
import unittest

# internal imports
from hytoperm import *


class TesttestBilevelOptimization(unittest.TestCase):
    
    def testBilevelOptimization(self, n_sets=10):
        ex = Experiment.generate(n_sets=n_sets)
        assert(isinstance(ex, Experiment))
        ex.agent().computeVisitingSequence()
        ex.agent().op.alpha = 0.1
        ex.agent().op.beta = 0.9
        ex.agent().op.optimization_iters = 10
        ex.agent().optimizeCycle()


if __name__ == "__main__":
    unittest.main()
