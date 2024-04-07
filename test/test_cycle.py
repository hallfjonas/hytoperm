
# external imports
import unittest

# internal imports
from hytoperm import *


class TesttestCycle(unittest.TestCase):
    
    def testCycle(self, n_sets=8):
        ex = Experiment.generate(n_sets=n_sets)
        assert(isinstance(ex, Experiment))
        ex._agent.computeVisitingSequence()
        ex._agent.initializeCycle()
        ex._agent._cycle.simulate()


if __name__ == "__main__":
    unittest.main()