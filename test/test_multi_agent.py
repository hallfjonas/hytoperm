
# external imports
import unittest

# internal imports
from hytoperm import *


class TestMultiAgent(unittest.TestCase):
    
    def testCycle(self):
        n_sets = 8
        ex = Experiment.generate(n_sets=n_sets, n_agents=3)
        assert(isinstance(ex, Experiment))
        ex.agent().computeVisitingSequence()
        ex.agent().initializeCycle()
        ex.agent()._cycle.simulate()


if __name__ == "__main__":
    unittest.main()