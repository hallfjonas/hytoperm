
# external imports
import unittest

# internal imports
from hytoperm import *


class TesttestDistToBoundary(unittest.TestCase):
    
    def testDistToBoundary(self):
        n_sets = 10
        ex = Experiment.generate(n_sets=n_sets)
        fig, ax = ex.plotWorld()
        ex._world.plotdistToBoundary(ax)


if __name__ == "__main__":
    unittest.main()
