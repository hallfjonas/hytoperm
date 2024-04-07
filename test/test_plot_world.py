
# external imports
import unittest

# internal imports
from hytoperm import *


class TestPlotWorld(unittest.TestCase):
    
    def testPlotWorld(self):
        n_sets = 10
        ex = Experiment.generate(n_sets=n_sets)
        fig, ax = ex.plotWorld()


if __name__ == "__main__":
    unittest.main()
