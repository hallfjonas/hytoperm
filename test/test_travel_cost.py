
# external imports
import unittest

# internal imports
from hytoperm import *


class TesttestTravelCost(unittest.TestCase):

    def testTravelCost(self):
        ex = Experiment.generate()
        fig, ax = ex.plotWorld()
        ex._world.plotTravelCostPerRegion(ax)


if __name__ == "__main__":
    unittest.main()
