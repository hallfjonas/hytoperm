
# external imports
import unittest

# internal imports
from hytoperm import *


class TestRandomBoundaryPoint(unittest.TestCase):
    
    def testRandomBoundaryPoint(self):
        ex = Experiment()
        assert(isinstance(ex, Experiment))
        ex.addRandomVoronoiPoints(10)
        ex.generatePartitioning()
        ex.addRandomTargets()
        fig, ax = ex.plotWorld()
        region = ex._world.target(0).region()
        p = region.randomBoundaryPoint()
        Node(p, set([region])).plot(ax, color='yellow', marker='d')


if __name__ == "__main__":
    unittest.main()
