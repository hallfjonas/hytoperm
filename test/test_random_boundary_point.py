
# external imports
import unittest

# internal imports
from hytoperm import *


class TestRandomBoundaryPoint(unittest.TestCase):
    
    def testRandomBoundaryPoint(self):
        n_sets = 10
        ex = Experiment()
        assert(isinstance(ex, Experiment))
        ex.addRandomVoronoiPoints(10)
        ex.addRandomTargets()
        ex.generatePartitioning()
        ex.plotWorld()
        p = ex._world.target(0).region().randomBoundaryPoint()
        plt.plot(p[0][0], p[0][1], 'gd')


if __name__ == "__main__":
    unittest.main()
