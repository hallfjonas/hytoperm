
# external imports
import unittest
import random
import time

# internal imports
from hytoperm import *


class TestAgent(unittest.TestCase):
    
    def testDecomposition(self):
        n_sets = 20
        ex = SphericalExperiment.generate(n_sets=n_sets)
        assert(isinstance(ex, Experiment))
        
        ex.agent()
        ex.agent().computeVisitingSequence()
        ex.agent().initializeDecomposition(N = 100)
        dec = ex.agent().decomposition

        vec = dec.toVector()
        dec.fromVector(vec)
        vec2 = dec.toVector()

        for i in range(len(vec)):
            self.assertAlmostEqual(vec[i], vec2[i])

        
if __name__ == "__main__":
    unittest.main()
