
# external imports
import unittest
import numpy as np

# internal imports
from hytoperm import *


class TestOptimization(unittest.TestCase):
    
    def testBFGS(self):
        
        # minimize f(x,y) = x^2 + y^2
        #    s.t.  x = 0
        
        nx = 2
        ng = 1
        x0 = np.array([1.0, 1.0]).reshape(-1,1)

        xstar, ystar = BFGS(
            nx=nx,
            ng=ng,
            x0=x0,
            nabla_f=lambda x: np.array([2*x[0,0], 2*x[1,0]]).reshape(-1,1),
            nabla_g=lambda x: np.array([1, 0]).reshape(-1,1),
            g=lambda x: np.array([x[0]]).reshape(-1,1),
            lbg=np.array([0]),
            ubg=np.array([0])
        ).solve()

        self.assertAlmostEqual(xstar[0,0], 0.0)
        self.assertAlmostEqual(xstar[1,0], 0.0)
        
if __name__ == "__main__":
    unittest.main()
