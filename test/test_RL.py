
# external imports
import unittest

# internal imports
from hytoperm import *


class Testtest_RL(unittest.TestCase):
    def testRL(self):
        n_sets = 8
        ex = Experiment.generate(n_sets=n_sets)
        assert(isinstance(ex, Experiment))
        
        ag = AgentRL(ex.world().target(0).p(), ex.world(), ex.agent().sensor())
        state = State([ag], ex.world().targets()) 
        rew = 0
        nSteps = 10
        for i in range(nSteps):
            u = np.random.rand(2)

            rew = rew + reward(state, u)
            transition(state, u, 0.01)
            print("{:3d} | {:9.2e} | ({:2.2e}, {:2.2e})".format(
                i, 
                rew, 
                ag.p()[0],
                ag.p()[1]
            ))

        return


if __name__ == "__main__":
    unittest.main()
