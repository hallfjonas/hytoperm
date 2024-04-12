
# external imports
import unittest
import random

# internal imports
from hytoperm import *


class TestMultiAgent(unittest.TestCase):
    
    def testCycle(self):
        n_sets = 8
        ex = Experiment.generate(n_sets=n_sets, n_agents=3)
        assert(isinstance(ex, Experiment))
        ex.agent(0).computeVisitingSequence()
        ex.agent(1).setTargetVisitingSequence(
            [ex.world().target(i) for i in range(ex.world().nTargets())]
            )
        ex.agent(2).setTargetVisitingSequence(
            random.shuffle(
                [ex.world().target(i) for i in range(ex.world().nTargets())]
                )
            )
        
        for agent in ex.agents():
            agent.computeVisitingSequence()
            agent.initializeCycle()
            agent._cycle.simulate()
            
if __name__ == "__main__":
    unittest.main()
