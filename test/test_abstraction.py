
# external imports
import unittest

# internal imports
from hytoperm import *


class TestAbstraction(unittest.TestCase):
    def testCompleteGraph(self):
        ex = VoronoiExperiment.generate(n_sets=20)
        assert(isinstance(ex, Experiment))
        gpp = ex.agent().gpp()
        opts = AbstractionOptions()
        opts.onlyDirectConnections = False
        ga = GraphAbstraction(ex._world, gpp, opts)
        ga.abstract(ex.world(), gpp)
        
        fig, ax = plt.subplots()
        ga.plotAbstraction(ax=ax)
        plt.close()

    def testIncompleteGraph(self):
        ex = VoronoiExperiment.generate(n_sets=20)
        assert(isinstance(ex, Experiment))
        gpp = ex.agent().gpp()
        opts = AbstractionOptions()
        opts.onlyDirectConnections = True
        ga = GraphAbstraction(ex.world(), opts=opts)
        ga.abstract(ex._world, gpp)
        
        fig, ax = plt.subplots()
        ga.plotAbstraction(ax=ax)
        plt.close()


if __name__ == "__main__":
    unittest.main()
