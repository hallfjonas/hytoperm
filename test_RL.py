
from hytoperm import *

if __name__ == "__main__":
    ex = Experiment.generate(n_sets=20)
    assert(isinstance(ex, Experiment))
    gpp = GlobalPathPlanner(ex._world)
    ga = GraphAbstraction(ex._world, gpp)
    
    nx.draw(ga.graph)
    plt.show()
    