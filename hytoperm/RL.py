
import numpy as np

from .World import *
from .Experiment import *
from .Estimator import *


class State:

    def __init__(self, agents : List[AgentRL], targets : List[Target]) -> None:
        self._agents : List[AgentRL] = agents
        self._targets : List[Target] = targets
        self._estimator : Estimator = Estimator(agents, targets)

    def getCovarianceMatrix(self) -> np.ndarray:
        pass

    def plotToMissionSpace(self, ax : plt.Axes = None) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        for agent in self._agents:
            po.add(agent.plot(ax))
        

def transition(x : State, u, dt) -> State:
    for agent in x._agents:
        agent.update(u, dt)

    x._estimator.update(x, dt)

    for target in x._targets:
        target.update(dt)

def reward(x : State, u) -> float:
    rew = 0
    for j, target in enumerate(x._targets):
        err = x._estimator._estimates[j] - target.internalState()
        rew = rew - np.dot(err, err)
    
    assert(isinstance(rew, float))
    return rew
