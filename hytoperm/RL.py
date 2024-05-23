
import numpy as np

from .World import *
from .Experiment import *
from .Estimator import *


class State:

    def __init__(
            self,
            agents : List[AgentRL],
            targets : List[Target],
            estimator : Estimator
    ) -> None:
        self._agents : List[AgentRL] = agents
        self._targets : List[Target] = targets
        self._estimator : Estimator = estimator

    def getCovarianceMatrix(self) -> np.ndarray:
        pass

def transition(x : State, u, dt) -> State:
    for agent in x._agents:
        agent.update(x, u, dt)

    x._estimator.update(x, dt)

    for target in x._targets:
        target.update(dt)

def reward(self, x : State, u) -> float:
    cost = 0
    for j, target in enumerate(x._targets):
        err = x._estimator._estimates[j] - target.internalState()
        cost -= np.dot(err, err)
