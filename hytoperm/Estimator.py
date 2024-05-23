
from .World import *
from .Agent import *
from .Sensor import *
from .Experiment import *
from .RL import AgentRL

class Estimator:
    def __init__(self, agents : List[AgentRL], targets : List[Target]) -> None:
        self._agents : List[AgentRL] = agents
        self._targets : List[Target] = targets
        self._estimates : List[np.ndarray] = []
        self._covariances : List[np.ndarray] = []
        self.initialize()

    def initialize(self) -> None:
        for target in self._targets:
            nstates = target.getNumberOfStates()
            self._estimates.append(np.ones(nstates))
            self._covariances.append(np.eye(nstates))

    # update the estimate of target i
    def updateTarget(self, j : int, dt : float) -> None:

        target = self._targets[j]
        estimate_dot = target.A @ self._estimates[j] 
        Omega = self._covariances[j] 
        Omega_dot = Omega @ target.A.T + target.A @ Omega + target.Q

        for i, agent in enumerate(self._agents):
            sensor = agent.sensor()
            gammaij = float(sensor.getSensingQuality(target))
            Hij = sensor.getMeasurementMatrix(target)
            nuij = dt*sensor.drawNoise(target)
            phij = target.internalState()
            Rinvj = sensor.getMeasurementNoiseInverse(target)
            OHR = Omega @ Hij @ Rinvj

            # get the measurement
            zij = gammaij * Hij @ phij + nuij

            # update the estimate
            est = self._estimates[j]
            estimate_dot += OHR @ (zij - gammaij * Hij @ est)
            self._estimates[j] += dt * estimate_dot

            # update the covariance
            Omega_dot -= OHR @ Hij.T @ Omega
        
        self._covariances[j] += dt * Omega_dot

    def update(self, z : List[np.ndarray], dt : float) -> None:
        for j, target in enumerate(self._targets):
            self.updateTarget(j, dt)
