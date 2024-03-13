
import numpy as np
from Plotters import PlotObject
import matplotlib.pyplot as plt

class Trajectory:
    def __init__(self, x0 : np.ndarray, t0 : np.ndarray) -> None:
        self.assert_dims(x0, t0)
        self.x = x0.copy()
        self.t = t0.copy()
        self._po : PlotObject = PlotObject()

    def extend(self, x : np.ndarray, t : np.ndarray) -> None:
        self.assert_dims(x, t)
        self.x = np.append(self.x, x, 1)
        self.t = np.append(self.t, t)

    def appendTrajectory(self, trj : 'Trajectory') -> None:
        self.extend(trj.x, trj.t)

    def assert_dims(self, x : np.ndarray, t : np.ndarray) -> None:
        assert(x.ndim == 2); assert(t.ndim == 1)
        assert(x.shape[1] == t.shape[0])

    def getInitialValue(self):
        return self.x[:,0]

    def getEndPoint(self):
        return self.x[:,-1]

    def plot(self, ax : plt.Axes = plt, **kwargs) -> PlotObject:
        for i in range(self.x.shape[0]):
            self._po.add(ax.plot(self.t.flatten(), self.x[i,:].flatten(), **kwargs))
        return self._po

    def plotStateVsTime(self, idx, ax : plt.Axes = plt, **kwargs) -> PlotObject:
        self._po.add(ax.plot(self.t.flatten(), self.x[idx,:].flatten(), **kwargs))
        return self._po
    
    def plotStateVsState(self, idx1, idx2, ax : plt.Axes = plt, **kwargs) -> PlotObject:
        self._po.add(ax.plot(self.x[idx1, :].flatten(), self.x[idx2, :].flatten(), **kwargs))
        return self._po
    
    def shiftTime(self, delta : float) -> None:
        self.t = self.t + delta

    def getDuration(self) -> float:
        return self.t[-1] - self.t[0]

class ControlledTrajectory(Trajectory):
    def __init__(self, x0 : np.ndarray, t0 : np.ndarray, u0 : np.ndarray) -> None:
        super().__init__(x0, t0)
        self.u = u0.copy()

    def extend(self, x : np.ndarray, t : np.ndarray, u : np.ndarray) -> None:
        self.assert_dims(x, t)
        self.x = np.append(self.x, x, 1)
        self.u = np.append(self.u, u, 1)
        self.t = np.append(self.t, t)

    