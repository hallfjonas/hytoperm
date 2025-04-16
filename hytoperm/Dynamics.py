
# external imports
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

# internal imports
from .PyPlotHelpers.Plotters import PlotObject, getAxes
from .DataStructures import Tree


class Trajectory:
    def __init__(self, x0 : np.ndarray, t0 : np.ndarray) -> None:
        self.assertDims(x0, t0)
        self.x = x0.copy()
        self.t = t0.copy()
        self._po : PlotObject = PlotObject()

    def extend(self, x : np.ndarray, t : np.ndarray) -> None:
        x, t = self.assertDims(x, t)
        self.x = np.append(self.x, x, 1)
        self.t = np.append(self.t, t)

    def appendTrajectory(self, trj : 'Trajectory') -> None:
        self.extend(trj.x, trj.t)

    def assertDims(self, x : np.ndarray, t : np.ndarray) -> None:
        if t.ndim != 1:
            raise ValueError("Time array must be one-dimensional")
        x_ret = x
        if x.ndim != 2:
            x_ret = x.reshape(-1,1)
        if x_ret.shape[1] != t.shape[0]:
            raise ValueError("State and time arrays must have the same length")
        return x_ret, t
    
    def getInitialValue(self) -> np.ndarray:
        return self.x[:,0]

    def getEndPoint(self) -> np.ndarray:
        return self.x[:,-1]

    def plot(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        for i in range(self.x.shape[0]):
            ln = ax.plot(self.t.flatten(), self.x[i,:].flatten(), **kwargs)
            self._po.add(PlotObject(ln))
        return self._po

    def plotStateVsTime(
            self, 
            idx, 
            ax : plt.Axes = None, 
            **kwargs
            ) -> PlotObject:
        ax = getAxes(ax)
        ln = ax.plot(self.t.flatten(), self.x[idx,:].flatten(), **kwargs)
        self._po.add(PlotObject(ln))
        return self._po
    
    def plotStateVsState(
            self, 
            idx1, 
            idx2, 
            ax : plt.Axes = None, 
            **kwargs
            ) -> PlotObject:
        ax = getAxes(ax)
        ln = ax.plot(self.x[idx1,:],self.x[idx2, :],**kwargs)
        self._po.add(PlotObject(ln))
        return self._po
    
    def shiftTime(self, delta : float) -> None:
        self.t = self.t + delta

    def getDuration(self) -> float:
        return self.t[-1] - self.t[0]

    def fromPath(path: Tree) -> Trajectory:
        """
        Constructs a trajectory from a tree of nodes.
        """
        x = []
        t = []
        T = path.getData().costToRoot()
        while path is not None:
            x.append(path.getData().p())
            t.append(T - path.getData().costToRoot())
            path = path.getParent()
        return Trajectory(np.array(x).T, np.array(t))

class ControlledTrajectory(Trajectory):
    def __init__(
            self, 
            x0 : np.ndarray, 
            t0 : np.ndarray, 
            u0 : np.ndarray
            ) -> None:
        super().__init__(x0, t0)
        self.u = u0.copy()

    def extend(
            self, 
            x : np.ndarray, 
            t : np.ndarray, 
            u : np.ndarray
            ) -> None:
        self.assertDims(x, t)
        self.x = np.append(self.x, x, 1)
        self.u = np.append(self.u, u, 1)
        self.t = np.append(self.t, t)
    