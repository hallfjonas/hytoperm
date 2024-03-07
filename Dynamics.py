
import numpy as np
from Plotters import PlotObject
import matplotlib.pyplot as plt

class Trajectory:
    def __init__(self, x0, t0) -> None:
        self.x = np.array(x0)
        self.t = np.array(t0)
        self._po : PlotObject = PlotObject()

    def extend(self, x, t):
        self.x = np.append(self.x, x, 1)
        self.t = np.append(self.t, t, 1)

    def getInitialValue(self):
        return self.x[:,0]

    def getEndPoint(self):
        return self.x[:,-1]

    def plot(self, ax : plt.Axes, **kwargs):
        self._po.remove()
        self._po.add(ax.plot(self.t, self.x, **kwargs))
        return self._po
    
    def plotStateVsTime(self, idx, ax : plt.Axes, **kwargs):
        self._po.remove()
        self._po.add(ax.plot(self.t, self.x[idx,:], **kwargs))
        return self._po
    
    def plotStateVsState(self, idx1, idx2, ax : plt.Axes, **kwargs):
        self._po.remove()
        self._po.add(ax.plot(self.x[idx1, :], self.x[idx2, :], **kwargs))
        return self._po