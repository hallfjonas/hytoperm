from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from pdfCropMargins import crop
import os
from dataclasses import dataclass

def export(name, directory = '.', ext='.pdf') -> None:
    out_dir = os.path.join(os.path.dirname(__file__), directory)
    file = os.path.join(out_dir, name + ext)
    plt.savefig(file)

    if ext == '.pdf':
        crop(["-p", "20", "-u", "-s", str(file), "-o", str(out_dir)])

class PlotAttribute:
    def __init__(self, c='black', ls='-', lw=2, m=None, ms=None, a=1, cm=None, aa=None) -> None:
        self.color = c
        self.linestyle = ls
        self.linewidth = lw
        self.marker = m
        self.markersize = ms
        self.alpha = a
        self.cmap = cm
        self.antialiased = aa

    def getAttributes(self) -> dict:
        return {
            'color': self.color,
            'linestyle': self.linestyle,
            'linewidth': self.linewidth,
            'marker': self.marker,
            'markersize': self.markersize,
            'alpha': self.alpha,
            'cmap': self.cmap,
            'antialiased': self.antialiased,
        }

# Fixed plot attributes for different objects
@dataclass
class PlotAttributes:
    agent = PlotAttribute(c='navy')
    target = PlotAttribute(c='red', m='o', ms=10)
    partition = PlotAttribute(c='black')
    phi = PlotAttribute(c='green', m='d', ms=10)
    psi = PlotAttribute(c='yellow', m='d', ms=10)
    u1_switch = PlotAttribute(c='blue', ls='-')
    u2_switch = PlotAttribute(c=u1_switch.color, ls='--')
    u_norm_switch = PlotAttribute(u1_switch.color, ls='-', a=0.3)
    u1_monitor = PlotAttribute(c='red', ls='-')
    u2_monitor = PlotAttribute(c=u1_monitor.color, ls=u2_switch.linestyle)
    u_norm_monitor = PlotAttribute(c=u1_monitor.color, ls='-', a=0.3)
    vector_field = PlotAttribute(c='black', a=0.3)
    cmap = 'viridis'
    sensor_quality = PlotAttribute(aa=True, a=0.4, cm=cmap)
    partition_background = PlotAttribute(c=get_cmap(sensor_quality.cmap)(0.001), a=sensor_quality.alpha)
    target_colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

def extend_keyword_args(p : dict, **kwargs) -> dict:
    eka = kwargs.copy()
    for key in p.keys():
        if not key in eka.keys() and p[key] is not None:
            eka[key] = p[key]
    return eka

class PlotObject:
    def __init__(self, *args) -> None:
        self._objs = []
        self.add(*args)

    def add(self, *args) -> None:
        for obj in args:
            if isinstance(obj, PlotObject):
                self.add(obj._objs)
            else:
                try:
                    for o in obj:
                        self.add(o)
                except:
                    self._objs.append(obj)

    def remove(self) -> None:
        for obj in self._objs:
            obj.remove()
        self._objs.clear()
