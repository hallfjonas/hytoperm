from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from pdfCropMargins import crop
import os
from dataclasses import dataclass

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 13

class Exporter:
    def __init__(self) -> None:
        self.WIDTH = 7
        self.HEIGHT = 7
        self.DPI = 300
        self.FONTSIZE = 14
        self.EXT = '.pdf'
        self.DIR = '/home/jonas/PhD/papers/CDC2024/figures/experiments'

    def prep_dir(self, out_dir : str) -> None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    def export(self, name, fig = None) -> None:
        self.prep_dir(self.DIR)
        file = os.path.join(self.DIR, name + self.EXT)
        
        if fig is not None:
            fig.set_size_inches(self.WIDTH, self.HEIGHT)

        plt.savefig(file, bbox_inches='tight', dpi=self.DPI)
        self.crop_margins(file)

    def crop_margins(self, file) -> None:
        if self.EXT == '.pdf':
            crop(["-p", "20", "-u", "-s", str(file), "-o", str(self.DIR)])

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
    phi = PlotAttribute(c='forestgreen', m='X', ms=7.5, ls='')
    psi = PlotAttribute(c='yellow', m='p', ms=10, ls='')
    u1_switch = PlotAttribute(c='salmon', ls='-')
    u2_switch = PlotAttribute(c='mediumaquamarine', ls='--')
    u_norm_switch = PlotAttribute('black', ls='-', lw=3)
    u1_monitor = PlotAttribute(c=u1_switch.color, ls=u1_switch.linestyle)
    u2_monitor = PlotAttribute(c=u2_switch.color, ls=u2_switch.linestyle)
    u_norm_monitor = PlotAttribute(c=u_norm_switch.color, ls=u_norm_switch.linestyle, a=u_norm_switch.alpha, lw=u_norm_switch.linewidth)
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
