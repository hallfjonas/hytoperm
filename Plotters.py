from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from pdfCropMargins import crop
import os
from dataclasses import dataclass

plt.rcParams['text.usetex'] = True

'''
Exporter: A class for export automation of matplotlib figures.

Created on: Mar 28 2024

@author: Jonas Hall       
'''
class Exporter:
    def __init__(self) -> None:
        self.WIDTH = 7
        self.HEIGHT = 7
        self.DPI = 300
        self.FONTSIZE = 14
        self.EXT = '.pdf'
        self.DIR = ''
        self.CROP = True

    def prepareDir(self, out_dir : str) -> None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    def export(self, name, fig = None) -> None:
        self.prepareDir(self.DIR)
        file = os.path.join(self.DIR, name + self.EXT)
        
        if fig is not None:
            fig.set_size_inches(self.WIDTH, self.HEIGHT)

        plt.rcParams['font.size'] = self.FONTSIZE

        plt.savefig(file, bbox_inches='tight', dpi=self.DPI)
        
        if self.CROP:
            self.cropMargins(file)

    def cropMargins(self, file) -> None:
        if self.EXT == '.pdf':
            crop(["-p", "20", "-u", "-s", str(file), "-o", str(self.DIR)])


'''
PlotAttribute: A class to store plot attributes.

Created on: Mar 28 2024

@author: Jonas Hall
'''
class PlotAttribute:
    def __init__(
            self, 
            c='black', 
            ls='-', 
            lw=2, 
            m=None, 
            ms=None, 
            a=1, 
            cm=None, 
            aa=None
        ) -> None:
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


'''
PlotAttributes: Fixed plot attributes for different PM objects

Created on: Mar 28 2024

@author: Jonas Hall
'''
@dataclass
class PlotAttributes:
    agent = PlotAttribute(c='navy')
    target = PlotAttribute(c='red', m='o', ms=10)
    partition = PlotAttribute(c='black')
    vector_field = PlotAttribute(c='black', a=0.3)
    cmap = 'viridis'
    sensor_quality = PlotAttribute(aa=True, a=0.4, cm=cmap)
    partition_background = PlotAttribute(
        c=get_cmap(sensor_quality.cmap)(0.001), 
        a=sensor_quality.alpha
    )
    target_colors = [
        '#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', 
        '#984ea3', '#999999', '#e41a1c', '#dede00'
    ]
    
    phi = PlotAttribute(c='forestgreen', m='X', ms=7.5, ls='')
    psi = PlotAttribute(c='yellow', m='p', ms=10, ls='')

    u1_switch = PlotAttribute(c='salmon', ls='-')
    u2_switch = PlotAttribute(c='mediumaquamarine', ls='--')
    u_norm_switch = PlotAttribute('black', ls='-', lw=3)
    u1_monitor = PlotAttribute(c=u1_switch.color, ls=u1_switch.linestyle)
    u2_monitor = PlotAttribute(c=u2_switch.color, ls=u2_switch.linestyle)
    u_norm_monitor = PlotAttribute(
        c=u_norm_switch.color, 
        ls=u_norm_switch.linestyle, 
        a=u_norm_switch.alpha, 
        lw=u_norm_switch.linewidth
    )


'''
extendKeywordArgs: Extend a dictionary with keyword arguments.

Created on: Mar 28 2024

@author: Jonas Hall

@details: Returns a new dictionary containing all keyword arguments in kwargs
          together with all non-None key-value pairs in p that don't exist in 
          kwargs.
'''
def extendKeywordArgs(p : dict, **kwargs) -> dict:
    eka = kwargs.copy()
    for key in p.keys():
        if not key in eka.keys() and p[key] is not None:
            eka[key] = p[key]
    return eka

'''
PlotObject: A container for matplotblib objects.

Created on: Mar 28 2024

@author: Jonas Hall

@details: This container class can be used to store matplotlib objects to simplify their removal later on.
'''
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
