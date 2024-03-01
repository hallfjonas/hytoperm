import matplotlib.pyplot as plt
from pdfCropMargins import crop
import os

def export(name, directory = '../figures/experiments', ext='.pdf') -> None:
    out_dir = os.path.join(os.path.dirname(__file__), directory)
    file = os.path.join(out_dir, name + ext)
    plt.savefig(file)

    if ext == '.pdf':
        crop(["-p", "20", "-u", "-s", str(file), "-o", str(out_dir)])

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
