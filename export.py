
from hytoperm.PyPlotHelpers.Plotters import Exporter
import matplotlib.pyplot as plt

# set tex options
plt.rc('text', usetex=True)

exporter = Exporter()
exporter.DIR = "/home/jonas/PhD/papers/BTO/figures/"
exporter.EXT = ".png"
exporter.DPI = 300
