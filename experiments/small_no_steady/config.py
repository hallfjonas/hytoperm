import os 
from Plotters import Exporter

# Experiment generation
NAME = "small_no_steady"
SEED=235
NSETS=10
NTARGETS=None
FRACTION= 0.4 if NTARGETS is None else NTARGETS/NSETS

# Experiment name and output directory
out_dir = os.path.join("/home/jonas/PhD/papers/CDC2024/figures/experiments", NAME)

# Plot settings
GENERATEPLOTS = False
WIDTH = 6.5

# A standard exporter
exporter = Exporter()
exporter.DIR = out_dir
exporter.WIDTH = WIDTH
