import os 
from Plotters import Exporter

# Experiment generation
NAME = "steady_vs_nonsteady"

# SEED SUCKSSS! (458 is ok!)
SEED=345
NSETS=12
NTARGETS=None
FRACTION= 0.75 if NTARGETS is None else NTARGETS/NSETS

# Experiment name and output directory
out_dir = os.path.join("/home/jonas/PhD/papers/CDC2024/figures/experiments", NAME)

# Plot settings
GENERATEPLOTS = False
WIDTH = 6.5

# A standard exporter
exporter = Exporter()
exporter.DIR = out_dir
exporter.WIDTH = WIDTH
