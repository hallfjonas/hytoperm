
# external imports
import os 

# internal imports
from src.Plotters import Exporter


# Experiment generation
NAME = "small"
SEED=235
NSETS=10
NTARGETS=None
FRACTION= 0.4 if NTARGETS is None else NTARGETS/NSETS

# Experiment name and output directory
out_dir = os.path.join("experiments", NAME)

# Plot settings
GENERATEPLOTS = False
WIDTH = 6.5

# A standard exporter
exporter = Exporter()
exporter.DIR = out_dir
exporter.WIDTH = WIDTH
