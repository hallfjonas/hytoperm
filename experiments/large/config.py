
# external imports
import os 

# internal imports
from src.Plotters import Exporter


# Experiment generation
NAME = "large"
SEED=11
NSETS=20
NTARGETS=2
FRACTION= 0.75 if NTARGETS is None else NTARGETS/NSETS

# Experiment name and output directory
out_dir = os.path.join("experiments", NAME)

# Plot settings
GENERATEPLOTS = False
WIDTH = 6.5

# A standard exporter
exporter = Exporter()
exporter.DIR = out_dir
exporter.WIDTH = WIDTH
