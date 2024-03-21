import os 
from Plotters import Exporter
from Optimization import OptimizationParameters

# Experiment generation
NAME = "large_comparison"
SEED=999
NSETS=20
NTARGETS=10
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

# adapt the optimization parameters
op = OptimizationParameters()
op._kkt_tolerance = 1e-1
op._sim_to_steady_state_tol = 1e-1
op._optimization_iters = 100
op._beta = 0.95