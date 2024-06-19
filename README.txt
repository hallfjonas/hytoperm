# Hybrid Trajectory Optimizer for Persistent Monitoring
This repository contains a collection of tools that can be used for the design of trajectories in Persistent Monitoring (PM) settings with partitioned mission spaces. The partitioning is assumed to be based on the agent's dynamics, which are allowed to vary throughout the environment. This repository contains all 
code used for [this preprint](https://arxiv.org/abs/2403.19769).
## Installation
1. Setup & activate virtual environment, then install.
```
$ virtualenv env --python=python3
$ source env/bin/activate
$ pip install -e .
```

## Usage Example
Utilizing the random experiment generator, we can run a simple test experiment
```
import matplotlib.pyplot as plt
from hytoperm import *

# generate and plot the expriment setup
ex = Experiment.generate()
fig, ax = ex.plotWorld()
ex.agent().plotSensorQuality()

# let the agent compute the visiting sequence and optimize the cycle
ex.agent().computeVisitingSequence()
ex.agent().optimizeCycle()

# plot the optimal cycle
ex.agent().plotCycle()
plt.show()
```

## Coding Conventions
- All classes use the `CamelCase` convention
- All methods use the `camelCase` convention
    - exceptions are unit test methods, which use `test_<test_name>` convention
- Properties and methods that have the prefix `_` are considered private
- All setters and getters are defined in the traditional sense (not the Python properties way)
- All plot functions are expected to work as follows:
    - they return a `PlotObject` containing all plot objects that were added by the function
    - they take at least the arguments ax and **kwargs. ax is assumed to be either None or a matplotlib Axes. If ax is None then the matplotlib function `gca` is invoked to retrieve the current active Axes.
- Each class definition is seperated by two lines
- Respect the 80 column limit in all code files. Exceptions:
    - class property comments, which start at the 80 column mark
    - error and warning messages
    