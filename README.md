# Hybrid Trajectory Optimizer for Persistent Monitoring
This repository contains a collection of tools that can be used for the design of trajectories in Persistent Monitoring (PM) settings with partitioned mission spaces. The partitioning is based on the agent's dynamics, which are allowed to vary throughout the environment. This leads to a hybrid dynamical system and introduces an additional layer of complexity towards the planning portion of the problem: we must not only identify in which order to visit the points of interest, but also in which order to traverse the regions. 
We split the problem into an
1. offline high-level sequence planner; and
2. online trajectory optimizer realizing the computed visiting sequence.

## Offline High-Level Sequence Planner

## Online Trajectory Optimizer

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
    