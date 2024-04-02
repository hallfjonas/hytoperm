

## TOP X
1. unification of property and member names
2. documentation
3. warm starting the tree structure in HighLevelPlanner
    - one tree per target to capture how to reach that target quickest from anywhere in the mission space
4. improve code quality: 
    - automated getters and setters in python?
    - improve plot functions (increase flexibility)
    - ensure that all plot functions work coherently

## Coding Todos (Apr 2 2024)
1. unification of property and member names. Files that are done:
    - Agent
    - Tests
    - Experiment
    - World
    - GlobalPlanning
    - Dynamics
    - Optimization
    - Sensor
    - DataStructures
    - Plotters

2. organize files, classes, functions 
3. documentation

### future work
- optimize switching points/switching segments
- update global planner online (intertwined with optimization of switching segments)
- adequate termination criterion/improved bilevel optimization
    - backtracking
    - cost descent
    - KKT residual (nope, due to negative result)
- utilization of non-smooth OCP solvers on switching level
- relaxing vector field assumption || v || < || u|| 
    - come up with conditions for existence of solutions
- explore sensitivity wrt initialization of monitoring durations

## Nice to have
- warm starting for monitoring segments
    - maybe not too important as we are utilizing IPOPT
- remove the constraints in CPRegion that are not required (would make constraint activation much more efficient)
- 1.0 Investigate 'nlp_g failed Inf detected for output g'
    - check this https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
    - doesn't seem to affect optimization as of right now...
