
# Paper CDC 2024!! Let's goooo

## TOP X
1. DONE BY EOD March 19
1.0 Investigate 'nlp_g failed Inf detected for output g'
    - check this https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
1.1 Change large seed 785 to something else and create new results for large space!!
1.2 Write results section 
1.3 Negative result?
1.4 Write Future work section
1.5 Make comparison: 1 steady state iter (i.e., change every cycle ) vs 100

## Coding Todos
1.0 Investigate 'nlp_g failed Inf detected for output g'
    - check this https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F

## writing specifications
### contributions
- vector fields (in problem formulation)
- sensing regions (in view of target fading effects or similar) (in problem formulation)
- optimization scheme (own section)

### global planning:

### decomposition into local control problems:

### optimizing the global cost (bilevel optimization)

### experiments:
- illustrative small:
    - negative result: put in small failed and discuss IPOPT and u bound
    - controls plot cleaned up
- demonstrative medium sized:
    - with sinusoidal sensing (if it is reliable enough)
    - find good balance between Q and sensing quality
    - three plots: TSP Sol, Initial Cycle, Optimal Cycle
    - steady state MSE plot

### future work
- limitation: global optimization suboptimality due to missing optimization of switching points
- adequate termination criterion/improved bilevel optimization
    - backtracking
    - cost descent
    - KKT residual (nope, due to negative result)
- utilization of non-smooth OCP solvers on switching level
- relaxing vector field assumption || v || < || u|| 
    - come up with conditions for existence of solutions

## needs more testing

## Nice to have
- improve optimization convergence
    - the numerical gradients near an active constraint (tau >= min_tau) is off. Why?
    - slowly allow the constraints to become active
    - can I get a dual variable somehow?
    - does it make sense to implement an interior point method?
- warm starting for monitoring segments
    - maybe not too important as we are utilizing IPOPT
- warm starting the tree structure in HighLevelPlanner
    - one tree per target to generate capture how to reach that target quickest...
- remove the constraints in CPRegion that are not required (would make constraint activation much more efficient)