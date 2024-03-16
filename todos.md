
# Paper CDC 2024!! Let's goooo

## TOP X
1. DONE BY EOD March 16
1.1 Write decomposition section
1.2 Write global cost optimization section
1.3 Include small results
1.4 Investigate 'nlp_g failed Inf detected for output g'
    - check this https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
2. Write global planning section (EOD Match 17)
3. Write results section 
4. Write Future work section

## Coding Todos
1. Clean up controls plot
    - add labels (monitoring switch, u1, u2, norm)
2. TSP, initial cycle, optimal cycle plot
    - all in one plot or multiple?

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