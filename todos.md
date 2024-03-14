# Coding todos TOP 5
1. What kind of plots would be nice for paper?
    - control labels (monitoring switch, u1, u2)
    - ...?
2. 

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