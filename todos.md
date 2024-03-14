# Coding todos TOP 5
1. Optimize the local parameters based on gradient information
    - Let's test this!!
2. What kind of plots would be nice for paper?

## Needs more testing
- optimization routine

## Nice to have
- warm starting for monitoring segments
    - maybe not too important as we are utilizing IPOPT
- warm starting the tree structure in HighLevelPlanner
    - one tree per target to generate capture how to reach that target quickest...
- remove the constraints in CPRegion that are not required (would make constraint activation much more efficient)