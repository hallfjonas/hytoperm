# Coding todos TOP 5
1. Optimize the local parameters based on gradient information
    - for now let's restrict our attention to monitoring time parameters
2. What kind of plots would be nice for paper?
3. Steady state computation would currently call the monitoring OCP method each time, but the trajectories shouldn't really change much...
    - generate mse trajectories given an agent trajectory
    - this should make simulation to steady state much more efficient!! (doesn't require solving OCPs in each cycle)

## Needs more testing

## Nice to have
- warm starting for monitoring segments
    - maybe not too important as we are utilizing IPOPT
- warm starting the tree structure in HighLevelPlanner
    - one tree per target to generate capture how to reach that target quickest...
