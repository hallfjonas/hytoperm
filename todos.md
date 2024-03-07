# Coding todos TOP 5
1. Perform the tests below...
    - I want to see a periodic trajectory!!
2. Given a periodic trajectory, obtain steady state
    - Should just be a repitition of simulateCycle without updating parameters
    - Want to see a periodic MSE Trajectory Plot!
3. Optimize the local parameters based on gradient information
    - for now let's restrict our attention to monitoring time parameters
4. What kind of plots would be nice for paper?
5. Steady state computation would currently call the monitoring OCP method each time, but the trajectories shouldn't really change much...
    - generate mse trajectories given an agent trajectory
    - this should make simulation to steady state much more efficient!! (doesn't require solving OCPs in each cycle)

## Needs more testing
- Agent class:
    - computeVisitingSequence
    - getCycleTime
    - initializeCycle
    - simulateCycle
    - plotMSE
    - plotCycle

## Nice to have
- warm starting for monitoring segments
    - maybe not too important as we are utilizing IPOPT

