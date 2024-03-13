# Coding todos TOP 5
1. Optimize the local parameters based on gradient information
    - for now let's restrict our attention to monitoring time parameters
    - I have gradients for local monitoring cost dJ/dtau for the monitored target
    - How do I get gradients for the increase of the not monitored targets?
        - analytical or via the agent._ucs function using AD?
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

# 03/12
- letss take a breaakkkk

## plan for evening 
- fix refine visiting sequence (extract the switching paths and extend vs)
- test all the improvements below
- initial tests for gradient stuff.....

## TEST THOSE IMPROVEMENTS 03/12
- replace monitoring params by switching params
    - let switching params hold the initial cov values of all targets
- move Dict[Target, Trajectory] mseTrajectory into trajectorysegment 
- simulate all mse trajectories for both switching and monitoring segments
- include the cost of other targets in the local monitoring cost (not when optimizing but when returning the cost, gradients, etc.)
- create class cycle that contains all cycle computation stuff. The agent will then maintain a (list of) cycle instance(s)
    - the cycle class is able to compute a gradient wrt the parameters
- do the gradient stuff