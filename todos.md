# Coding todos TOP 5
1. From a TSP solution, initialize all parameters:
    - local monitoring times
    - switching points on the target region boundaries
    - uncertainty at beginning of local segment
2. Build simulation basis for computing a cycle by concatenating all local monitoring and switching segments using the local and global controllers, i.e., write a function
    - arguments: all local parameters
    - returns: 
        - control of agent
        - trajectories of
            - agent 
            - all mean squared cov errors
3. Given a periodic trajectory, obtain steady state
4. Update the local parameters based on gradient information

## Needs more testing
- Build simulation basis for covariance mean squared error, i.e., write a casadi function:
    - arguments: local parameters
    - returns: time grid with N equidistant points with integrated mean squared error values
- Given local parameters, find optimal local monitoring path
