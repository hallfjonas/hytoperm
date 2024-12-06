
from .World import *
from .Dynamics import *
from .Sensor import *
from .Optimization import *

'''
omegaDot: the ODE right hand side of the state estimator's covariance matrix
'''
def omegaDot(p, Omega, target : Target, sensor : Sensor, inTargetRegion=False):
    A = target.A
    Q = target.Q
    H = sensor.getMeasurementMatrix(target)
    R_inv = sensor.getMeasurementNoiseInverse(target)
    unmonitored = Q + A @ Omega + Omega @ A.T
    if inTargetRegion:
        mf = sensor.getQualityFunction(target=target)(p, target.p())
        return unmonitored  - mf*mf*Omega @ H.T @ R_inv @ H @ Omega
    return unmonitored


'''
unmonitoredOmegaSimulator: build a parameterized casadi function that can be 
    utilized to simulate the estimator's covariance matrix while the target
    is not being monitored.
'''
def unmonitoredOmegaSimulator(
        target : Target, 
        sensor : Sensor,
        N : int
        ) -> cad.Function:
    # states
    no = target.getNumberOfStates()
    Omega = cad.SX.sym('Omega', no*no)
    nx = Omega.shape[0]
    
    # parameters
    tf = cad.SX.sym('tf', 1)
    Omega0 = cad.SX.sym('Omega0', no*no)
    params = cad.vertcat(tf, Omega0)

    # Model equations
    oDot = omegaDot(None, Omega, target, sensor, False)

    # Objective term
    L = 0
    for i in range(no):
        L += Omega[i*no + i]

    # Fixed step Runge-Kutta 4 integrator
    M = 4 # RK4 steps per interval
    DT = tf/N/M
    f = cad.Function('f', [Omega, params], [oDot, L])
    X0 = cad.SX.sym('X0', no*no)
    X = X0
    Q = 0
    for j in range(M):
        k1, k1_q = f(X, params)
        k2, k2_q = f(X + DT/2 * k1, params)
        k3, k3_q = f(X + DT/2 * k2, params)
        k4, k4_q = f(X + DT * k3, params)
        X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
        Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
    F = cad.Function('F', [X0, params], [X, Q], ['x0','p'], ['xf','qf'])

    # Start with an empty NLP
    OmegaTrajectory=[]
    J = 0
    mse = [sum([Omega0[i*no + i] for i in range(no)])]
    
    # "Lift" initial conditions
    Xk = Omega0
    OmegaTrajectory += [Omega0]
    
    # Formulate the integrator
    for k in range(N):

        # Integrate till the end of the interval
        Fk = F(x0=Xk, p=params)
        Xk = Fk['xf']
        OmegaTrajectory += [Xk]
        J=J+Fk['qf']

        # Add Xk to Omega vec
        mse += [sum([Xk[i*no + i] for i in range(no)])]

    omegaSim = cad.Function(
        'OmegaSim', 
        [params], 
        [J, cad.vertcat(*mse), cad.vertcat(*OmegaTrajectory)], 
        ['p'], 
        ['Ik', 'mse', 'OmegaTrajectory']
    )

    return omegaSim


'''
simulateUnmonitoredOmega: simulate the estimator's covariance matrix while the
    target is not being monitored. 

@param fun: the casadi function that simulates the estimator's covariance matrix
    (see unmonitoredOmegaSimulator)
'''
def simulateUnmonitoredOmega(fun : cad.Function, tf, Omega0):
    
    # set params and evaluate
    params = cad.vertcat(tf, np.matrix(Omega0).getA1())
    sim = fun(params)
    
    # get dimensions
    no = int(np.sqrt(len(Omega0)))
    N = int(len(sim[1].full().flatten())/no) - 1

    # generate trajectories
    t_grid = np.array([tf/N*k for k in range(N+1)]).flatten()
    Ik = sim[0].full().flatten()[0]
    mseTrajectory = Trajectory(cadToNumpy(sim[1], no, N+1), t_grid)
    omegaTrajectory = Trajectory(cadToNumpy(sim[2], no*no, N+1), t_grid)
    
    return mseTrajectory, omegaTrajectory, Ik, mseTrajectory.getEndPoint()
