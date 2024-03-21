      
from World import *
from Dynamics import *
from HighLevel import *
from Optimization import *

class SensingQualityFunction:
    def __init__(self):
        pass

    @abstractclassmethod
    def __call__(self, p, q):
        pass

class ConstantSensingQualityFunction(SensingQualityFunction):
    def __init__(self, c = 1):
        self._c : float = None
        self.assign_constant(c)

    def assign_constant(self, c : float) -> None:
        assert(c >= 0)
        assert(c <= 1)
        self._c = c

    def __call__(self, p, q):
        return self._c
    
class GaussianSensingQualityFunction(SensingQualityFunction):
    def __init__(self, c = 50):
        self._c : float = None
        self.assign_constant(c)

    def assign_constant(self, c : float) -> None:
        assert(c > 0)
        self._c = c

    def __call__(self, p, q):
        delta = p - q
        sqr_dist = cad.dot(delta, delta)
        return cad.exp(-self._c*sqr_dist)

class SinusoidalSensingQualityFunction(SensingQualityFunction):
    def __init__(self, c1 = 3.0, c2 = 20.0, c3 = 40.0):
        self._c1 : float = None
        self._c2 : float = None
        self._c3 : float = None
        self.assign_constants(c1, c2, c3)

    def assign_constants(self, c1 : float, c2 : float, c3 : float) -> None:
        assert(c1 > 0)
        assert(c2 > 0)
        assert(c3 > 0)
        self._c1 = c1
        self._c2 = c2
        self._c3 = c3

    def __call__(self, p, q):
        delta = p - q
        return cad.exp(-self._c3*cad.dot(delta,delta))*0.5*(cad.sin(self._c1*delta[0])**2 + cad.cos(self._c2*delta[1])**2)

class Sensor:
    def __init__(self, p : np.ndarray = None) -> None:
        self._p : np.ndarray = None
        self._ttsqm : Dict[Target, SensingQualityFunction] = {}     # target to sensing quality function mapper
        self._ttHm : Dict[Target, np.ndarray] = {}                  # target to measurement matrix mapper
        self._ttRm : Dict[Target, np.ndarray] = {}                  # target to measurement noise mapper
        self._ttRinvm : Dict[Target, np.ndarray] = {}               # target to measurement noise inverse mapper
        
        if p is not None:
            self.setPosition(p)

    def getPosition(self) -> np.ndarray:
        return self._p
    
    def setPosition(self, p : np.ndarray) -> None:
        assert(isinstance(p, np.ndarray))
        self._p = p

    def targetToSQFMapper(self) -> Dict[Target, SensingQualityFunction]:
        return self._ttsqm
        
    def drawNoise(self, target : Target) -> np.ndarray:
        return np.random.multivariate_normal(0, self.getMeasurementNoiseMatrix(target))

    def sensingQualityFunction(self, target : Target) -> SensingQualityFunction:
        return self.targetToSQFMapper()[target]

    def setSensingQualityFunction(self, target : Target, sqf : SensingQualityFunction) -> None:
        assert(isinstance(target, Target))
        assert(isinstance(sqf, SensingQualityFunction))
        self.targetToSQFMapper()[target] = sqf

    def getSensingQuality(self, target : Target) -> float:
        return self.sensingQualityFunction(target)(self.getPosition(), target.p())

    def setNoiseMatrix(self, target : Target, R : np.ndarray) -> None:
        assert(isinstance(target, Target))
        assert(isinstance(R, np.ndarray))
        self._ttRm[target] = R
        self._ttRinvm[target] = np.linalg.inv(R)

    def setMeasurementMatrix(self, target : Target, H : np.ndarray) -> None:
        assert(isinstance(target, Target))
        assert(isinstance(H, np.ndarray))
        self._ttHm[target] = H

    def getMeasurementMatrix(self, target : Target) -> np.ndarray:
        return self._ttHm[target]

    def getMeasurementNoiseMatrix(self, target : Target) -> np.ndarray:
        return self._ttRm[target]

    def getMeasurementNoiseInverseMatrix(self, target : Target) -> np.ndarray:
        return self._ttRinvm[target]

    def getMeasurement(self, p : np.ndarray, target : Target):
        q = target.p()
        quality = self.getSensingQuality(target)(p, q)
        H = self.getMeasurementMatrix(target)
        return quality * H @ target.internalState() + np.random.normal(0, 0.1, H.shape[0])   

def OmegaDot(p, Omega, target : Target, sensor : Sensor, inTargetRegion = False):
    A = target.A
    Q = target.Q
    H = sensor.getMeasurementMatrix(target)
    R_inv = sensor.getMeasurementNoiseInverseMatrix(target)
    unmonitored = Q + A @ Omega + Omega @ A.T
    if inTargetRegion:
        mf = sensor.sensingQualityFunction(target)(p, target.p())
        return unmonitored  - mf*mf*Omega @ H.T @ R_inv @ H @ Omega
    return unmonitored

def UnmonitoredOmegaSimulator(target : Target, sensor : Sensor, N) -> cad.Function:
    # states
    no = target.getNumberOfStates()
    Omega = cad.SX.sym('Omega', no*no)
    nx = Omega.shape[0]
    
    # parameters
    tf = cad.SX.sym('tf', 1)
    Omega0 = cad.SX.sym('Omega0', no*no)
    params = cad.vertcat(tf, Omega0)

    # Model equations
    omegaDot = OmegaDot(None, Omega, target, sensor, False)

    # Objective term
    L = 0
    for i in range(no):
        L += Omega[i*no + i]

    # Fixed step Runge-Kutta 4 integrator
    M = 4 # RK4 steps per interval
    DT = tf/N/M
    f = cad.Function('f', [Omega, params], [omegaDot, L])
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

    omegaSim = cad.Function('OmegaSim', [params], [J, cad.vertcat(*mse), cad.vertcat(*OmegaTrajectory)], ['p'], ['Ik', 'mse', 'OmegaTrajectory'])
    return omegaSim

def cadToNP(x : cad.SX, nrow=None, ncol=None) -> np.ndarray:
    if nrow is None:
        return x.full().flatten()
    if ncol is None:
        return x.full().flatten().reshape(-1,nrow)
    return np.reshape(x.full().flatten(), (nrow, ncol), order='F')

def SimulateUnmonitoredOmega(fun : cad.Function, tf, Omega0):
    
    # set params and evaluate
    params = cad.vertcat(tf, np.matrix(Omega0).getA1())
    sim = fun(params)
    
    # get dimensions
    no = int(np.sqrt(len(Omega0)))
    N = int(len(sim[1].full().flatten())/no) - 1

    # generate trajectories
    t_grid = np.array([tf/N*k for k in range(N+1)]).flatten()
    Ik = sim[0].full().flatten()[0]
    mseTrajectory = Trajectory(cadToNP(sim[1], no, N+1), t_grid)
    omegaTrajectory = Trajectory(cadToNP(sim[2], no*no, N+1), t_grid)
    
    # gradient: dIk_dtf is given by the loss function evaluated at the terminal time
    dIk_dtf = mseTrajectory.getEndPoint()

    return mseTrajectory, omegaTrajectory, Ik, dIk_dtf

class SwitchingPoint:
    def __init__(self, p : np.ndarray):
        self._p : np.ndarray = p

    def p(self) -> np.ndarray:
        return self._p

    def plot(self, ax : plt.Axes = plt, **kwargs) -> PlotObject:
        if kwargs.get('marker') is None:
            kwargs['marker'] = 'o'
        return PlotObject(ax.plot(self._p[0], self._p[1], **kwargs))

class SwitchingParameters:
    def __init__(self, phi : SwitchingPoint, psi : SwitchingPoint, tf : float, Omega0 : Dict[Target, np.ndarray] = {}, N = 100):
        self._phi : SwitchingPoint = phi
        self._psi : SwitchingPoint = psi
        self._Omega0 : Dict[Target, np.ndarray] = Omega0
        self._tf : float = tf
        self._N = N

class CovarianceParameters:
    def __init__(self, target : Target, sensor : Sensor, Omega0 : np.ndarray):
        self.A : np.ndarray = target.A
        self.Q : np.ndarray = target.Q
        self.H : np.ndarray = sensor.getMeasurementMatrix(target)
        self.R_inv : np.ndarray = sensor.getMeasurementNoiseInverseMatrix(target)
        self.sqf : SensingQualityFunction = sensor.sensingQualityFunction(target)

class TrajectorySegment:
    def __init__(self, ucs : Dict[Target, cad.Function], params : SwitchingParameters = None):
        self.pTrajectory : Trajectory = None
        self.uTrajectory : Trajectory = None
        self.mseTrajectories : Dict[Target, Trajectory] = {}
        self.params : SwitchingParameters = params
        self._cost : float = None
        self._gradient_tau = None

        self._ucs : Dict[Target : cad.Function] = ucs
        self._cov_f : Dict[Target, np.ndarray] = {}

    @abstractclassmethod
    def update(self) -> None:
        pass

    # getters
    def getStartPoint(self) -> np.ndarray:
        return self.pTrajectory.x[:,0]
    
    def getEndPoint(self) -> np.ndarray:
        return self.pTrajectory.x[:,-1]
    
    def getDuration(self) -> float:
        return self.pTrajectory.t[-1] - self.pTrajectory.t[0]

    def getCost(self) -> float:
        return self._cost
    
    def getGradient(self):
        return self._gradient_tau

    def getTerminalCovarianceMatrices(self) -> Dict[Target, np.ndarray]:
        return self._cov_f
        
    # modifiers
    def updateInitialCovarianceMatrices(self, omega0 : Dict[Target, np.ndarray]) -> None:
        self.params._Omega0 = omega0

    def updateTerminalCovarianceMatrix(self, target : Target, omega_f : np.ndarray) -> None:
        self._cov_f[target] = omega_f

    def updateMSETrajectory(self, target : Target, mseTrajectory : Trajectory) -> None:
        self.mseTrajectories[target] = mseTrajectory

    def shiftTime(self, t0 : float) -> None:
        self.pTrajectory.shiftTime(t0)
        self.uTrajectory.shiftTime(t0)
        for target in self.mseTrajectories.keys():
            self.mseTrajectories[target].shiftTime(t0)

    # plotters
    def plotInMissionSpace(self, ax : plt.Axes = plt, **kwargs) -> PlotObject:
        return self.pTrajectory.plotStateVsState(0, 1, ax, **kwargs)

class MonitoringController:
    def __init__(self, target : Target, sensor : Sensor, N : int = 100) -> None:
        self.solver : NLPSolver = None   
        self.target : Target = target                           # target to be monitored
        self.N : int = N                                        # number of control intervals
        self.nx : int = None                                    # number of states (assigned by builder)
        self.nu : int = None                                    # number of controls (assigned by builder)
        self.no : int = None                                    # number of target states (assigned by builder)
        self.add_region_constraints = False
        self.buildOptimalMonitoringSolver(target, sensor)

    def buildOptimalMonitoringSolver(self, target : Target, sensor : Sensor) -> None:
        
        # states
        no = target.getNumberOfStates()
        p = cad.SX.sym('p', 2)
        Omega = cad.SX.sym('Omega', no*no)
        x = cad.vertcat(p, Omega)
        nx = x.shape[0]

        # controls
        u = cad.SX.sym('u', 2)       
        nu = u.shape[0]
        
        # parameters
        phi = cad.SX.sym('phi', 2)
        psi = cad.SX.sym('psi', 2)
        tf = cad.SX.sym('tf', 1)
        Omega0 = cad.SX.sym('Omega0', no*no)
        params = cad.vertcat(phi, psi, tf, Omega0)

        T = tf                      # Time horizon
        N = self.N                  # number of control intervals

        # Model equations
        dynamics = target.region().dynamics()
        assert(isinstance(dynamics, ConstantDynamics))
        pDot = dynamics.v() + u
        omegaDot = OmegaDot(p, Omega, target, sensor, True)
        xDot = cad.vertcat(pDot, omegaDot)

        # Objective term
        L = 0
        for i in range(no):
            L += Omega[i*no + i]
            
        # Fixed step Runge-Kutta 4 integrator
        M = 4 # RK4 steps per interval
        DT = T/N/M
        f = cad.Function('f', [x, u, params], [xDot, L])
        X0 = cad.SX.sym('X0', nx)
        U = cad.SX.sym('U', nu)
        X = X0
        Q = 0
        for j in range(M):
            k1, k1_q = f(X, U, params)
            k2, k2_q = f(X + DT/2 * k1, U, params)
            k3, k3_q = f(X + DT/2 * k2, U, params)
            k4, k4_q = f(X + DT * k3, U, params)
            X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
            Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
        F = cad.Function('F', [X0, U, params], [X, Q],['x0','u0','p'],['xf','qf'])

        # Region constraints
        region = target.region()
        assert(isinstance(region, CPRegion))
        g_constr = region.g()
        b_constr = region.b()
        g_constr_term = []
        for i in g_constr.keys():
            g_constr_term.append(cad.dot(g_constr[i], X0[0:2]) - b_constr[i])
        R = cad.Function('region', [X0], [cad.vertcat(*g_constr_term)], ['x0'], ['region'])

        lbx = -np.inf*np.ones(nx)
        ubx = -lbx
        lbu = -np.inf*np.ones(nu)
        ubu = -lbu
        x0 = np.zeros(nx)
        u0 = np.zeros(nu)

        # Start with an empty NLP
        w=[]
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g=[]
        lbg = []
        ubg = []

        # "Lift" initial conditions
        Xk = cad.SX.sym('X0', nx)
        w += [Xk]
        lbw += [lbx]
        ubw += [ubx]
        w0 += [x0]

        # Initial constraints
        g += [cad.vertcat(Xk[0:2]-phi, Xk[2:] - Omega0)]
        lbg += [np.zeros(nx)]
        ubg += [np.zeros(nx)]
        
        # Formulate the NLP
        for k in range(N):
            # New NLP variable for the control
            Uk = cad.SX.sym('U_' + str(k), nu)
            w   += [Uk]
            lbw += [lbu]
            ubw += [ubu]
            w0  += [u0]

            # Integrate till the end of the interval
            Fk = F(x0=Xk, u0=Uk, p=params)
            Xk_end = Fk['xf']
            J=J+Fk['qf']

            # New NLP variable for state at end of interval
            Xk = cad.SX.sym('X_' + str(k+1), nx)
            w   += [Xk]
            lbw += [lbx]
            ubw += [ubx]
            w0  += [x0]

            # Add equality constraint
            g   += [Xk_end-Xk]
            lbg += [np.zeros(nx)]
            ubg += [np.zeros(nx)]

            # Add control bound
            g   += [cad.dot(Uk, Uk)]
            lbg += [-np.inf]
            ubg += [1]

            # Add region constraints
            if (self.add_region_constraints):
                g   += [R(Xk)]
                lbg += [-np.inf*np.ones(len(g_constr_term))]
                ubg += [np.zeros(len(g_constr_term))]

        # Terminal constraint
        g   += [Xk[0:2]-psi]
        lbg += [np.zeros(2)]
        ubg += [np.zeros(2)]

        # Create an NLP solver
        prob = {'f': J, 'x': cad.vertcat(*w), 'g': cad.vertcat(*g), 'p': params}
        
        # Allocate an NLP solver
        self.solver = NLPSolver(prob, w0, lbw, ubw, lbg, ubg, quiet=True)
        self.nx = nx
        self.nu = nu
        self.no = no
        
    def optimalMonitoringControl(self, params : SwitchingParameters) -> Tuple[Trajectory, Trajectory, Trajectory, Trajectory, float, float]:
        self.solver.params = cad.vertcat(*[params._phi.p(), params._psi.p(), np.array(params._tf), np.matrix(params._Omega0[self.target]).getA1()])
        sol = self.solver.solve()

        if not self.solver.solver.stats()['success']:
            params._psi.plot(color='green')
            params._phi.plot(color='blue')
            print("|| psi - phi || = ", np.linalg.norm(params._psi.p() - params._phi.p()))
            print("Monitoring time: ", params._tf)
            print("Minimum travel time: ", self.target.region().TravelCost(params._phi.p(), params._psi.p()))
            print("Initial Uncertainty: ", params._Omega0[self.target])
            raise Exception("Optimal monitoring control failed to converge...")
        w_opt = sol['x'].full()

        # Plot the solution
        nx = self.nx; nu = self.nu; no = self.no; N = self.N 
        p = np.zeros((2, N+1))
        omega = np.zeros((no*no, N+1))
        mse = np.zeros((1, N+1))
        u = np.nan*np.zeros((nu, N+1))
        p[0,:] = w_opt[0::nx+nu].flatten()
        p[1,:] = w_opt[1::nx+nu].flatten()
        for i in range(no*no):
            omega[i,:] = w_opt[2+i::nx+nu].flatten()
            if i % no == 0:
                mse[0,:] = mse[0,:] + omega[i,:]

        u[0,0:-1] = w_opt[nx::nx+nu].flatten()
        u[1,0:-1] = w_opt[nx+1::nx+nu].flatten()
        t_grid = np.array([params._tf/N*k for k in range(N+1)])

        pTrajectory = Trajectory(p, t_grid)
        mseTrajectory = Trajectory(mse, t_grid)
        omegaTrajectory = Trajectory(omega, t_grid)
        uTrajectory = Trajectory(u, t_grid)

        # extract trajectories
        return pTrajectory, mseTrajectory, omegaTrajectory, uTrajectory, sol['f'].full().flatten()[0], sol['lam_p']

class MonitoringSegment(TrajectorySegment):
    def __init__(self, target : Target, sensor : Sensor, ucs : Dict[Target, cad.Function], params : SwitchingParameters = None):
        self._target : Target = target
        self.monitoring_controller : MonitoringController = MonitoringController(target, sensor, N = params._N)
        super().__init__(ucs, params)

    def update(self):
        p, m, omega, u, Jk, duals_p = self.monitoring_controller.optimalMonitoringControl(self.params)
        self.pTrajectory = p
        self.updateMSETrajectory(self._target, m)
        self.uTrajectory = u
        self.updateTerminalCovarianceMatrix(self._target, omega.getEndPoint())
        
        self._cost = Jk
        self._gradient_tau = -duals_p[4].full().flatten()
        for target in self._ucs.keys():
            if target == self._target:
                continue
            mse, Omega, Ik, dIk_dt = SimulateUnmonitoredOmega(self._ucs[target], self.params._tf, self.params._Omega0[target])
            self.updateMSETrajectory(target, mse)
            self.updateTerminalCovarianceMatrix(target, Omega.getEndPoint())
            self._cost += Ik
            self._gradient_tau += dIk_dt       

class SwitchingSegment(TrajectorySegment):
    def __init__(self, ucs : Dict[Target, cad.Function], params : SwitchingParameters = None):
        super().__init__(ucs, params)

    def update(self) -> None:
        self._cost = 0
        self._gradient_tau = 0

        # Don't need to update the control or path, but reset to 0 relative time
        t0 = self.uTrajectory.t[0]
        self.uTrajectory.shiftTime(-t0)
        self.pTrajectory.shiftTime(-t0)

        # update mse trajectories
        for target in self._ucs.keys():
            mse, Omega, Ik, dIk_dtf = SimulateUnmonitoredOmega(self._ucs[target], self.params._tf, self.params._Omega0[target])
            self.updateMSETrajectory(target, mse)
            self.updateTerminalCovarianceMatrix(target, Omega.getEndPoint())
            self._cost += Ik
            self._gradient_tau += dIk_dtf

class Cycle:
    def __init__(self, ts : List[TrajectorySegment], omega0 : Dict[Target, np.ndarray] = {}, t0 : float = 0.0, counter : int = 0):
        self.pTrajectory : Trajectory = None
        self.uTrajectory : Trajectory = None
        self.mseTrajectories : Dict[Target, Trajectory] = {}
        self._trajectorySegments : List[TrajectorySegment] = ts
        self._covAtCycleStart : Dict[Target, np.ndarray] = omega0
        self._covAtCycleEnd : Dict[Target, np.ndarray] = None

        # statistics
        self._cycle_start : float = t0
        self._counter : int = counter

        # plot controls
        self._switchColor = 'blue'
        self._monitorColor = 'red'
            
    def simulate(self) -> None:
        t0 = self._cycle_start
        omega0 = self._covAtCycleStart.copy()
        self.clearTrajectories()
        for ts in self._trajectorySegments:
            ts.updateInitialCovarianceMatrices(omega0)
            ts.update()
            omega0 = ts.getTerminalCovarianceMatrices()
            ts.shiftTime(t0)
            t0 += ts.getDuration()

            self.appendTrajectorySegment(ts)
            
        self._covAtCycleEnd = omega0

    # getters
    def getDuration(self) -> float:
        return sum([ts.getDuration() for ts in self._trajectorySegments])

    def getStartTime(self) -> float:
        return self._cycle_start
    
    def getEndTime(self) -> float:
        return self._cycle_start + self.getDuration()

    def getCost(self) -> float:
        return sum(ts.getCost() for ts in self._trajectorySegments)
    
    def getGradient(self):
        monSeg : List[TrajectorySegment] = []
        for ts in self._trajectorySegments:
            if isinstance(ts, MonitoringSegment):
                monSeg.append(ts)
        return np.array([ts.getGradient() for ts in monSeg]).flatten()

    def getInitialCovarianceMatrices(self) -> Dict[Target, np.ndarray]:
        return self._covAtCycleStart
    
    def getTerminalCovarianceMatrices(self) -> Dict[Target, np.ndarray]:
        return self._covAtCycleEnd.copy()
    
    def steadyState(self, tol=1e-2) -> bool:
        for target in self._covAtCycleStart.keys():
            if self._covAtCycleEnd is None or not np.allclose(self._covAtCycleStart[target], self._covAtCycleEnd[target], atol=tol):
                return False
        return True
    
    # modifiers
    def updateInitialCovarianceMatrices(self, omega0 : Dict[Target, np.ndarray]) -> None:
        self._covAtCycleStart = omega0.copy()
    
    def shiftTime(self, deltaT : float) -> None:
        self._cycle_start += deltaT
        for ts in self._trajectorySegments:
            ts.shiftTime(deltaT)
        if self.pTrajectory is not None:
            self.pTrajectory.shiftTime(deltaT)
            self.uTrajectory.shiftTime(deltaT)
            for target in self.mseTrajectories.keys():
                self.mseTrajectories[target].shiftTime(deltaT)

    def clearTrajectories(self) -> None:
        self.pTrajectory = None
        self.uTrajectory = None
        self.mseTrajectories = {}

    def appendTrajectorySegment(self, ts : TrajectorySegment) -> None:
        if self.pTrajectory is None:
            self.pTrajectory = Trajectory(ts.pTrajectory.x, ts.pTrajectory.t)
            self.uTrajectory = Trajectory(ts.uTrajectory.x, ts.uTrajectory.t)
            for target in ts.mseTrajectories.keys():
                self.mseTrajectories[target] = Trajectory(ts.mseTrajectories[target].x, ts.mseTrajectories[target].t)
        else:
            self.pTrajectory.appendTrajectory(ts.pTrajectory)
            self.uTrajectory.appendTrajectory(ts.uTrajectory)
        for target in ts.mseTrajectories.keys():
            self.mseTrajectories[target].appendTrajectory(ts.mseTrajectories[target])

    # plotters
    def plot(self, ax : plt.Axes = plt, **kwargs) -> PlotObject:
        eka = extend_keyword_args(plotAttributes.agent.getAttributes(), **kwargs)
        return self.pTrajectory.plotStateVsState(0, 1, ax, **eka)
    
    def plotControls(self, ax : plt.Axes = plt, add_monitoring_labels=True, **kwargs) -> PlotObject:
        po = PlotObject()
        po.add(self.plotMonitoringControls(ax, add_monitoring_labels=add_monitoring_labels, **kwargs))
        po.add(self.plotSwitchingControls(ax, **kwargs))
        return po
    
    def plotSwitchingControls(self, ax : plt.Axes, **kwargs) -> PlotObject:
        po = PlotObject()
        u1_pA = plotAttributes.u1_switch
        u2_pA = plotAttributes.u2_switch
        un_pA = plotAttributes.u_norm_switch
        eka1 = extend_keyword_args(u1_pA.getAttributes(), **kwargs)
        eka2 = extend_keyword_args(u2_pA.getAttributes(), **kwargs)
        eka3 = extend_keyword_args(un_pA.getAttributes(), **kwargs)
        for ts in self._trajectorySegments:
            if not isinstance(ts, SwitchingSegment):
                continue
            u_norm = np.sqrt(np.square(ts.uTrajectory.x[0,:])+np.square(ts.uTrajectory.x[1,:]))
            po.add(ax.plot(ts.uTrajectory.t, u_norm, **eka3))
            po.add(ts.uTrajectory.plotStateVsTime(0, ax, **eka1))
            po.add(ts.uTrajectory.plotStateVsTime(1, ax, **eka2))
        return po
    
    def plotMonitoringControls(self, ax : plt.Axes, add_monitoring_labels=True, **kwargs) -> PlotObject:
        po = PlotObject()
        u1_pA = plotAttributes.u1_monitor
        u2_pA = plotAttributes.u2_monitor
        un_pA = plotAttributes.u_norm_monitor
        eka1 = extend_keyword_args(u1_pA.getAttributes(), **kwargs)
        eka2 = extend_keyword_args(u2_pA.getAttributes(), **kwargs)
        eka3 = extend_keyword_args(un_pA.getAttributes(), **kwargs)
        for ts in self._trajectorySegments:
            if not isinstance(ts, MonitoringSegment):
                continue

            if ts == self._trajectorySegments[-1]:
                eka1['label'] = '$u_1$'
                eka2['label'] = '$u_2$'
                eka3['label'] = '$\|u\|$'

            u_norm = np.sqrt(np.square(ts.uTrajectory.x[0,:])+np.square(ts.uTrajectory.x[1,:]))
            po.add(ax.plot(ts.uTrajectory.t, u_norm, **eka3))
            po.add(ts.uTrajectory.plotStateVsTime(0, ax, **eka1))
            po.add(ts.uTrajectory.plotStateVsTime(1, ax, **eka2))
   
            if add_monitoring_labels:
                tText = np.median(ts.uTrajectory.t)
                uText = np.nanmedian(u_norm) + 0.1
                po.add(ax.text(tText, uText, ts._target.name))

        return po

    def plotMSE(self, ax : plt.Axes = plt, add_labels=False, **kwargs) -> PlotObject:
        po = PlotObject()
        i = 0
        for target in self.mseTrajectories.keys():
            eka = kwargs.copy()
            if add_labels:
                eka['label'] = target.name
            eka = extend_keyword_args({'color': plotAttributes.target_colors[-i]}, **eka)
            po.add(self.mseTrajectories[target].plotStateVsTime(0, ax, **eka))
            mseStart = self.mseTrajectories[target].getInitialValue() 
            po.add(ax.hlines(mseStart, self._cycle_start, self._cycle_start + self.getDuration(), alpha=0.2, color=eka['color']))
            i += 1
        if add_labels:
            plt.legend()
        return po
    
class Agent:
    def __init__(self, world : World, sensor : Sensor) -> None:
        self._world : World = world
        self._sensor : Sensor = sensor
        self._gpp : GlobalPathPlanner = None
        self._tvs : List[Target] = []
        self._switchingPaths : List[Tree] = []
        self._cycle : Cycle = None

        # optimization parameters
        self._tau : Dict[int, float] = {}                                   # map a target visit index to a monitoring duration
        self._lambda : Dict[int, float] = {}                                # map a target visit index to a monitoring duration dual
        self._tau_min : Dict[int, float] = {}                               # map a target visit index to a minimum monitoring duration
        self._op : OptimizationParameters = OptimizationParameters()        # optimization parameters
        
        # optimization statistics
        self._kkt_residuals : Dict[int, float] = {}                         # map a target visit index to a KKT residual
        self._global_costs : List[float] = []                               # global cost (per steady state cycle) 
        self._global_gradient_norms : List[float] = []                      # global gradient norm (per steady state cycle) 
        self._tau_vals : List[np.ndarray] = []                              # monitoring durations (per steady state cycle)
        self._kkt_violations : List[np.ndarray] = []                        # KKT residuals (per steady state cycle)
        self._steady_state_iters : List[int] = []                           # number of iterations to reach steady state
        self._alphas : List[float] = []                                     # step sizes (per steady state cycle)
        self._isSteadyState : List[bool] = []                               # steady state reached (per steady state cycle)

        # decomposition
        self._switchingSegments : List[SwitchingSegment] = []
        self._monitoringSegments : List[MonitoringSegment] = []
        self._N = 100                                                       # number of discretization nodes per region

        # simulation functions for forward integration of covariance matrices
        self._ucs : Dict[Target, cad.Function] = {}
        self._gradient_ucs : Dict[Target, cad.Function] = {}

        self.initialize()

    def initialize(self) -> None:
        for target in self.world().targets():
            self._ucs[target] = UnmonitoredOmegaSimulator(target, self.sensor(), self._N)
        
        self._gpp = GlobalPathPlanner(self.world())
        self._gpp._plot_options.toggle_all_plotting(False)
   
    def computeVisitingSequence(self) -> None:
        self.gpp().SolveTSP()
        self._tvs = self.gpp().tsp.getTargetVisitingSequence()

    def refineVisitingSequence(self) -> None:
        # If we switch through another target region along the way
        # let us add the target to the sequence
        i = 0
        tvs = []
        for i in range(len(self._tvs)):
            old_target = self._tvs[i-1]
            current_target = self._tvs[i]
            switchPath = self.gpp().target_paths[old_target][current_target]
            p : Tree = switchPath.getParent()
            while not p.isRoot():
                region = p.getData().active_region_to_parent()
                for target in self.world().targets():
                    if region == target.region():
                        tvs.append(target)
                p = p.getParent()
            i += 1
        self._tvs = tvs
    
    def optimizeCycle(self) -> None:
        self.initializeCycle()
        it = 0
        po = PlotObject()
        while True:
            steady, ssc = self.simulateToSteadyState(maxIter=self._op._steady_state_iters)
            self._steady_state_iters.append(ssc)
            self._isSteadyState.append(steady)

            if not steady:
                Warning("Did not reach steady state...")

            # po.remove()
            # po.add(self._cycle.plot())
            dJ_dt = self.updateMonitoringDurations(it)

            # Compute first order stationarity condition
            for i in range(len(self._tvs)):
                self._kkt_residuals[i] = dJ_dt[i] - self._lambda[i]
            self._kkt_violations.append(np.array(list(self._kkt_residuals.values())))
            
            self.printIteration(it)

            if np.max(np.abs(self._kkt_violations[-1])) < self._op._kkt_tolerance and steady:
                print("Optimal cycle found!")
                break

            if it > self._op._optimization_iters:
                print("Max iterations reached...")
                break

            it += 1
    
    def initializeCycle(self) -> None:
        
        if (len(self._tvs) <= 1):
            raise Exception("Expected at least two targets in the visiting sequence. Did you run 'computeVisitingSequence()'?")
                    
        # create lists of trajectory segments
        self._switchingSegments, self._tvs = self.initializeSwitchingSegments()
        self._monitoringSegments = self.initializeMonitoringSegments()
        trajectorySegments : List[TrajectorySegment] = []

        # combine trajectory segments
        for i in range(len(self._tvs)):
            trajectorySegments.append(self._switchingSegments[i])
            trajectorySegments.append(self._monitoringSegments[i])

        # assign cycle
        self._cycle = Cycle(trajectorySegments, self.getInitialCovarianceMatrices())

    def initializeSwitchingSegments(self) -> Tuple[List[SwitchingSegment], List[Target]]:
        segments : List[SwitchingSegment] = []
        refined_tvs = []

        for i in range(len(self._tvs)):
            old_target = self._tvs[i-1]
            current_target = self._tvs[i]
            switchPath : Tree = self.gpp().target_paths[old_target][current_target].getParent()

            while switchPath:
                switchSegment, target, switchPath = self.extractSwitchingSegment(switchPath)
                segments.append(switchSegment)
                refined_tvs.append(target)
        
        return segments, refined_tvs

    def extractSwitchingSegment(self, path : Tree) -> Tuple[SwitchingSegment, Target, Tree]:
        '''
        Move up the tree and extract the switching points until a target region is reached.
        We then return the switching segment together with the remaining tree beginning
        from the first node where the active region is not the reached target region.
        '''
        assert(isinstance(path, Tree))
        node : Tree = path
        assert(node is not None)
        phi = node.getData().p().reshape(-1,1)
        tf = np.array(0).reshape(1)
        u = np.nan*np.zeros((2,1))
        pTrajectory = Trajectory(phi, tf)
        uTrajectory = Trajectory(u, tf)
        ep = SwitchingPoint(phi)
        dp = ep
        while node is not None and not node.isRoot():

            # check if we have reached a target region
            done = False
            for target in self.world().targets():
                if node.getData().active_region_to_parent() == target.region():
                    while node.hasParent() and node.getData().active_region_to_parent() == target.region():
                        sp = SwitchingParameters(ep,dp,tf,N=self._N)
                        ts = SwitchingSegment(self._ucs, sp)
                        ts.pTrajectory = pTrajectory
                        ts.uTrajectory = uTrajectory
                        node = node.getParent() if node.getParent().hasParent() else None
                        return ts, target, node

            # otherwise add the next switching point
            psi = node.getParent().getData().p().reshape(-1,1)
            dp = SwitchingPoint(psi)
            deltaT = node.getData().costToParent()
            
            # TODO(Jonas): Hacky solution currently in place
            # What I need here is the control from one node to the next
            # In my current setting (constant Dynamics on the regions, this is a constant control law)
            # In general doesn't need to be... 
            # So really, should have a trajectory to parent stored in the node or even better in an edge between the two nodes...
            # Will have to fix a lot of things, so let's improve this after CDC submission
            v = node.getData().active_region_to_parent().dynamics().v().reshape(-1,1)
            u = (psi - phi)/deltaT - v
            
            # update trajectories
            uTrajectory.extend(u, tf)
            uTrajectory.extend(u, tf + deltaT)
            pTrajectory.extend(psi, tf + deltaT)

            # update time
            tf = tf + deltaT
            phi = psi

            # get next node
            node = node.getParent()

        raise Exception("Should not be reached ...")

    def initializeMonitoringSegments(self) -> List[MonitoringSegment]:
        segments : List[MonitoringSegment] = []
        for i in range(len(self._tvs)):
            target = self._tvs[i]
            phi = self._switchingSegments[i].getEndPoint()
            psi = self._switchingSegments[(i+1) % len(self._switchingSegments)].getStartPoint()
            tf = max(0.1, 1.1*target.region().TravelCost(phi, psi))
            self._tau[i] = tf
            self._tau_min[i] = target.region().TravelCost(phi, psi)

            params = SwitchingParameters(SwitchingPoint(phi), SwitchingPoint(psi), tf, N=self._N)
            segment = MonitoringSegment(target, self.sensor(), self._ucs, params)
            segments.append(segment)
        return segments

    def simulateToSteadyState(self, maxIter = 100) -> Tuple[bool, int]:
        it = 0
        omega_f = self._cycle.getInitialCovarianceMatrices()
        while True:
            it += 1

            self._cycle.simulate()
            omega_f = self._cycle.getTerminalCovarianceMatrices()
            
            cycle_average_cost = self._cycle.getCost()/self._cycle.getDuration()
            self._global_costs.append(cycle_average_cost)
            
            isSteady = self._cycle.steadyState(tol=self._op._sim_to_steady_state_tol)
            self._cycle.updateInitialCovarianceMatrices(omega_f)

            if isSteady:
                return True, it

            if it >= maxIter:
                return False, it
            
            self._cycle._cycle_start += self._cycle.getDuration()
    
    def globalCostGradient(self):
        T = self._cycle.getDuration()
        J = self._cycle.getCost()
        dJ_dt = self._cycle.getGradient()
        
        # global average cost gradient
        return (dJ_dt * T - J * np.ones(len(dJ_dt)))/T**2

    def updateMonitoringDurations(self, it) -> None:
        '''
        Simple projected gradient descend
        '''
        dJ_dt = self.globalCostGradient()
        self._global_gradient_norms.append(np.linalg.norm(dJ_dt, ord=np.inf))
        if self._global_gradient_norms[-1] > self._op._tr:
            dJ_dt = dJ_dt * self._op._tr / self._global_gradient_norms[-1]
        
        self._tau_vals.append(np.array([self._tau[i] for i in range(len(self._tvs))]))
        for i in range(len(self._tvs)):
            if self._tau[i] == self._tau_min[i] and dJ_dt[i] > 0:
                self._lambda[i] = dJ_dt[i]
            else:
                self._lambda[i] = 0
                self._tau[i] = max(self._tau_min[i] + self._op._sigma, self._tau[i] - self._op._alpha * dJ_dt[i])
            self._monitoringSegments[i].params._tf = self._tau[i]
        self._op._alpha *= self._op._beta
        self._alphas.append(self._op._alpha)
        return dJ_dt

    # Getters
    def gpp(self) -> GlobalPathPlanner:
        return self._gpp   

    def world(self) -> World:
        return self._world

    def sensor(self) -> Sensor:
        return self._sensor
    
    def getInitialCovarianceMatrices(self) -> Dict[Target, np.ndarray]:
        Omegas = {}
        for target in self.world().targets():
            Omegas[target] = np.eye((target.getNumberOfStates()))
        return Omegas        
    
    # Plotters
    def plotMSE(self, ax : plt.Axes = plt, add_labels=False, **kwargs) -> PlotObject:
        return self._cycle.plotMSE(ax=ax, add_labels=add_labels, **kwargs)
    
    def plotControls(self, ax : plt.Axes = plt, add_monitoring_labels=False, **kwargs) -> PlotObject:
        return self._cycle.plotControls(ax, add_monitoring_labels=add_monitoring_labels, **kwargs)

    def plotCycle(self, ax : plt.Axes = plt, **kwargs) -> PlotObject:
        po = PlotObject()
        self._cycle.plot(ax, **kwargs)
        return po
    
    def plotMonitoringSegments(self, ax : plt.Axes = plt, **kwargs) -> PlotObject:
        po = PlotObject()
        for ms in self._monitoringSegments:
            po.add(ms.pTrajectory.plotStateVsState(0, 1, ax, **kwargs))
        return po
    
    def plotEntryPoints(self, ax : plt.Axes = plt, **kwargs) -> PlotObject:
        po = PlotObject()
        eka = extend_keyword_args(plotAttributes.phi.getAttributes(), **kwargs)
        for ms in self._monitoringSegments:
            po.add(ms.params._phi.plot(ax, **eka))
        return po
    
    def plotDeparturePoints(self, ax : plt.Axes = plt, **kwargs) -> PlotObject:
        po = PlotObject()
        eka = extend_keyword_args(plotAttributes.psi.getAttributes(), **kwargs)
        for ms in self._monitoringSegments:
            po.add(ms.params._psi.plot(ax, **eka))
        return po

    def plotSwitchingPoints(self, ax : plt.Axes = plt, **kwargs) -> PlotObject:
        po = PlotObject()
        for ms in self._monitoringSegments:
            eka = extend_keyword_args(plotAttributes.phi.getAttributes(), **kwargs)
            po.add(ms.params._phi.plot(ax, **eka))
            
            eka = extend_keyword_args(plotAttributes.psi.getAttributes(), **kwargs)
            po.add(ms.params._psi.plot(ax, **eka))
        return po

    def plotSwitchingSegments(self, ax : plt.Axes = plt, **kwargs) -> PlotObject:
        po = PlotObject()
        for ss in self._switchingSegments:
            po.add(ss.pTrajectory.plotStateVsState(0, 1, ax, **kwargs))
        return po

    def plotGlobalCosts(self, ax : plt.Axes, **kwargs) -> PlotObject:
        return PlotObject(ax.plot(self._global_costs, **kwargs))
    
    def plotGlobalGradientNorms(self, ax : plt.Axes, **kwargs) -> PlotObject:
        return PlotObject(ax.plot(self._global_gradient_norms, **kwargs))
    
    def plotTauVals(self, ax : plt.Axes, add_lower_bounds=True, **kwargs) -> PlotObject:
        po = PlotObject()
        tv = np.array(self._tau_vals)
        for i in range(tv.shape[1]):
            eka = extend_keyword_args({'color': plotAttributes.target_colors[-i]}, **kwargs)
            po.add(ax.plot(tv[:,i], **eka))

            if add_lower_bounds:
                eka = extend_keyword_args({'alpha' : 0.75, 'linestyle' : '--'}, **eka)
                po.add(ax.hlines(self._tau_min[i], 0, len(tv[:,i])-1, **eka))
        return po

    def plotKKTViolations(self, ax : plt.Axes, **kwargs) -> PlotObject:
        po = PlotObject()
        po.add(PlotObject(ax.plot(self._kkt_violations, **kwargs)))
        po.add(PlotObject(ax.axhspan(-self._op._kkt_tolerance, self._op._kkt_tolerance, alpha=0.2, color='green')))
        return po

    def plotAlphas(self, ax : plt.Axes, **kwargs) -> PlotObject:
        return PlotObject(ax.plot(self._alphas, **kwargs))

    def plotSensorQuality(self, ax : plt.Axes, **kwargs) -> PlotObject:
        X, Y, Z = self._world.get_meshgrid()
        sensor = self.sensor()
        zero_threshold = 1e-2
        for i in range(X.shape[0]):
            for j in range(Y.shape[1]):
                p = np.array((X[i,j], Y[i,j]))
                sensor.setPosition(p)
                for target in self._world.targets():
                    assert(isinstance(target, Target))
                    region = target.region()
                    if region.Contains(p):
                        Z[i,j] = sensor.getSensingQuality(target)
        eka = extend_keyword_args(plotAttributes.sensor_quality.getAttributes(), **kwargs)
        cf = ax.contourf(X, Y, Z, **eka)
        # plt.colorbar(res)
        return PlotObject(cf)
    
    def addSteadyStateLines(self, ax : plt.Axes, **kwargs) -> PlotObject:
        cumsum = np.cumsum(self._steady_state_iters)
        po = PlotObject()
        for i in range(len(cumsum)):
            po.add(ax.axvline(cumsum[i], **kwargs))
        return po

    # printers
    def printHeader(self) -> None:
        print("----|-----------|-----------|-----------|----------|--------")
        print(" it | avrg cost | grad. nrm | step size | kkt viol | steady ")
        print("----|-----------|-----------|-----------|----------|--------")
              
    def printIteration(self, it) -> None:
        if it % 10 == 0:
            self.printHeader()
        print("{:3d} | {:9.2e} | {:9.2e} | {:9.2e} | {:7.2e} | {:5d}".format(
            it, self._global_costs[-1], self._global_gradient_norms[-1], self._alphas[-1], np.max(np.abs(self._kkt_violations[-1])), self._steady_state_iters[-1]
        ))
