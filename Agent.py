      
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
        return cad.exp(-self._c3*cad.dot(delta,delta))*(cad.sin(self._c1*delta[0])**2 + cad.cos(self._c2*delta[1])**2)

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
    Rinv = sensor.getMeasurementNoiseInverseMatrix(target)
    unmonitored = Q + A @ Omega + Omega @ A.T
    if inTargetRegion:
        mf = sensor.sensingQualityFunction(target)(p, target.p())
        return unmonitored  - mf*mf*Omega @ H.T @ Rinv @ H @ Omega
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
    Omegadot = OmegaDot(None, Omega, target, sensor, False)

    # Objective term
    L = 0
    for i in range(no):
        L += Omega[i*no + i]

    # Fixed step Runge-Kutta 4 integrator
    M = 4 # RK4 steps per interval
    DT = tf/N/M
    f = cad.Function('f', [Omega, params], [Omegadot, L])
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
    w=[]
    J = 0
    mse = [sum([Omega0[i*no + i] for i in range(no)])]
    
    # "Lift" initial conditions
    Xk = Omega0
    w += [Xk]
    
    # Formulate the integrator
    for k in range(N):

        # Integrate till the end of the interval
        Fk = F(x0=Xk, p=params)
        Xk = Fk['xf']
        J=J+Fk['qf']

        # Add Xk to Omega vec
        mse += [sum([Xk[i*no + i] for i in range(no)])]

    return cad.Function('OmegaSim', [params], [J, cad.vertcat(*mse)], ['p'], ['Ik', 'mse'])

def SimulateUnmonitoredOmega(fun : cad.Function, tf, Omega0, N):
    params = cad.vertcat(tf, np.matrix(Omega0).getA1())
    sim = fun(params)
    Ik = sim[0].full().flatten()
    mse = sim[1].full().flatten().reshape(-1,1).T
    tgrid = np.array([tf/N*k for k in range(N+1)])
    return Trajectory(mse, tgrid), Ik

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
    def __init__(self, phi : SwitchingPoint, psi : SwitchingPoint, tf):
        self._phi : SwitchingPoint = phi
        self._psi : SwitchingPoint = psi
        self._tf : float = tf

class MonitoringParameters(SwitchingParameters):
    def __init__(self, phi : SwitchingPoint, psi : SwitchingPoint, tf : float, Omega0 : np.ndarray = None):
        self._Omega0 : np.ndarray = Omega0
        super().__init__(phi, psi, tf)

class CovarianceParameters:
    def __init__(self, target : Target, sensor : Sensor, Omega0 : np.ndarray):
        self.A : np.ndarray = target.A
        self.Q : np.ndarray = target.Q
        self.H : np.ndarray = sensor.getMeasurementMatrix(target)
        self.Rinv : np.ndarray = sensor.getMeasurementNoiseInverseMatrix(target)
        self.sqf : SensingQualityFunction = sensor.sensingQualityFunction(target)

class TrajectorySegment:
    def __init__(self):
        self.pTrajectory : Trajectory = None
        self.uTrajectory : Trajectory = None

    @abstractclassmethod
    def update(self):
        pass

    def getStartPoint(self) -> np.ndarray:
        return self.pTrajectory.x[:,0]
    
    def getEndPoint(self) -> np.ndarray:
        return self.pTrajectory.x[:,-1]
    
    def getDuration(self) -> float:
        return self.pTrajectory.t[-1] - self.pTrajectory.t[0]

    def plotInMissionSpace(self, ax : plt.Axes = plt, **kwargs) -> PlotObject:
        return self.pTrajectory.plotStateVsState(0, 1, ax, **kwargs)

    def shiftTime(self, deltaT : float) -> None:
        self.pTrajectory.shiftTime(deltaT)
        self.uTrajectory.shiftTime(deltaT)

class SwitchingSegment(TrajectorySegment):
    def __init__(self, path : Tree):
        self._sp : SwitchingParameters = None
        self._path : Tree = path
        super().__init__()
        self.update()

    def update(self) -> None:
        assert(isinstance(self._path, Tree))
        node : Tree = self._path.getParent()
        assert(node is not None)
        phi = node.getData().p().reshape(-1,1)
        tf = np.array(0).reshape(1)
        u = np.nan*np.zeros((2,1))
        self.pTrajectory = Trajectory(phi, tf)
        self.uTrajectory = Trajectory(u, tf)
        while not node.getParent().isRoot():

            # update end point
            psi = node.getParent().getData().p().reshape(-1,1)
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
            self.uTrajectory.extend(u, tf)
            self.uTrajectory.extend(u, tf + deltaT)
            self.pTrajectory.extend(psi, tf + deltaT)

            # update time
            tf = tf + deltaT
            phi = psi

            # get next node
            node = node.getParent()
        
        # set final control interval boundary
        self.uTrajectory.extend(u, tf)

class MonitoringController:
    def __init__(self, target : Target, sensor : Sensor) -> None:
        self.solver : NLPSolver = None   
        self.N : int = 100
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
        pdot = dynamics.v() + u
        Omegadot = OmegaDot(p, Omega, target, sensor, True)
        xdot = cad.vertcat(pdot, Omegadot)

        # Objective term
        L = 0
        for i in range(no):
            L += Omega[i*no + i]
            
        # Fixed step Runge-Kutta 4 integrator
        M = 4 # RK4 steps per interval
        DT = T/N/M
        f = cad.Function('f', [x, u, params], [xdot, L])
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
        self.solver = NLPSolver(prob, w0, lbw, ubw, lbg, ubg)
        self.nx = nx
        self.nu = nu
        self.no = no
        
    def optimalMonitoringControl(self, params : MonitoringParameters) -> Tuple[Trajectory, Trajectory, Trajectory]:
        self.solver.params = cad.vertcat(params._phi.p(), params._psi.p(), params._tf, np.matrix(params._Omega0).getA1())
        sol = self.solver.solve()

        if not self.solver.solver.stats()['success']:
            params._psi.plot(color='green')
            params._phi.plot(color='blue')
            print("\| psi - phi \| = ", np.linalg.norm(params._psi.p() - params._phi.p()))
            print("Monitoring time: ", params._tf)
            print("Initial Uncertainty: ", params._Omega0)
            raise Exception("Optimal monitoring control failed to converge...")
        w_opt = sol['x'].full()

        # Plot the solution
        nx = self.nx; nu = self.nu; no = self.no; N = self.N 
        p = np.zeros((2, N+1))
        mse = np.zeros((1, N+1))
        u = np.nan*np.zeros((nu, N+1))
        p[0,:] = w_opt[0::nx+nu].flatten()
        p[1,:] = w_opt[1::nx+nu].flatten()
        for i in range(no):
            mse[i,:] = mse[i,:] + w_opt[2+i::nx+nu].flatten()

        u[0,0:-1] = w_opt[nx::nx+nu].flatten()
        u[1,0:-1] = w_opt[nx+1::nx+nu].flatten()
        tgrid = np.array([params._tf/N*k for k in range(N+1)])

        pTraj = Trajectory(p, tgrid)
        mseTraj = Trajectory(mse, tgrid)
        uTraj = Trajectory(u, tgrid)

        # extract trajctories
        return pTraj, mseTraj, uTraj

class MonitoringSegment(TrajectorySegment):
    def __init__(self, target : Target, sensor : Sensor, mp : MonitoringParameters):
        self._mp : MonitoringParameters = mp
        self._target : Target = target
        self._mseTrajectory : Trajectory = None
        self.monitoring_controller : MonitoringController = MonitoringController(target, sensor)
        super().__init__()

    def update(self):
        p, m, u = self.monitoring_controller.optimalMonitoringControl(self._mp)
        self.pTrajectory = p
        self.mseTrajectory = m
        self.uTrajectory = u

    def shiftTime(self, deltaT : float) -> None:
        super().shiftTime(deltaT)
        self.mseTrajectory.shiftTime(deltaT)

class Agent:
    def __init__(self, world : World, sensor : Sensor) -> None:
        self._world : World = world
        self._sensor : Sensor = sensor
        self._gpp : GlobalPathPlanner = None

        # things to collect for each cycle
        self._tvs : List[Target] = []
        self._switchingPoints : List[SwitchingPoint] = []
        self._epm : Dict[int, SwitchingPoint] = {}                           # map a target visit index to an entry point
        self._dpm : Dict[int, SwitchingPoint] = {}                           # map a target visit index to a departure point
        self._switchingSegments : List[SwitchingSegment] = []
        self._monitoringSegments : List[MonitoringSegment] = []
        self._cycle_start : float = 0.0

        # things to collect for each target
        self._mseTrajectories : Dict[Target, Trajectory] = {}           # one cycle of mean squared error trajectories
        self._ucs : Dict[Target, cad.Function] = {}

        self.initialize()

    def initialize(self, Omegas : Dict[Target, np.ndarray] = None) -> None:
        for target in self.world().targets():
            self._ucs[target] = UnmonitoredOmegaSimulator(target, self.sensor(), N=200)
        self.initializeCovarianceMatrices(Omegas)
        self._gpp = GlobalPathPlanner(self.world())
        self._gpp._plot_options.toggle_all_plotting(False)
        
    def initializeCovarianceMatrices(self, Omegas : Dict[Target, np.ndarray] = None) -> None:
        if Omegas is None:
            Omegas = {}
            for target in self.world().targets():
                Omegas[target] = np.eye((target.getNumberOfStates()))
        for target in self.world().targets():
            self._mseTrajectories[target] = Trajectory(np.array([np.trace(Omegas[target])]).reshape(-1,1), np.array([0]))

    def computeVisitingSequence(self) -> None:
        self.gpp().SolveTSP()
        self._tvs = self.gpp().tsp.getTargetVisitingSequence()

    def refineVisitingSequence(self) -> None:
        # If we switch through another target region along the way
        # let us add the target to the sequence
        i = 0
        while i < len(self._tvs):
            old_target = self._tvs[i-1]
            current_target = self._tvs[i]
            switchPath = self.gpp().target_paths[old_target][current_target]
            p : Tree = switchPath.getParent()
            appended = False
            while not p.getParent().isRoot():
                region = p.getData().active_region_to_parent()
                for target in self.world().targets():
                    if region == target.region():
                        self._tvs.insert(i, target)
                        appended = True
                        break
                p = p.getParent()
            
            if not appended:
                i += 1

    def getCycleTime(self) -> float:
        mst = sum([ms.getDuration() for ms in self._monitoringSegments])
        sst = sum([ss.getDuration() for ss in self._switchingSegments])
        return mst + sst

    def initializeCycle(self) -> None:
        
        if (len(self._tvs) <= 1):
            Warning("Expected at least two targets in the visiting sequence...")
            return
        
        # cache the following timings for initializing monitoring time (index is corresponding to ):
        # 1) time from target to switching point
        # 2) time from switching point to next target
        target_to_switching : Dict[int, float] = {}
        switching_to_target : Dict[int, float] = {}
        self._switchingSegments.clear()
        self._monitoringSegments.clear()
        self._switchingPoints.clear()

        # assign switching segments
        for i in range(len(self._tvs)):
            old_target = self._tvs[i-1]
            current_target = self._tvs[i]
            switchPath = self.gpp().target_paths[old_target][current_target]
            switchSegment = SwitchingSegment(switchPath)
            self._switchingSegments.append(switchSegment)
            
            startSp = SwitchingPoint(switchSegment.getStartPoint())
            self._switchingPoints.append(startSp)
            self._dpm[(i-1) % len(self._tvs)] = self._switchingPoints[-1]

            if switchSegment.getDuration() > 0:
                endSp = SwitchingPoint(switchSegment.getEndPoint())
                self._switchingPoints.append(endSp)
            self._epm[i] = self._switchingPoints[-1]             

            # update cache values
            target_to_switching[(i-1) % len(self._tvs)] = switchPath.getData().costToParent()
            p : Tree = switchPath.getParent()
            while not p.isRoot():
                switching_to_target[i] = p.getData().costToParent()
                p = p.getParent() 

        # assign monitoring segments
        for i in range(len(self._tvs)):
            target = self._tvs[i]
            phi = self.getEntrancePoint(i)
            psi = self.getDeparturePoint(i)
            tf = 2*(switching_to_target[i] + target_to_switching[i])
            
            monitoringParams = MonitoringParameters(phi, psi, tf)            
            self._monitoringSegments.append(MonitoringSegment(target, self.sensor(), monitoringParams))

    def getEntrancePoint(self, idx : int) -> SwitchingPoint:
        return self._epm[idx]
    
    def getDeparturePoint(self, idx : int) -> SwitchingPoint:
        return self._dpm[idx]

    def simulateToSteadyState(self, maxIter = 10) -> None:
        omega0 = {}
        for target in self.world().targets():
            omega0[target] = self._mseTrajectories[target].getEndPoint()
        it = 0
        while True:
            it += 1
            self.simulateCycle()
            
            tol = 1e-3
            steadyState = True
            for target in self.world().targets():
                omegaend = self._mseTrajectories[target].getEndPoint()
                omegadiff = np.linalg.norm(omegaend - omega0[target])
                if omegadiff > tol:
                    steadyState = False
                    break
            if steadyState or it >= maxIter:
                break

    def simulateCycle(self) -> None:
        '''
        In this method, we simulate the entire cycle of the agent. This includes the following steps:
            1) Place the agent at the starting point of the first switching segment, i.e., the departure point of the 0 index target in the visiting sequence
            2) Then repeat the following steps (starting with t0 = 0)
                2.1) Simulate the switching segment and update t0 += switching duration
                2.2) Simulate the monitoring segment and update t0 += monitoring duration

        '''
        assert(len(self._tvs))

        if(len(self._switchingSegments) == 0):
            self.initializeCycle()

        if (len(self._tvs) <= 1):
            return
        
        t0 = self._cycle_start

        for i in range(len(self._tvs)):
            # switch to next target
            self._switchingSegments[i].update()
            self._switchingSegments[i].shiftTime(t0)
            for target in self.world().targets():
                mseTraj, Ik  = SimulateUnmonitoredOmega(self._ucs[target], self._switchingSegments[i].getDuration(), self._mseTrajectories[target].getEndPoint(), 200)
                self._mseTrajectories[target].extend(mseTraj.x, mseTraj.t + t0)
            print("Switching segment from target {} to target {} over period [{},{}]".format(self._tvs[i-1].name, self._tvs[i].name, t0, t0 + self._switchingSegments[i].getDuration()))
            t0 += self._switchingSegments[i].getDuration()

            # monitor target
            self._monitoringSegments[i]._mp._Omega0 = self._mseTrajectories[self._tvs[i]].getEndPoint()
            self._monitoringSegments[i].update()
            self._monitoringSegments[i].shiftTime(t0)
            self._mseTrajectories[self._tvs[i]].extend(self._monitoringSegments[i].mseTrajectory.x, self._monitoringSegments[i].mseTrajectory.t)
            for target in self.world().targets():
                if target == self._tvs[i]:
                    continue
                mseTraj, Ik = SimulateUnmonitoredOmega(self._ucs[target], self._monitoringSegments[i].getDuration(), self._mseTrajectories[target].getEndPoint(), 200)
                self._mseTrajectories[target].extend(mseTraj.x, mseTraj.t + t0)
            print("Monitoring segment of target {} over period [{},{}]".format(self._tvs[i].name, t0, t0 + self._monitoringSegments[i].getDuration()))
            t0 += self._monitoringSegments[i].getDuration()
            
            assert(np.abs(self._switchingSegments[i].pTrajectory.t[-1] - self._monitoringSegments[i].pTrajectory.t[0]) < 1e-6)
            assert(np.abs(self._switchingSegments[i].uTrajectory.t[-1] - self._monitoringSegments[i].uTrajectory.t[0]) < 1e-6)
            assert(np.abs(self._switchingSegments[i].uTrajectory.t[-1] - self._switchingSegments[i].pTrajectory.t[-1]) < 1e-6)
            assert(np.abs(self._monitoringSegments[i].uTrajectory.t[-1] - self._monitoringSegments[i].pTrajectory.t[-1]) < 1e-6)
    
        self._cycle_start = t0

    def plotMSE(self, ax : plt.Axes = plt, **kwargs) -> PlotObject:
        po = PlotObject()
        hasLabel = kwargs.get('label') is not None
        for target in self.world().targets():
            if not hasLabel:
                kwargs['label'] = target.name
            po.add(self._mseTrajectories[target].plot(ax, **kwargs))
        return po
    
    def plotControls(self, ax : plt.Axes = plt, **kwargs) -> PlotObject:
        fig, ax2 = plt.subplots()
        po = PlotObject()
        for ss in self._switchingSegments:
            po.add(ss.uTrajectory.plotStateVsTime(0, ax2, color='blue', linestyle='-', **kwargs))
            po.add(ss.uTrajectory.plotStateVsTime(1, ax2, color='blue', linestyle='--', **kwargs))
            u_norm = np.sqrt(np.square(ss.uTrajectory.x[0,:])+np.square(ss.uTrajectory.x[1,:]))
            po.add(ax.plot(ss.uTrajectory.t, u_norm, color='blue', alpha=0.3, linestyle='-', **kwargs))
        for ms in self._monitoringSegments:
            po.add(ms.uTrajectory.plotStateVsTime(0, ax2, color='red', linestyle='-', **kwargs))
            po.add(ms.uTrajectory.plotStateVsTime(1, ax2, color='red', linestyle='--', **kwargs))
            u_norm = np.sqrt(np.square(ms.uTrajectory.x[0,:])+np.square(ms.uTrajectory.x[1,:]))
            po.add(ax.plot(ms.uTrajectory.t, u_norm, color='red', alpha=0.3, linestyle='-', **kwargs))
            
            tText = np.median(ms.uTrajectory.t)
            uText = np.nanmedian(u_norm) + 0.1
            po.add(ax.text(tText, uText, ms._target.name))
        return po

    def plotCycle(self, ax : plt.Axes, **kwargs) -> PlotObject:
        po = PlotObject()
        po.add(self.plotMonitoringSegments(ax, **kwargs))
        po.add(self.plotSwitchingSegments(ax, **kwargs))
        return po
    
    def plotMonitoringSegments(self, ax : plt.Axes, **kwargs) -> PlotObject:
        po = PlotObject()
        for ms in self._monitoringSegments:
            po.add(ms.pTrajectory.plotStateVsState(0, 1, ax, **kwargs))
        return po
    
    def plotSwitchingPoints(self, ax : plt.Axes, **kwargs) -> PlotObject:
        po = PlotObject()
        for sp in self._switchingPoints:
            po.add(sp.plot(ax, **kwargs))
        return po

    def plotSwitchingSegments(self, ax : plt.Axes, **kwargs) -> PlotObject:
        po = PlotObject()
        for ss in self._switchingSegments:
            po.add(ss.pTrajectory.plotStateVsState(0, 1, ax, **kwargs))
        return po

    def gpp(self) -> GlobalPathPlanner:
        return self._gpp   

    def world(self) -> World:
        return self._world

    def sensor(self) -> Sensor:
        return self._sensor
    