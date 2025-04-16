from typing import List, Dict
import numpy as np
import math

from .World import *
from .Dynamics import *
from .Sensor import *
from .GlobalPlanning import *
from .PyPlotHelpers import *
from .Optimization import *
from .LinearSensingModel import *
from .PyPlotHelpers import *
from .Statistics import *
_plotAttr = PlotAttributes()

'''
SwitchingPoint: a point in space where the agent switches modes
'''
class SwitchingPoint:
    def __init__(self, p : np.ndarray):
        self._p : np.ndarray = p

    def p(self) -> np.ndarray:
        return self._p

    def plot(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        if kwargs.get('marker') is None:
            kwargs['marker'] = 'o'
        return PlotObject(ax.plot(self._p[0], self._p[1], **kwargs))


'''
LocalParameters: the parameters that define a trajectory segment, i.e., it is
    made up of start and end SwitchingPoints, a duration, and the initial 
    covariance matrix for all targets.
'''
class LocalParameters:
    def __init__(
            self, 
            a_phi: np.ndarray,                                                         # entrance point (polar angle of the boundary point on the first region)
            a_psi: np.ndarray,                                                         # departure point (polar angle of the boundary point on the last region)
            tf: float,                                                          # duration
            Omega0 : Dict[Target, np.ndarray] = {},                             # initial covariance matrices for all targets
            N = 100,                                                            # number of discretization nodes
            rho: float = 0.0,                                                   # slack variable
            ) -> None:
        self._a_phi: np.ndarray = a_phi
        self._a_psi: np.ndarray = a_psi
        self._Omega0 : Dict[Target, np.ndarray] = Omega0
        self._rho : float = rho
        self._tf : float = float(tf)
        self._N = N

    def numberOfParameters(self) -> int:
        """
        Get the number of parameters in the local parameters.
        
        Warning: For now hard-coded to 4 (tf, phi, psi, rho).
        """
        return 4

    def getNumberOfEqualityConstraints(self) -> int:
        """
        Get the number of equality constraints in the local parameters.

        Warning: For now hard-coded to capture
            - tau - delta(bp(phi), bp(psi)) - rho = 0   (1D)
        """
        return 1

    def getLowerBounds(self) -> np.ndarray:
        """
        Get the lower bounds for the local parameters.
        """
        return np.array([-np.inf, -np.inf, -np.inf, 0])

    def toVector(self) -> np.ndarray:
        """
        Get the local parameters in a vector form.
        """
        return np.array([
            float(self._tf),
            float(self._phi),
            float(self._psi),
            float(self._rho)
        ])
    
    def fromVector(self, vec: np.ndarray) -> None:
        """
        Set the local parameters from a vector form.
        """
        self._tf = float(vec[0])
        self._phi = float(vec[1])
        self._psi = float(vec[2])
        self._rho = float(vec[3])

    def getDuration(self) -> float:
        """
        Get the duration.
        """
        return self._tf

    def getStartPoint(self) -> SwitchingPoint:
        """
        Get the initial boundary point.
        """
        return SwitchingPoint(self._a_phi)

    def getEndPoint(self) -> SwitchingPoint:
        """
        Get the terminal boundary point.
        """
        return SwitchingPoint(self._a_psi)
    

"""
LocalMonitoringParameters: the parameters that define a monitoring segment
"""
class LocalMonitoringParameters(LocalParameters):
    def __init__(
            self, 
            r: Region,
            a_phi: np.ndarray,                                                         # entrance point (angle)
            a_psi: np.ndarray,                                                         # departure point (angle)
            tf: float,                                                          # duration
            Omega0 : Dict[Target, np.ndarray] = {},                             # initial covariance matrices for all targets
            N = 100,                                                            # number of discretization nodes
            rho: float = 0.0,                                                   # slack variable
            ) -> None:

        if not isinstance(r, Region):
            raise RuntimeError("Expected input of type Region.")
        
        super().__init__(a_phi, a_psi, tf, Omega0, N, rho)
        self._r = r
        self._phi: float = r.getPolarAngle(a_phi)
        self._psi: float = r.getPolarAngle(a_psi)

    def getRegion(self) -> Region:
        """
        Get the region associated with the segment.
        """
        return self._r

    def hk(self):
        """
        The slacked local inequality constraints.

        Args (all params are assumed to be floats/np.ndarrays or symbolics):
            t_param: the duration of the segment
            a_phi_param: the entrance point of the segment
            a_psi_param: the departure point of the segment
            rho_param: the slack variable
        """

        tau = self.getDuration()
        a_phi = self.getStartPoint().p()
        a_psi = self.getEndPoint().p()
        return tau - self.getRegion().travelCost(a_phi, a_psi)
    
    def hk_slacked(self):
        """
        The slacked local inequality constraints.

        Args (all params are assumed to be floats/np.ndarrays or symbolics):
            t_param: the duration of the segment
            a_phi_param: the entrance point of the segment
            a_psi_param: the departure point of the segment
            rho_param: the slack variable
        """

        return self.hk() - self._rho

    def nabla_hk(self, **kwargs):

        db_dphi = self.getRegion().getBoundaryDerivative(self._phi)
        db_dpsi = self.getRegion().getBoundaryDerivative(self._psi)
        a_phi = self.getStartPoint()
        a_psi = self.getEndPoint()
        dDelta = self.getRegion().travelCostJacobian(a_phi.p(), a_psi.p())
        dDelta_aphi = dDelta[0:2]
        dDelta_apsi = dDelta[2:4]
        return np.array([
            1.0,
            -db_dphi @ dDelta_aphi,
            -db_dpsi @ dDelta_apsi,
            -1.0
        ])


'''
TrajectorySegment: an abstract description of a trajectory segment. 
'''
class TrajectorySegment:
    def __init__(
            self, 
            ucs : Dict[Target, cad.Function],
            params : LocalParameters = None
            ) -> None:
        self.pTrajectory : Trajectory = None                                    # the agent's trajectory for this segment    
        self.uTrajectory : Trajectory = None                                    # the agent's control for this segment    
        self.mseTrajectories : Dict[Target, Trajectory] = {}                    # the estimator's mean squared error trajectory for each target
        self.params : LocalParameters = params                                  # switching parameters for the segment
        self._cost : float = None                                               # the cost of the segment           
        self._gradient_tau = None                                               # the cost gradient with respect to trajectory duration
        self._gradient_a_phi = None                                             # the cost gradient with respect to entrance point      
        self._gradient_a_psi = None                                             # the cost gradient with respect to departure point
        self._ucs : Dict[Target : cad.Function] = ucs                           # an unmmonitored covariance simulator for each target
        self._cov_f : Dict[Target, np.ndarray] = {}

        self.initializeTrajectory()

    def update(self) -> None:
        pass

    def initializeTrajectory(self) -> None:
        ep = self.params.getStartPoint().p()
        dp = self.params.getEndPoint().p()
        tf = self.params.getDuration()
        self.pTrajectory = Trajectory(
            x0 = np.array((ep, dp)).T,
            t0 = np.array((0, tf))
        )

    def getStartPoint(self) -> np.ndarray:
        return self.params._a_phi
    
    def getEndPoint(self) -> np.ndarray:
        return self.params._a_psi
    
    def getStartRegion(self) -> Region:
        return self.params._r[0]
    
    def getEndRegion(self) -> Region:
        return self.params._r[-1]

    def getDuration(self) -> float:
        return self.pTrajectory.t[-1] - self.pTrajectory.t[0]

    def getCost(self) -> float:
        return self._cost
    
    def getGradientTau(self) -> np.ndarray:
        return self._gradient_tau
    
    def getGradientPhi(self, da_dphi) -> np.ndarray:
        return np.dot(self._gradient_a_phi, da_dphi)
    
    def getGradientPsi(self, da_dpsi) -> np.ndarray:
        return np.dot(self._gradient_a_psi, da_dpsi)

    def getTerminalCovarianceMatrices(self) -> Dict[Target, np.ndarray]:
        return self._cov_f
        
    # modifiers
    def updateInitialCovarianceMatrices(
            self, 
            omega0 : Dict[Target, np.ndarray]
            ) -> None:
        self.params._Omega0 = omega0

    def updateTerminalCovarianceMatrix(
            self, 
            target : Target, 
            omega_f : np.ndarray
            ) -> None:
        self._cov_f[target] = omega_f

    def updateMSETrajectory(
            self, 
            target : Target, 
            mseTrajectory : Trajectory
            ) -> None:
        self.mseTrajectories[target] = mseTrajectory

    def shiftTime(self, t0 : float) -> None:
        self.pTrajectory.shiftTime(t0)
        self.uTrajectory.shiftTime(t0)
        for target in self.mseTrajectories.keys():
            self.mseTrajectories[target].shiftTime(t0)

    # plotters
    def plot(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        return self.plotInMissionSpace(ax, **kwargs)

    def plotInMissionSpace(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        return self.pTrajectory.plotStateVsState(0, 1, ax, **kwargs)

    def plotBoundaryPoints(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        po.add(self.params.getStartPoint().plot(ax, **kwargs))
        po.add(self.params.getEndPoint().plot(ax, **kwargs))
        return po


'''
MonitoringController: a class that encapsulates the optimal monitoring control
'''
class MonitoringController:
    def __init__(self, target : Target, sensor : Sensor, N : int = 100) -> None:
        self.solver : NLPSolver = None   
        self.target : Target = target                                           # target to be monitored
        self.N : int = N                                                        # number of control intervals
        self.nx : int = None                                                    # number of states (assigned by builder)
        self.nu : int = None                                                    # number of controls (assigned by builder)
        self.no : int = None                                                    # number of target states (assigned by builder)
        self.add_region_constraints = False                                     # whether or not to include all region boundaries as constraints
        self.buildOptimalMonitoringSolver(target, sensor)

    def buildOptimalMonitoringSolver(
            self, 
            target : Target, 
            sensor : Sensor
            ) -> None:
        
        # input assertions
        region = target.region()
        v = np.zeros(2)
        if hasattr(region, 'dynamics'):
            dynamics : ConstantDynamics = region.dynamics()
            if not isinstance(dynamics, ConstantDynamics):
                    raise Exception("Not implemented for dynamics other than constant dyanmics.")
            v = dynamics.v()

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
        pDot = v + u
        oDot = omegaDot(p, Omega, target, sensor, True)
        xDot = cad.vertcat(pDot, oDot)

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
        F = cad.Function('F',[X0,U,params],[X,Q],['x0','u0','p'],['xf','qf'])

        # Region constraints
        if (self.add_region_constraints):
            region = target.region()
            if isinstance(region, CPRegion):
                g_constr = region.g()
                b_constr = region.b()
                g_constr_term = []
                for i in g_constr.keys():
                    g_constr_term.append(cad.dot(g_constr[i], X0[0:2]) - b_constr[i])
                R = cad.Function('r',[X0],[cad.vertcat(*g_constr_term)],['x0'],['r'])
            elif isinstance(region, SphericalRegion):
                R = cad.Function('r',[X0],[cad.norm_2(X0[0:2]-region.center()) - region.radius()],['x0'],['r'])

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
        
    def optimalMonitoringControl(
            self, 
            params : LocalParameters
            ) -> Tuple[Trajectory,Trajectory,Trajectory,Trajectory,float,float]:
        self.solver.params = cad.vertcat(
            *[params.getStartPoint().p(), 
              params.getEndPoint().p(), 
              np.array(params.getDuration()), 
              np.matrix(params._Omega0[self.target]).getA1()]
        )
        sol = self.solver.solve()

        if not self.solver.solver.stats()['success']:
            raise Exception("Optimal monitoring control failed to converge...")
        w_opt = sol['x'].full()

        # Plot the solution
        nx = self.nx; nu = self.nu; no = self.no; N = self.N 
        p = np.zeros((2, N+1))
        omega = np.zeros((no*no, N+1))
        mse = np.zeros((1, N+1))
        u = np.nan*np.zeros((nu, N))
        p[0,:] = w_opt[0::nx+nu].flatten()
        p[1,:] = w_opt[1::nx+nu].flatten()
        for i in range(no*no):
            omega[i,:] = w_opt[2+i::nx+nu].flatten()
            if i % no == 0:
                mse[0,:] = mse[0,:] + omega[i,:]

        u[0,:] = w_opt[nx::nx+nu].flatten()
        u[1,:] = w_opt[nx+1::nx+nu].flatten()
        t_grid = np.array([params._tf/N*k for k in range(N+1)])

        pTrajectory = Trajectory(p, t_grid)
        mseTrajectory = Trajectory(mse, t_grid)
        omegaTrajectory = Trajectory(omega, t_grid)
        uTrajectory = Trajectory(u, t_grid[0:-1])
        f = sol['f'].full().flatten()[0]
        lam = sol['lam_p']

        # extract trajectories
        return pTrajectory, mseTrajectory, omegaTrajectory, uTrajectory, f, lam


'''
MonitoringSegment: monitoring a single target region
'''
class MonitoringSegment(TrajectorySegment):
    def __init__(
            self, 
            target : Target, 
            sensor : Sensor, 
            ucs : Dict[Target, cad.Function], 
            params : LocalParameters = None
            ) -> None:
        self._target : Target = target
        self._monitoring_controller : MonitoringController = None
        self._monitoring_controller = MonitoringController(
            target=target, 
            sensor=sensor, 
            N=params._N
        )
        super().__init__(ucs, params)

    def update(self):
        mc = self._monitoring_controller
        p, m, omega, u, Jk, duals_p = mc.optimalMonitoringControl(self.params)
        self.pTrajectory = p
        self.updateMSETrajectory(self._target, m)
        self.uTrajectory = u
        self.updateTerminalCovarianceMatrix(self._target, omega.getEndPoint())
        
        self._cost = Jk
        self._gradient_a_phi = -duals_p[0:2].full().flatten()
        self._gradient_a_psi = -duals_p[2:4].full().flatten()
        self._gradient_tau = -duals_p[4].full().flatten()
        for target in self._ucs.keys():
            if target == self._target:
                continue
            mse, Omega, Ik, dIk_dt = simulateUnmonitoredOmega(
                self._ucs[target], 
                self.params.getDuration(), 
                self.params._Omega0[target]
            )
            self.updateMSETrajectory(target, mse)
            self.updateTerminalCovarianceMatrix(target, Omega.getEndPoint())
            self._cost += Ik
            self._gradient_tau += dIk_dt 
            # gradients of phi and psi are equal to 0 here!


'''
SwitchingSegment: switching from one target region to the next
'''
class SwitchingSegment(TrajectorySegment):
    def __init__(
            self, 
            ucs : Dict[Target, cad.Function], 
            gpp : GlobalPathPlanner,
            params : LocalParameters = None
            ) -> None:
        super().__init__(ucs, params)
        self._gpp : GlobalPathPlanner = gpp

    def gpp(self) -> GlobalPathPlanner:
        return self._gpp
    
    def update(self) -> None:
        self._cost = 0
        self._gradient_tau = 0

        # computed by chain rule (first compute the outer value)
        self._gradient_a_psi = 0
        self._gradient_a_phi = 0
        
        for target in self._ucs.keys():
            mse, Omega, Ik, mseEnd = simulateUnmonitoredOmega(
                self._ucs[target], 
                self.params.getDuration(), 
                self.params._Omega0[target]
            )
            self.updateMSETrajectory(target, mse)
            self.updateTerminalCovarianceMatrix(target, Omega.getEndPoint())
            self._cost += Ik
            self._gradient_tau += mseEnd 

            # chain rule (outer)
            self._gradient_a_phi += mseEnd
            self._gradient_a_psi += mseEnd

        # chain rule (inner)
        da_psi, da_phi = self.gpp().getGradientDelta(
            self.getStartPoint(),
            self.getEndPoint()
        )

        # chain rule (outer * inner)
        self._gradient_a_phi = -self._gradient_a_phi * da_phi 
        self._gradient_a_psi = -self._gradient_a_psi * da_psi

        # Don't need to update the control or path, but reset to 0 relative time
        t0 = self.uTrajectory.t[0]
        self.uTrajectory.shiftTime(-t0)
        self.pTrajectory.shiftTime(-t0)


'''
DecomposedCycle: a sequence of the decomposition based trajectory segments.
'''
class DecomposedCycle:
    def __init__(
            self,
            t0 : float = 0.0, 
            counter : int = 0
            ) -> None:
        self.pTrajectory : Trajectory = None
        self.uTrajectory : Trajectory = None
        self.mseTrajectories : Dict[Target, Trajectory] = {}
        self._trajectorySegments : List[TrajectorySegment] = []
        self._monitoringSegments: List[MonitoringSegment] = []
        self._switchingSegments: List[SwitchingSegment] = []
        self._covAtCycleStart : Dict[Target, np.ndarray] = {}
        self._covAtCycleEnd : Dict[Target, np.ndarray] = None
        
        # statistics
        self._cycle_start : float = t0
        self._counter : int = counter

        # plot controls
        self._switchColor = 'blue'
        self._monitorColor = 'red'

    def assignCycle(
            self, 
            ms : List[MonitoringSegment], 
            omega0: Dict[Target, np.ndarray]
            ) -> None:
        
        ss = self._switchingSegments
        if ss is None:
            raise Exception("Switching segments must be initialized beforehand.")
        if len(ms) != len(ss):
            raise Exception("Monitoring and switching segments must have the same length.")

        # assign the initial covariance matrices        
        self._covAtCycleStart = omega0
        self._monitoringSegments = ms

        # combine the segments (monitoring, switching, monitoring, ...)
        self._trajectorySegments.clear()
        for i in range(self.K()):
            self._trajectorySegments.append(ms[i])
            self._trajectorySegments.append(ss[i])
            
        # ensure all is well
        self.checkCycle()

    def replaceSwitchingSegments(self, ss : List[SwitchingSegment]) -> None:
        if len(ss) != self.K():
            raise Exception("Expected a single switching segment.")
        self._switchingSegments = ss
        self._trajectorySegments[1::2] = ss
        self.checkCycle()

    def checkCycle(self) -> None:
        for k, ts in enumerate(self._trajectorySegments):
            old_seg = self._trajectorySegments[k-1]
            new_seg = self._trajectorySegments[k]

            if k % 2 == 0:
                if not isinstance(ts, MonitoringSegment):
                    raise Exception("Assumed to begin with monitoring segment, and end with switching segment.")
            else:
                if not isinstance(ts, SwitchingSegment):
                    raise Exception("Assumed to begin with monitoring segment, and end with switching segment.")
            
            if np.linalg.norm(old_seg.getEndPoint() - new_seg.getStartPoint()) != 0:
                raise Exception("Subsequent trajectories don't connect.")
            
    # getters
    def getDuration(self) -> float:
        return sum([ts.getDuration() for ts in self._trajectorySegments])

    def K(self) -> int:
        return len(self._switchingSegments)

    def getStartTime(self) -> float:
        return self._cycle_start
    
    def getEndTime(self) -> float:
        return self._cycle_start + self.getDuration()

    def getCost(self) -> float:
        return sum(ts.getCost() for ts in self._trajectorySegments)
    
    def getGradientTau(self) -> np.ndarray:
        monSeg : List[TrajectorySegment] = []
        for ts in self._trajectorySegments:
            if isinstance(ts, MonitoringSegment):
                monSeg.append(ts)
        return np.array([ts.getGradientTau() for ts in monSeg]).flatten()

    def getGradientSwitches(self) -> np.ndarray:
        grad_phi = np.zeros(self.K())
        grad_psi = np.zeros(self.K())
        for k in range(self.K()):
            ss_old = self._switchingSegments[k-1]
            ms = self._monitoringSegments[k]
            ss_next = self._switchingSegments[k]
            msp: LocalMonitoringParameters = ms.params
            da_dphi = msp.getRegion().getBoundaryDerivative(msp._phi)
            da_dpsi = msp.getRegion().getBoundaryDerivative(msp._psi)
            grad_phi[k] = ss_old.getGradientPsi(da_dphi) + ms.getGradientPhi(da_dphi)
            grad_psi[k] = ms.getGradientPsi(da_dpsi) + ss_next.getGradientPhi(da_dpsi)
        return grad_phi, grad_psi

    def getCycleCostGradients(self) -> Dict[str, np.ndarray]:
        grad_phi, grad_psi = self.getGradientSwitches()
        return {
            'tau': self.getGradientTau(),
            'phi': grad_phi,
            'psi': grad_psi
        }
        
    def getInitialCovarianceMatrices(self) -> Dict[Target, np.ndarray]:
        return self._covAtCycleStart
    
    def getTerminalCovarianceMatrices(self) -> Dict[Target, np.ndarray]:
        return self._covAtCycleEnd.copy()
    
    def steadyState(self, tol=1e-2) -> bool:
        for target in self._covAtCycleStart.keys():
            if self._covAtCycleEnd is None:
                return False
            
            if not np.allclose(
                self._covAtCycleStart[target], 
                self._covAtCycleEnd[target], 
                atol=tol):
                return False
            
        return True
    
    # modifiers
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

    def updateInitialCovarianceMatrices(
            self, 
            omega0 : Dict[Target, np.ndarray]
            ) -> None:
        self._covAtCycleStart = omega0.copy()
    
    def shiftTime(self, deltaT : float = None) -> None:
        if deltaT is None:
            deltaT = -self.pTrajectory.t[0]
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
                self.mseTrajectories[target] = Trajectory(
                    ts.mseTrajectories[target].x, 
                    ts.mseTrajectories[target].t
                )
        else:
            self.pTrajectory.appendTrajectory(ts.pTrajectory)
            self.uTrajectory.appendTrajectory(ts.uTrajectory)
        
            for target in ts.mseTrajectories.keys():
                self.mseTrajectories[target].appendTrajectory(
                    ts.mseTrajectories[target]
                )
                
                L = self.mseTrajectories[target].t
                for x, y in zip(L, L[1:]):
                    if x > y + 1e-6:
                        warnings.warn("Time vector is not strictly increasing.")

    # plotters
    def plot(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        eka = extendKeywordArgs(_plotAttr.agent.getAttributes(), **kwargs)
        return self.pTrajectory.plotStateVsState(0, 1, ax, **eka)
    
    def plotControls(
            self, 
            ax : plt.Axes = None, 
            add_monitoring_labels=True, 
            **kwargs
            ) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        po.add(self.plotMonitoringControls(
            ax, 
            add_monitoring_labels=add_monitoring_labels, 
            **kwargs)
        )
        po.add(self.plotSwitchingControls(ax, **kwargs))
        return po
    
    def plotSwitchingControls(
            self, 
            ax : plt.Axes = None, 
            **kwargs
            ) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        u1_pA = _plotAttr.u1_switch
        u2_pA = _plotAttr.u2_switch
        un_pA = _plotAttr.u_norm_switch
        eka1 = extendKeywordArgs(u1_pA.getAttributes(), **kwargs)
        eka2 = extendKeywordArgs(u2_pA.getAttributes(), **kwargs)
        eka3 = extendKeywordArgs(un_pA.getAttributes(), **kwargs)
        for ts in self._trajectorySegments:
            if not isinstance(ts, SwitchingSegment):
                continue
            u1 = ts.uTrajectory.x[0,:]
            u2 = ts.uTrajectory.x[1,:]
            u_norm = np.sqrt(np.square(u1)+np.square(u2))
            po.add(ax.plot(ts.uTrajectory.t, u_norm, **eka3))
            po.add(ts.uTrajectory.plotStateVsTime(0, ax, **eka1))
            po.add(ts.uTrajectory.plotStateVsTime(1, ax, **eka2))
        return po
    
    def plotMonitoringControls(
            self, 
            ax : plt.Axes, 
            add_monitoring_labels=True, 
            **kwargs
            ) -> PlotObject:
        po = PlotObject()
        u1_pA = _plotAttr.u1_monitor
        u2_pA = _plotAttr.u2_monitor
        un_pA = _plotAttr.u_norm_monitor
        eka1 = extendKeywordArgs(u1_pA.getAttributes(), **kwargs)
        eka2 = extendKeywordArgs(u2_pA.getAttributes(), **kwargs)
        eka3 = extendKeywordArgs(un_pA.getAttributes(), **kwargs)
        for ts in self._trajectorySegments:
            if not isinstance(ts, MonitoringSegment):
                continue

            if ts == self._trajectorySegments[-1]:
                eka1['label'] = '$u_1$'
                eka2['label'] = '$u_2$'
                eka3['label'] = '$\|u\|$'

            u1 = ts.uTrajectory.x[0,:]
            u2 = ts.uTrajectory.x[1,:]
            u_norm = np.sqrt(np.square(u1)+np.square(u2))
            po.add(ax.plot(ts.uTrajectory.t, u_norm, **eka3))
            po.add(ts.uTrajectory.plotStateVsTime(0, ax, **eka1))
            po.add(ts.uTrajectory.plotStateVsTime(1, ax, **eka2))
   
            if add_monitoring_labels:
                tText = np.median(ts.uTrajectory.t)
                uText = np.nanmedian(u_norm) + 0.1
                po.add(ax.text(tText, uText, ts._target.name))

        return po

    def plotTargetMSE(
            self, 
            target : Target,
            add_label : bool = False,
            ax : plt.Axes = None, 
            **kwargs
            ) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        eka = kwargs.copy()
        if add_label:
            eka['label'] = target.name
        ext = extendKeywordArgs(eka, **kwargs)
        po.add(self.mseTrajectories[target].plotStateVsTime(0, ax, **ext))
        mseStart = self.mseTrajectories[target].getInitialValue() 
        po.add(ax.hlines(
            mseStart, 
            self._cycle_start, 
            self._cycle_start + self.getDuration(), 
            alpha=0.2, 
            **ext
            ))
        return po

    def plotMSE(
            self, 
            add_labels=False, 
            ax : plt.Axes = None, 
            **kwargs
            ) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        i = 0
        for target in self.mseTrajectories.keys():
            ext = {'color': _plotAttr.target_colors[-int(target.name)+1]}
            eka = extendKeywordArgs(ext, **kwargs)
            po.add(self.plotTargetMSE(target, add_labels, ax, **eka))
            i += 1
        return po


"""
Decomposition: the parameters that define a decomposition of the agent
    trajectory into a sequence of local trajectories.
"""
class Decomposition:
    def __init__(self, **kwargs):
        """
        Initialize the decomposition parameters.
        
        Args:
            gpp (GlobalPathPlanner, optional): the global path planner
            tvs (list, optional): the target visiting sequence
            local_params (Dict, optional): the local parameters associated with 
                each monitoring segment
        """
        self._world: World = kwargs.get('world', None)                          # The world
        self._sensor: Sensor = kwargs.get('sensor', None)                       # The sensor
        self._gpp: GlobalPathPlanner = kwargs.get('gpp', None)                  # Global path planner
        self._tvs: List[Target] = kwargs.get('tvs', [])                         # target visiting sequence
        self._dJ: Dict[str, np.ndarray] = {}                                    # Derivatives of the cost function
        self._cycle: DecomposedCycle = kwargs.get('cycle', None)                # The decomposed cycle
        self._N: int = kwargs.get('N', 100)                                     # Number of control intervals
      
        # simulation functions for forward integration of covariance matrices
        self._ucs : Dict[Target, cad.Function] = {}                             # unmonitored covariance simulators

        # optimization parameters
        self._subSolver: SubproblemSolver = None                                # SubproblemSolver for the bilevel optimization
        self._lambda : Dict[int, float] = {}                                    # map a target visit index to a dual of the kth bilevel optimization constraint
        self.op : OptimizationParameters = OptimizationParameters()             # optimization parameters

    def initialize(self) -> None:

        if (self.K() <= 1):
            warnings.warn("Expected at least two targets in the visiting sequence. Did you run 'computeVisitingSequence()'?")
            return

        # for each target build an unmonitored covariance simulator
        for target in self.world().targets():
            self._ucs[target] = unmonitoredOmegaSimulator(
                target, 
                self.sensor(), 
                self._N
            )

        # initialize the switching segments (and potentially set refined tvs)
        refined_tvs = self.initializeSwitchingSegments()
        self.setTargetVisitingSequence(refined_tvs)
        
        # initialize monitoring and switching segments
        mns = self.initializeMonitoringSegments()
        
        # assign cycle
        self._cycle.assignCycle(mns, self.initialCovarianceMatrices())

        # Gradient Descent Subsolver
        if self.op.subsolver == 'gd':
            self._subSolver = GradientDescent(
                nx=self.getNumberParameters(),
                ng=self.getNumberEqualityConstraints(),
                x0=self.toVector(),
                nabla_f=lambda x: self.nabla_f(),
                nabla_g=lambda x: self.jacobian_h_slacked().T,
                g=lambda x: self.h_slacked(),
                lbx=self.getBFGSLowerBounds(),
                alpha=self.op.alpha
            )

        elif self.op.subsolver == 'bfgs':
            self._subSolver = BFGS(
                nx=self.getNumberParameters(),
                ng=self.getNumberEqualityConstraints(),
                x0=self.toVector(),
                nabla_f=lambda x: self.nabla_f(),
                nabla_g=lambda x: self.jacobian_h_slacked().T,
                g=lambda x: self.h_slacked(),
                lbx=self.getBFGSLowerBounds()
            )

        else:
            raise ValueError(f"Unknown subsolver type {self.op.subsolver}.")

    def getBFGSLowerBounds(self) -> np.ndarray:
        lbs = []
        for k in range(self.K()):
            lbs.extend(self.lpr(k).getLowerBounds())
        return lbs

    def updateParameters(
            self, 
            op: OptimizationParameters, 
            stats: IterationStats = None
            ) -> None:
        '''
        Simple projected gradient descend
        '''

        steady, ssc = self.simulateToSteadyState(op, stats)
        
        if isinstance(stats, IterationStats):
            stats.is_steady_state.append(steady)
            stats.steady_state_iterations.append(ssc)
        
        self.updateGlobalCostGradients(stats)

        if not self.checkFeasibility():
            raise RuntimeError("Infeasible initialization.")       

        # update the parameters
        xold = self.toVector()

        self._subSolver.alpha = 0.1
        self._subSolver.iterate()
        sk = self._subSolver.sk
        
        alpha = 1.0
        beta = 0.9
        while True:
            self.fromVector(xold + alpha * sk)
            if self.checkFeasibility():
                break
            alpha *= beta

            if alpha < 1e-6:
                raise RuntimeError("Backtracking failed (step size too small.)")
        
        swPaths = []
        for k in range(self.K()):
            apsi = self.lpr(k).getEndPoint().p()
            aphi = self.lpr(k+1).getStartPoint().p()
            swPaths.append(self.gpp().planPath(apsi, aphi)[0])
        refined_tvs, sw = self.refineVisitingSequence(swPaths)
        self._cycle.replaceSwitchingSegments(sw)
        
        for tr, to in zip(refined_tvs, self._tvs):
            if tr != to:
                raise RuntimeError("Changes in the visiting sequence is not yet supported.")

        # store the parameters
        self._storeParameters(stats)
        stats.alphas.append(alpha)

    def checkFeasibility(self) -> bool:
        for k in range(self.K()):
            lp = self.lpr(k)
            hkv = lp.hk()
            if hkv < -1e-6:
                return False
        return True

    def lpr(self, k: int) -> LocalMonitoringParameters:
        return self.monitoringSegment(k).params

    def updateSwitchingSegments(self) -> None:
        for k in range(self.K()):
            a_psik = self.lpr(k).getEndPoint().p()
            a_phikn = self.lpr(k+1).getEndPoint().p()
            path, tf = self.gpp().planPath(a_psik, a_phikn)
            sw, nt, _ = self.extractSwitchingSegment(
                path, 
                self.getTarget(k)
            )
            self._switchingSegments[k].update()

    def setTargetVisitingSequence(self, tvs: List[Target]) -> None:
        """
        Set the target visiting sequence.

        Args:
            tvs (List[Target]): the target visiting sequence
        """
        self._tvs = tvs

    def initializeSwitchingSegments(self) -> None:
        
        swPaths: List[Tree] = []
        for k in range(self.K()):
            ot = self.getTarget(k)
            nt = self.getTarget(k+1)
            sp, _ = self.gpp().getSwitchingPath(ot, nt)
            swPaths.append(sp)

        tvs, sws = self.refineVisitingSequence(swPaths)
        self._cycle = DecomposedCycle()
        self._cycle._switchingSegments = sws
        return tvs

    def initialCovarianceMatrices(self) -> Dict[Target, np.ndarray]:
        Omegas = {}
        for target in self.world().targets():
            Omegas[target] = np.eye((target.getNumberOfStates()))
        return Omegas

    def K(self) -> int:
        return len(self._tvs)
        
    def gpp(self) -> GlobalPathPlanner:
        return self._gpp

    def world(self) -> World:
        return self._world

    def sensor(self) -> Sensor:
        return self._sensor

    def getParameters(self) -> List[LocalMonitoringParameters]:
        """
        Get the local parameters of the decomposition.
        """
        return [self.lpr(k) for k in range(self.K())]

    def getNumberParameters(self) -> int:
        """
        Get the number of parameters in the decomposition.
        """
        return sum(lp.numberOfParameters() for lp in self.getParameters())

    def getNumberEqualityConstraints(self) -> int:
        """
        Get the number of equality constraints in the decomposition.
        """
        return sum(lp.getNumberOfEqualityConstraints() for lp in self.getParameters())
    
    def toVector(self) -> np.ndarray:
        """
        Get the decomposition parameters in a vector form.
        """

        # initialize the vector
        vec = np.zeros(self.getNumberParameters())
        idx = 0
        for k in range(self.K()):
            lp = self.lpr(k)
            vec[idx:idx+lp.numberOfParameters()] = lp.toVector()
            idx += lp.numberOfParameters()
        return vec
    
    def fromVector(self, vec: np.ndarray) -> None:
        """
        Set the decomposition parameters from a vector form.
        """
        idx = 0
        for k in range(self.K()):
            np = self.lpr(k).numberOfParameters()
            pk = vec[idx:idx+np]
            self.lpr(k).fromVector(pk)
            idx += np

    def getIndex(self, key: str, idx: int) -> int:
        """
        Get the index of the local parameters in the vector form.

        Args:
            key (str): the type of parameter to get the index for
            idx (int): the index of the local parameters
        """
        idx_offset = sum(self.lpr(k).numberOfParameters() for k in range(idx))
        if key == 'tau':
            return idx_offset
        elif key == 'phi':
            return idx_offset + 1
        elif key == 'psi':
            return idx_offset + 2
        elif key == 'rho':
            return idx_offset + 3
        
        raise Exception(f"Invalid key {key} (expected to be one of 'tau', 'phi', 'psi', or 'rho').")

    def getTauLowerBounds(self) -> np.ndarray:
        """
        Get the lower bounds
        """
        lbs = []
        for k in range(self.K()):
            tf = self.getTarget(k).region().travelCost(
                self.lpr(k).getStartPoint().p(),
                self.lpr(k).getEndPoint().p()
            )
            lbs.append(tf)
        return lbs

    def getTauVec(self) -> np.ndarray:
        """
        Get the vector of durations.
        """
        return np.array([self.lpr(i)._tf for i in range(self.K())])

    def getPhiVec(self) -> np.ndarray:
        """
        Get the vector of entrance point polar angles.
        """
        return np.array([self.lpr(i)._phi for i in range(self.K())])

    def getPsiVec(self) -> np.ndarray:
        """
        Get the vector of departure point polar angles.
        """
        return np.array([self.lpr(i)._psi for i in range(self.K())])
    
    def nabla_f(self, **kwargs) -> np.ndarray:
        nf = np.zeros(self.getNumberParameters())
        for key in self._dJ.keys():
            for k in range(self.K()):
                nf[self.getIndex(key, k)] = self._dJ[key][k]
        return nf

    def h(self, **kwargs) -> np.ndarray:
        """
        Combine the slacked inequality constraints.
        """
        h = []
        for k in range(self.K()):
            hk = self.lpr(k).hk()
            h.append(hk)
        return np.array(h).flatten()

    def h_slacked(self, **kwargs) -> np.ndarray:
        """
        Combine the slacked inequality constraints.
        """
        h = []
        for k in range(self.K()):
            hk = self.lpr(k).hk_slacked()
            h.append(hk)
        return np.array(h).flatten()

    def jacobian_h_slacked(self, **kwargs) -> np.ndarray:
        nx = self.getNumberParameters()
        nh = self.getNumberEqualityConstraints()
        jac_h = np.zeros((nh, nx))
        for k in range(self.K()):
            grad_hk = self.lpr(k).nabla_hk()
            tau_idx = self.getIndex('tau', k)
            phi_idx = self.getIndex('phi', k)
            psi_idx = self.getIndex('psi', k)
            rho_idx = self.getIndex('rho', k)
            jac_h[k, tau_idx] = grad_hk[0]
            jac_h[k, phi_idx] = grad_hk[1]
            jac_h[k, psi_idx] = grad_hk[2]
            jac_h[k, rho_idx] = grad_hk[3]
        return jac_h

    def refineVisitingSequence(
            self,
            swPaths: List[Tree]
            ) -> Tuple[List[Target], List[SwitchingSegment]]:
        """
        Refine the visiting sequence:
            Given a sequence of paths swPaths, where swPaths[k] connects the 
            target regions from getTarget(k) to getTarget(k+1), we extract all 
            the additional switching segments and refine the visiting sequence.

            A new switching segment is added for each path in swPaths which 
            traverses an additional target region along its path. All such 
            additional visits are included to the revised visiting sequence.
            The return list of switching segments is the list of all new switching
            segments that were added plus all original segments that did not 
            pass through additional regions.

        Returns:
            Tuple[List[Target], List[SwitchingSegment]]: the refined visiting 
                sequence and the new switching segments
        """
        segments : List[SwitchingSegment] = []
        refined_tvs = [self.getTarget(0)]

        for i in range(self.K()):
            target = self.getTarget(i)
            swPath = swPaths[i]
            next_target = self.getTarget(i+1)
            while True:
                swSeg, target, swPath = self.extractSwitchingSegment(swPath,target)

                if swSeg is None:
                    raise Exception("Expected a switching segment.")

                segments.append(swSeg)
                refined_tvs.append(target)

                if target == next_target:
                    break
        
        return refined_tvs[0:-1], segments

    def getTarget(self, k: int) -> Target:
        return self._tvs[k % self.K()]

    def extractSwitchingSegment(
            self, 
            path : Tree,
            initialTarget: Target
            ) -> Tuple[SwitchingSegment, Target, Tree]:
        '''
        Move up the tree and extract the switching points until a target region 
        is reached. We then return the switching segment together with the 
        remaining tree beginning from the first node where the active region is
        not the reached target region.
        '''
        
        node : Tree = path
        if node is None:
            warnings.warn("Path is empty. Returning None")
            return None, None, None
        
        tf = np.array(0).reshape(1)
        u = np.nan*np.zeros((2,1))
        ep = SwitchingPoint(node.getData().p())
        dp = ep
        a_phi = ep.p()
        pTrajectory = Trajectory(ep.p().reshape(2,1), tf)
        uTrajectory = Trajectory(u, tf)

        while node is not None:

            # check if we have reached a target region
            for target in self.world().targets():
                if target == initialTarget:
                    continue
                if target.region().contains(node.getData().p(), tol=1e-3):
                    
                    sp = LocalParameters(
                        a_phi = ep.p(),
                        a_psi = dp.p(),
                        tf = tf,
                        N = self._N
                    )

                    ts = SwitchingSegment(
                        self._ucs, 
                        self.gpp(), 
                        sp
                    )

                    ts.pTrajectory = pTrajectory
                    ts.uTrajectory = uTrajectory
                    return ts, target, node.getParent()

            if node.isRoot():
                return None, None, None

            # otherwise add the next switching point
            a_psi = node.getParent().getData().p()
            dp = SwitchingPoint(a_psi)
            deltaT = node.getData().costToParent()
            
            # TODO(Jonas): Hacky solution currently in place
            # What I need here is the control from one node to the next
            # In my current setting (constant Dynamics on the regions, this is 
            # a constant control law). In general doesn't need to be... 
            # So really, should have a trajectory to parent stored in the node 
            # or even better in an edge between the two nodes...
            artp = node.getData().activeRegionToParent()
            v = np.zeros(2)
            if hasattr(artp, 'dynamics'):
                if isinstance(artp.dynamics(), ConstantDynamics):
                    v = artp.dynamics().v()
            
            u = (a_psi-a_phi)/deltaT - v
            
            # update trajectories
            uTrajectory.extend(u, tf)
            uTrajectory.extend(u, tf + deltaT)
            pTrajectory.extend(a_psi, tf + deltaT)

            # update time
            tf = tf + deltaT
            a_phi = a_psi

            # get next node
            node = node.getParent()

        return None, None, None

    def switchingSegment(self, k: int) -> SwitchingSegment:
        return self._cycle._switchingSegments[k % self.K()]
    
    def monitoringSegment(self, k: int) -> MonitoringSegment:
        return self._cycle._monitoringSegments[k % self.K()]

    def initializeMonitoringSegments(self) -> List[MonitoringSegment]:
        
        if self._cycle._switchingSegments is None:
            raise Exception("Expected switching segments to be initialized.")
        
        segments : List[MonitoringSegment] = []
        for i in range(self.K()):
            target = self.getTarget(i)
            a_phi = self.switchingSegment(i-1).getEndPoint()
            a_psi = self.switchingSegment(i).getStartPoint()
            min_t = target.region().travelCost(a_phi, a_psi)
            tf = max(0.1, 1.5*min_t)
            
            params = LocalMonitoringParameters(
                r = target.region(),
                a_phi = a_phi,
                a_psi = a_psi,
                tf = tf,
                N = self._N,
                rho = tf - min_t
            )

            segment = MonitoringSegment(
                target,
                self.sensor(),
                self._ucs,
                params
            )
            
            segments.append(segment)

        return segments
    
    def simulateToSteadyState(
            self, 
            op: OptimizationParameters,
            stats: IterationStats = None
            ) -> Tuple[bool, int]:
        it = 0
        cycle = self._cycle
        omega_f = cycle.getInitialCovarianceMatrices()
        while True:
            it += 1

            cycle.simulate()
            omega_f = cycle.getTerminalCovarianceMatrices()

            cycle_average_cost = cycle.getCost()/cycle.getDuration()
            if isinstance(stats, IterationStats):
                stats.global_costs.append(cycle_average_cost)

            isSteady = cycle.steadyState(tol=op.sim_to_steady_state_tol)
            cycle.updateInitialCovarianceMatrices(omega_f)

            if isSteady:
                return True, it

            if it >= op.steady_state_iters:
                return False, it
            
            cycle._cycle_start += cycle.getDuration()

    def getGradientCycleCost(self) -> Dict[str, np.ndarray]:
        """
        Compute the gradient of the cycle cost, i.e., of the cost of one 
        complete cycle (not scaled with the cycle duration).
        """
        d_a_phi, d_a_psi = self._cycle.getGradientSwitches()
        return {
            'tau': self._cycle.getGradientTau(),
            'phi': d_a_phi,
            'psi': d_a_psi
        }

    def getGradientT(self) -> Dict[str, np.ndarray]:
        """
        Get the gradient of the cycle duration with respect to all parameters.
        """

        nablaDelta = {
            'phi': np.zeros(self.K()),
            'psi': np.zeros(self.K())
        }
                    
        k = 0
        for k in range(self.K()):
            msp = self.lpr(k) 
            ts = self._cycle._switchingSegments[k]
            msn = self.lpr(k+1)

            # get the gradient of the switching duration wrt start & end
            a = ts.getStartPoint()
            b = ts.getEndPoint()
            da, db = self.gpp().getGradientDelta(a, b) 

            # get the gradient of the switching points wrt polar angles
            da_dangle = msp.getRegion().getBoundaryDerivative(msp._psi)
            db_dangle = msn.getRegion().getBoundaryDerivative(msn._phi)
            
            # chain rule
            nablaDelta['psi'][k] = np.dot(da,da_dangle)
            nablaDelta['phi'][(k+1) % self.K()] = np.dot(db, db_dangle)
            k += 1

        return {
            'tau': np.ones(self.K()),
            'phi': nablaDelta['phi'],
            'psi': nablaDelta['psi']
        }

    def getGlobalGradientNorm(self) -> float:
        return math.sqrt(sum(np.linalg.norm(nJ)**2 for nJ in self._dJ.values()))

    def updateGlobalCostGradients(
        self,
        stats: IterationStats = None
        ) -> Dict[str, np.ndarray]:
        """
        Computes the global cost gradient with respect to all parameters
        """
        T = self._cycle.getDuration()
        C = self._cycle.getCost()

        nablaT = self.getGradientT()
        nablaC = self.getGradientCycleCost()

        nablaJ = {}
        for param in nablaT.keys():
            nablaJ[param] = (nablaC[param] * T - nablaT[param] * C)/T**2
        
        # global average cost gradient
        self._dJ = nablaJ

        if isinstance(stats, IterationStats):
            stats.global_gradients.append(nablaJ)
            stats.global_gradient_norms.append(self.getGlobalGradientNorm())

    def _storeParameters(self, stats: IterationStats) -> None:
        if isinstance(stats, IterationStats):
            stats.tau_values.append(self.getTauVec())
            stats.phi_values.append(self.getPsiVec())
            stats.psi_values.append(self.getPhiVec())
            stats.iterate += 1

    def plotMonitoringSegments(
            self,
            ax : plt.Axes = None, 
            **kwargs
            ) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        for ms in self._cycle._monitoringSegments:
            po.add(ms.pTrajectory.plotStateVsState(0, 1, ax, **kwargs))
        return po
    
    def plotEntryPoints(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        eka = extendKeywordArgs(_plotAttr.phi.getAttributes(), **kwargs)
        for ms in self._cycle._monitoringSegments:
            po.add(ms.params.getStartPoint().plot(ax, **eka))
        return po
    
    def plotDeparturePoints(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        eka = extendKeywordArgs(_plotAttr.psi.getAttributes(), **kwargs)
        for ms in self._cycle._monitoringSegments:
            po.add(ms.params.getEndPoint().plot(ax, **eka))
        return po

    def plotSwitchingAngles(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        phi = self.getPhiVec()
        psi = self.getPsiVec()
        for k in range(self.K()):
            ext = {'color': _plotAttr.target_colors[-int(self.getTarget(k).name)+1]}
            eka = extendKeywordArgs(ext, **kwargs)
            ax.plot([np.cos(phi[k])], [np.sin(phi[k])], **eka, marker = 'x')
            ax.plot([np.cos(psi[k])], [np.sin(psi[k])], **eka, marker = 'o')
        return

    def plotSwitchingTimes(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()

        xlabels = [f'$\\tau_{k}$' for k in range(self.K())]
        values = self.getTauVec()
        lbs = self.getTauLowerBounds()
        hat_graph(ax, xlabels, [lbs, values], ['Lower bound', 'Value'])
        return

    def plotSwitchingPoints(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        for k in range(self.K()):
            eka = extendKeywordArgs(_plotAttr.phi.getAttributes(),**kwargs)
            po.add(self.lpr(k).getStartPoint().plot(ax, **eka))
            
            eka = extendKeywordArgs(_plotAttr.psi.getAttributes(),**kwargs)
            po.add(self.lpr(k).getEndPoint().plot(ax, **eka))
        return po

    def plotSwitchingSegments(
            self, 
            ax : plt.Axes = None, 
            **kwargs
            ) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        for ss in self._cycle._switchingSegments:
            po.add(ss.pTrajectory.plotStateVsState(0, 1, ax, **kwargs))
        return po


def hat_graph(ax, xlabels, values, group_labels):
    """
    Create a hat graph.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes to plot into.
    xlabels : list of str
        The category names to be displayed on the x-axis.
    values : (M, N) array-like
        The data values.
        Rows are the groups (len(group_labels) == M).
        Columns are the categories (len(xlabels) == N).
    group_labels : list of str
        The group labels displayed in the legend.
    """

    def label_bars(heights, rects):
        """Attach a text label on top of each bar."""
        for height, rect in zip(heights, rects):
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  # 4 points vertical offset.
                        textcoords='offset points',
                        ha='center', va='bottom')

    values = np.asarray(values)
    x = np.arange(values.shape[1])
    ax.set_xticks(x, labels=xlabels)
    spacing = 0.3  # spacing between hat groups
    width = (1 - spacing) / values.shape[0]
    heights0 = values[0]
    for i, (heights, group_label) in enumerate(zip(values, group_labels)):
        style = {'fill': False} if i == 0 else {'edgecolor': 'black'}
        rects = ax.bar(x - spacing/2 + i * width, heights - heights0,
                       width, bottom=heights0, label=group_label, **style)
        label_bars(heights, rects)