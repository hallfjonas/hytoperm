
# external imports
import casadi as cad
import numpy as np

class NLPSolver:
    def __init__(
            self, 
            prob = None, 
            w0 = None, 
            lbw = None, 
            ubw = None, 
            lbg = None, 
            ubg = None, 
            quiet = True,
            method = 'ipopt'
            ) -> None:
        self.solver = None
        self.lbw = None
        self.ubw = None
        self.lbg = None
        self.ubg = None
        self.w0 = None
        self.params = None
        self.print_level = 0 if quiet else 4
        self.method = method
        self.initialize(prob, w0, lbw, ubw, lbg, ubg)

    def initialize(
            self, 
            prob = None, 
            w0 = None, 
            lbw = None, 
            ubw = None, 
            lbg = None, 
            ubg = None
            ):
        opts = {}
        opts['ipopt.print_level'] = self.print_level 
        opts['print_time'] = self.print_level
        self.solver = cad.nlpsol('solver', self.method, prob, opts)
        self.lbw = cad.vertcat(*lbw)
        self.ubw = cad.vertcat(*ubw)
        self.lbg = cad.vertcat(*lbg)
        self.ubg = cad.vertcat(*ubg)
        self.w0 = cad.vertcat(*w0)

    def solve(self):
        if self.params is None:
            return self.solver(
                x0=self.w0, 
                lbx=self.lbw, 
                ubx=self.ubw, 
                lbg=self.lbg, 
                ubg=self.ubg
                )
        else:
            return self.solver(
                x0=self.w0, 
                lbx=self.lbw, 
                ubx=self.ubw, 
                lbg=self.lbg, 
                ubg=self.ubg, 
                p=self.params
                )

class SubproblemSolver:
    def __init__(self, **kwargs) -> None:
        self.nx = kwargs.get('nx')
        self.ng = kwargs.get('ng')
        self.xk = kwargs.get('x0', np.zeros(self.nx))
        self.nabla_f = kwargs.get('nabla_f')
        self.nabla_g = kwargs.get('nabla_g')
        self.g = kwargs.get('g')
        self.lbx = kwargs.get('lbx', -np.inf * np.ones(self.nx))
        self.ubx = kwargs.get('ubx', np.inf * np.ones(self.nx))
        self.sk = None
        self.yk = None

    def iterate(self) -> None:
        pass

class BFGS(SubproblemSolver):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.Bk = kwargs.get('B0', np.eye(self.nx))

    def iterate(self):
        
        # solve QP
        res = self.solveQP()        

        # update sk, yk
        self.sk = res['x'] - self.xk
        self.lk = res['lam_g']
        self.yk = self.nabla_L(res['x'], self.lk) - self.nabla_L(self.xk, self.lk)
        self.xk = res['x'].full()

        if self.sk.T @ self.yk < 1e-5:
            print("Skipping Bk update")
            return

        # update Bk
        Bksk = self.Bk @ self.sk        
        term_1 = Bksk @ Bksk.T / (self.sk.T @ Bksk)
        term_2 = (self.yk @ self.yk.T) / (self.sk.T @ self.yk)
        self.Bk += -term_1 + term_2

    def nabla_L(self, x, lam_g: np.ndarray):
        return self.nabla_f(x) - self.nabla_g(x) @ lam_g

    def solve(self):
        k = 0
        while True:
            self.iterate()
            if np.linalg.norm(self.sk) < 1e-6:
                break
            k += 1
        return self.xk, self.lk

    def solveQP(self):
        x = cad.SX.sym('x', self.nx)
        dx = x - self.xk
        J = cad.SX(cad.vertcat(*self.nabla_f(self.xk))).T @ dx + 0.5 * dx.T @ self.Bk @ dx
        G = self.g(self.xk) + self.nabla_g(self.xk).T @ dx 
        prob = {'f': J, 'x': x, 'g': G}
        
        qp = NLPSolver(
            prob, 
            self.xk,
            self.lbx,
            self.ubx,
            np.zeros(self.ng),
            np.zeros(self.ng),
            method='ipopt'
        )

        return qp.solve()


class GradientDescent(SubproblemSolver):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.alpha = 0.1

    def iterate(self) -> None:
        self.sk = -self.alpha * self.nabla_f(self.xk)

class OptimizationParameters:
    def __init__(self) -> None:
        self.kkt_tolerance : float = 1e-1                                       # KKT tolerance
        self.alpha : float = 1.0                                                # step size for gradient descent    
        self.sigma : float = 1e-1                                               # constraint regularization parameter
        self.beta : float = 0.95                                                # step size reduction factor for gradient descent
        self.tr : float = 0.5                                                   # trust region radius
        self.sim_to_steady_state_tol : float = 1e-1                             # tolerance for simulation to steady state
        self.steady_state_iters : int = 1                                       # maximum number of iterations for steady state simulation
        self.optimization_iters : int = 100                                     # maximum number of iterations for optimization
        self.subsolver: str = "bfgs"                                            # one of "bfgs" or "gd"

    def copy(self):
        op = OptimizationParameters()
        op.kkt_tolerance = self.kkt_tolerance
        op.alpha = self.alpha
        op.sigma = self.sigma
        op.beta = self.beta
        op.tr = self.tr
        op.sim_to_steady_state_tol = self.sim_to_steady_state_tol
        op.steady_state_iters = self.steady_state_iters
        op.optimization_iters = self.optimization_iters
        return op


'''
cadToNumpy: transform a casadi SX object to a numpy array
'''
def cadToNumpy(x: cad.SX, nrow=None, ncol=None) -> np.ndarray:
    if nrow is None:
        return x.full().flatten()
    if ncol is None:
        return x.full().flatten().reshape(-1,nrow)
    return np.reshape(x.full().flatten(), (nrow, ncol), order='F')
