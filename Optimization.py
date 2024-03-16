
import casadi as cad
import numpy as np
from typing import List, Dict

class NLPSolver:
    def __init__(self, prob = None, w0 = None, lbw = None, ubw = None, lbg = None, ubg = None, params = None, quiet = True) -> None:
        self.solver = None
        self.lbw = None
        self.ubw = None
        self.lbg = None
        self.ubg = None
        self.w0 = None
        self.params = None
        self.print_level = 0 if quiet else 1
        self.initialize(prob, w0, lbw, ubw, lbg, ubg)

    def initialize(self, prob = None, w0 = None, lbw = None, ubw = None, lbg = None, ubg = None):
        opts = {'ipopt.print_level': self.print_level, 'print_time': self.print_level}
        self.solver = cad.nlpsol('solver', 'ipopt', prob, opts)
        self.lbw = cad.vertcat(*lbw)
        self.ubw = cad.vertcat(*ubw)
        self.lbg = cad.vertcat(*lbg)
        self.ubg = cad.vertcat(*ubg)
        self.w0 = cad.vertcat(*w0)

    def solve(self):
        return self.solver(x0=self.w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg, p=self.params)

class OptimizationParameters:
    def __init__(self) -> None:
        self._kkt_tolerance : float = 1e-2                                  # KKT tolerance
        self._alpha : float = 1.0                                           # step size for gradient descent    
        self._sigma : float = 1e-1                                          # constraint regularization parameter
        self._beta : float = 0.9                                            # step size reduction factor for gradient descent
        self._tr : float = 0.5                                              # trust region radius
        self._sim_to_steady_state_tol : float = 1e-1                        # tolerance for simulation to steady state
        self._steady_state_iters : int = 10                                 # maximum number of iterations for steady state simulation
        self._optimization_iters : int = 10                                 # maximum number of iterations for optimization