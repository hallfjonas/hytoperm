
import casadi as cad

class NLPSolver:
    def __init__(self, prob = None, w0 = None, lbw = None, ubw = None, lbg = None, ubg = None, params = None):
        self.solver = None
        self.lbw = None
        self.ubw = None
        self.lbg = None
        self.ubg = None
        self.w0 = None
        self.params = None
        self.initialize(prob, w0, lbw, ubw, lbg, ubg, params)

    def initialize(self, prob = None, w0 = None, lbw = None, ubw = None, lbg = None, ubg = None, params = None):
        self.solver = cad.nlpsol('solver', 'ipopt', prob)
        self.lbw = cad.vertcat(*lbw)
        self.ubw = cad.vertcat(*ubw)
        self.lbg = cad.vertcat(*lbg)
        self.ubg = cad.vertcat(*ubg)
        self.w0 = cad.vertcat(*w0)

    def solve(self):
        return self.solver(x0=self.w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg, p=self.params)
