
from .Experiment import Experiment
from .Optimization import OptimizationParameters
from .Statistics import IterationStats
import warnings
from typing import List

import numpy as np

class BTO:
    def __init__(self, **kwargs) -> None:
        self._ex: Experiment = kwargs.get('ex')
        if not isinstance(self._ex, Experiment):
            raise ValueError("Please pass an Experiment ('ex') argument.")
        
        self._op: OptimizationParameters = kwargs.get('op', OptimizationParameters())
        self._stats: IterationStats = kwargs.get('stats', IterationStats())
        self._log_file: str = kwargs.get('log_file', 'log.txt')
        self._ignore_odd: List[str] = ['phi', 'psi']
        self._ignore_even: List[str] = ['tau']

    def solve(self) -> None:
        '''
        Solve the optimization problem
        '''
        self.printHeader('w')
        for i in range(self._op.optimization_iters):
            if isinstance(self._stats, IterationStats):
                self._stats.iterate = i
                
            self.updateParameters()
            self.printIteration()     
            
            if self._stats.global_gradient_norms[-1] < self._op.kkt_tolerance:
                print("Convergence tolerance reached...")
                break     
                    
    def updateParameters(self) -> None:
        '''
        Simple projected gradient descend
        '''
        ag = self._ex.agent()
        dc = ag.decomposition
        stats = self._stats
        op = self._op

        ag.simulateToSteadyState(op, stats)
        
        # compute direction
        dc.updateGlobalCostGradients(stats)

        # ignore whatever we want to ignore
        for ig in dc._dJ.keys():
            ignore = self._ignore_even if stats.iterate % 2 == 0 else self._ignore_odd
            if ig in ignore:
                dc._dJ[ig] *= 0.0

        if stats.global_gradient_norms[-1] > 1:
            print("Large norms detected...")
            raise ValueError("Gradient norm is too large, please check your model.")

        # update parameters
        xold = dc.toVector()
        dc.fromVector(xold - op.alpha * dc.nabla_f())

        # emergency mode
        # when seeing infeasibility project tau to min_dt
        for k, tau in enumerate(dc.getTauVec()):
            min_dt = dc.lpr(k).min_dt()
            if tau < min_dt:
                print(f"tau {tau} < min_dt {min_dt} for segment {k}")
                dc.lpr(k)._tf = min_dt * 1.005

        # store the parameters
        if isinstance(stats, IterationStats):
            stats.tau_values.append(dc.getTauVec())
            stats.phi_values.append(dc.getPhiVec())
            stats.psi_values.append(dc.getPsiVec())
            stats.iterate += 1
            stats.alphas.append(op.alpha)

        # update step size
        if op.step_size_strategy == "diminishing":
            op.alpha *= op.beta
        elif op.step_size_strategy == "momentum":
            op.alpha = op.beta * op.alpha + (1-op.beta) * stats.global_gradient_norms[-1]

    # printers
    def printHeader(self, mode='a') -> None:
        with open(self._log_file, mode) as f:
            f.write("----|-----------|-----------|-----------|------------|-------------\n")
            f.write(" it | avrg cost | grad. nrm | step size | std st itr | std st viol \n")
            f.write("----|-----------|-----------|-----------|------------|-------------\n")
              
    def printIteration(self) -> None:
        stats = self._stats
        if stats.iterate % 10 == 0:
            self.printHeader()
        
        if len(stats.global_costs) == 0:
            return
        
        with open(self._log_file, 'a') as f:
            f.write("{:3d} | {:9.2e} | {:9.2e} | {:9.2e} | {:10d} | {:9.2e} \n".format(
                stats.iterate, 
                stats.global_costs[-1], 
                stats.global_gradient_norms[-1],
                stats.alphas[-1], 
                stats.steady_state_iterations[-1],
                stats.steady_state_violations[-1]
            ))
