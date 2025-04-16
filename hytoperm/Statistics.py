
from typing import List
import numpy as np

class IterationStats:
    iterate: int = 0
    global_costs : List[float] = []
    global_gradients : List[np.ndarray] = []
    global_gradient_norms: List[float] = []
    steady_state_iterations : List[int] = []
    steady_state_violations: List[float] = []
    is_steady_state: List[bool] = []
    is_feasible: List[bool] = []
    tau_values: List[np.ndarray] = []
    phi_values: List[np.ndarray] = []
    psi_values: List[np.ndarray] = []
    rho_values: List[np.ndarray] = []
    alphas: List[float] = []
