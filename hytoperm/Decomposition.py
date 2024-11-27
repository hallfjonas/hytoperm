from typing import List
import numpy as np

class LocalParameters:
    def __init__(self, **kwargs):
        self._na = kwargs.get('na', 2)                                          # Dimension of the agent state
        self._no = kwargs.get('no', 2)                                          # Dimension of the uncertainty state    

        self._tau: float = kwargs.get('tau', 0.0)
        self._a_phi: np.ndarray = kwargs.get('a_phi', np.zeros(self._na))
        self._a_psi: np.ndarray = kwargs.get('a_psi', np.zeros(self._na))
        self._o_phi: np.ndarray = kwargs.get('a_psi', np.zeros(self._no))
        self._o_psi: np.ndarray = kwargs.get('a_psi', np.zeros(self._no))

    def nabla_I_m(self):
        pass

    def nabla_I_Delta(self):
        pass

class DecompositionParameters:
    def __init__(self, **kwargs):
        self._tvs: List[int] = kwargs.get('visiting_sequence', [])              # Time-varying states
        self._lpr: List[LocalParameters] = kwargs.get('local_params', [])       # Local parameters

    def nablak_I_sl(self, l: int, k: int):
        """
        Compute the gradient of the l-th switching segment wrt the kth 
        local parameters.
        """
        pass

    def __str__(self):
        return str(self.__dict__)
    