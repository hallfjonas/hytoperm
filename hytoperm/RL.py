
from typing import Tuple, List
import warnings
import numpy as np
import networkx as nx
import math

def JA(dt: float, 
       Omega: float,
       A: float, 
       Q: float, 
       H: float, 
       R: float
       ) -> float:
    '''
    Compute the local cost of monitored target.

    Parameters:
        dt (float): The integration step size.
        Omega (float): The current estimator variance.
        A (float): The target state transition factor.
        Q (float): The target state transition noise variance.
        H (float): The measurement factor.
        R (float): The measurment noise variance.
    '''
    if dt < 0:
        raise ValueError("Duration must be positive")
    
    G = H / R * H
    lam = 2 * math.sqrt(A**2 + Q*G)
    e = math.exp(-lam * dt)
    v1 = 1/Q * (-A + math.sqrt(A**2 + Q*G))
    v2 = 1/Q * (-A - math.sqrt(A**2 + Q*G))
    c1 = v2*Omega - 1
    c2 = -v1*Omega + 1
    return float(1/G * math.log((v1*c1 + v2*c2*e)/(v2-v1)) + dt/v1)


def JI(dt: float, 
       Omega: float,
       A: float, 
       Q: float
       ) -> float:
    '''
    Compute the local cost of an unmonitored target.

    Parameters:
        dt (float): The integration step size.
        Omega (float): The current estimator variance.
        A (float): The target state transition factor.
        Q (float): The target state transition noise variance.
    '''
    e = math.exp(2 * A * dt)
    return 1/(2*A)*(Omega + Q/(2*A))*(e - 1) - Q*dt/(2*A)


def OmegaA(dt: float, 
           Omega: float,
           A: float, 
           Q: float, 
           H: float, 
           R: float
           ) -> float:
    '''
    Update the estimator variance of a monitored target.

    Parameters:
        dt (float): The integration step size.
        Omega (float): The current estimator variance.
        Q (float): The target state transition noise variance.
        H (float): The measurement factor.
        R (float): The measurment noise variance.
    '''
    G = H / R * H
    lam = 2 * math.sqrt(A**2 + Q*G)
    e = math.exp(-lam * dt)
    v1 = 1/Q * (-A + math.sqrt(A**2 + Q*G))
    v2 = 1/Q * (-A - math.sqrt(A**2 + Q*G))
    c1 = v2*Omega - 1
    c2 = -v1*Omega + 1
    return (c1 + c2*e)/(v1*c1 + v2*c2*e)


def OmegaI(dt: float, 
           Omega: float,
           A: float, 
           Q: float
           ) -> float:
    '''
    Update the estimation variance of an unmonitored target.

    Parameters:
        dt (float): The integration step size.
        Omega (float): The initial variance of the estimator.
        target (Target): The target that is monitored.
    '''
    e = math.exp(2*A*dt)
    return (Omega + Q/(2*A))*e - Q/(2*A)


def GetFeasibleActions(p: int, 
                       graph: nx.DiGraph
                       ) -> List[int]:
    '''
    Get the feasible actions from a given node.

    Parameters:
        p (int): The current agent position.
                 Negative p refers to an node, positive p refers to an edge.
        graph (networkx.Graph): The graph of the environment.
    '''
    if p < 0:
        return list(graph.neighbors(-p))
    else:

        warnings.warn("This will most likely fail. Need to test.")
        warnings.warn("Trying to get the destination node of the pth edge.")
        return graph.edges[p][1]


def UpdateAgentPosition(dt: float,
                        p: int,
                        alpha: float,
                        action: int,
                        graph: nx.Graph
                        ) -> Tuple[int, float]:
    '''
    Update the agent position.

    Parameters:
        dt (float): The integration step size.
        p (int): The current agent position.
                 Negative p refers to an node, positive p refers to an edge.
        alpha (float): The agent progress along an edge (if traveling).
        action (int): The action to be taken.

    Returns:
        int: The new agent position.
        float: The new agent progress along an edge (if still traveling).
    '''

    if action not in GetFeasibleActions(p, graph):
        raise ValueError("Action is not feasible")
    
    if p > 0:
        alpha = min(1, alpha + dt/graph.edges[p]['weight'])
        if alpha == 1:
            p = action
            alpha = 0

    return p, alpha


def UpdateOmegas(dt: float,
                Omega: List[float],
                A: float, 
                Q: float, 
                p: List[int],
                H: List[float],
                R: List[float]
                ) -> float:
    '''
    Update the estimator variance.

    Parameters:
        dt (float): The integration step size.
        Omega (List[float]): The current estimator variances.
        A (List[float]): The target state transition factors.
        Q (List[float]): The target state transition noise variances.
        p (List[int]): The agent positions.
        H (List[float]): The measurement factors.
        R (List[float]): The measurment noise variances.

    Returns:
        List[float]: The new estimator variances.
    '''

    new_Omegas = []

    for i, Omega in enumerate(Omega):
        monitored = False
        for k, pos in enumerate(p):
            if pos < 0 and i == -pos:
                new_Omegas.append(OmegaA(dt, Omega, A[i], Q[i], H[k], R[k]))
                monitored = True
                break
        if not monitored:
            new_Omegas.append(OmegaI(dt, Omega, A[i], Q[i]))
            
    return new_Omegas
