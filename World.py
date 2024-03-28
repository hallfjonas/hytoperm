
from Plotters import *

import numpy as np
from typing import Tuple, List, Dict, Set
import matplotlib.pyplot as plt 
from abc import abstractclassmethod
from scipy.spatial import ConvexHull
import casadi as cad
import math

plotAttributes = PlotAttributes()

class Domain:
    def __init__(self, xrange = [0,1.5], yrange = [0,1]):
        self._xmin = xrange[0]
        self._xmax = xrange[1]
        self._ymin = yrange[0]
        self._ymax = yrange[1]

    def xrange(self) -> List[float]:
        return [self._xmin, self._xmax]

    def yrange(self) -> List[float]:
        return [self._ymin, self._ymax]
    
    def xmin(self) -> float:
        return self._xmin
    
    def xmax(self) -> float:
        return self._xmax
    
    def ymin(self) -> float:
        return self._ymin
    
    def ymax(self) -> float:    
        return self._ymax

class Region:

    """
    Represents an abstract region that.
    """

    def region(self):
        return self
    
    @abstractclassmethod
    def Contains(self, x: np.ndarray, tol : float = 0) -> bool:
        """
        Checks if a point is contained within the region.
        
        Args:
            x: Point to be checked.
            tol: Tolerance for the check.

        Returns:
            True iff the point is contained within the region.
        """
        pass

    @abstractclassmethod
    def Violates(self, x: np.ndarray, tol : float = 0) -> List[int]:
        """
        Checks which constraints are violated at x.
        
        Args:
            x: Point to be checked.
            tol: Tolerance for the check.

        Returns:
            A list of violated constraints.
        """
        pass

    @abstractclassmethod
    def DistToBoundary(self, x: np.ndarray) -> float:
        """
        Computes the distance to the boundary of the region.

        Args:
            x: Point to be checked.

        Returns:
            The distance to the boundary.
        """
        pass

    @abstractclassmethod
    def RandomBoundaryPoint(self) -> np.ndarray:
        pass

    @abstractclassmethod
    def ProjectToBoundary(self, x0, xf):
        '''
        This function projects a point onto the boundary of the region along the ray xf - x0, where x0 is assumed to be inside the region.
        
        Args:
            x0: Initial point (within the region).
            xf: Final point provides the direction (xf - x0).
        '''
        pass

    @abstractclassmethod
    def planPath(self, x0 : np.ndarray, xf : np.ndarray) -> List[np.ndarray]:
        '''
        This function plans a path between two points in the region.
        
        Args:
            x0: Initial point.
            xf: Final point.

        Returns:
            A list of waypoints.
        '''
        pass

    @abstractclassmethod
    def TravelCost(self, x0 : np.ndarray, xf : np.ndarray) -> float:
        """
        Computes the travel cost between two points in the region.

        Args:
            x0: Initial point.
            xf: Final point.
        """
        pass

    @abstractclassmethod
    def Plot(self, ax : plt.Axes, **kwargs) -> PlotObject:
        pass

    @abstractclassmethod
    def Fill(self, ax : plt.Axes, **kwargs) -> PlotObject:
        pass

    def GetGrid(self, domain : Domain, dx = 0.05, dy=0.05, mdtb = 0.1):
        assert(dx > 0); assert(dy > 0); assert(mdtb > 0)
        X = np.arange(domain.xmin(),domain.xmax(),dx)
        Y = np.arange(domain.ymin(),domain.ymax(),dy)
        V, W = np.meshgrid(X,Y)
        
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                tester = np.array([V[i,j],W[i,j]])
                if not self.Contains(tester):
                    V[i,j] = np.nan
                    W[i,j] = np.nan
                elif callable(getattr(self, 'DistToBoundary', None)):
                    dist = self.DistToBoundary(tester)
                    if dist < mdtb:
                        V[i,j] = np.nan
                        W[i,j] = np.nan
                else:
                    print(tester)
        return V, W

class Partition:
    def __init__(self, regions : Set[Region]) -> None:
        self._regions = regions

    def regions(self) -> Set[Region]:
        return self._regions

    def Plot(self, ax : plt.Axes, **kwargs) -> PlotObject:
        po = PlotObject()
        extended_kwargs = extendKeywordArgs(plotAttributes.partition.getAttributes())
        for r in self.regions():
            po.add(r.Plot(ax, **extended_kwargs))
        return po
    
from DataStructures import Node
class CPRegion(Region):
    """
    Represents a convex polygon region defined by linear inequalities g*x <= b.

    Attributes:
        g:                      Linear constraint functions.
        b:                      Constraint bounds.
        p:                      A point in the region.
    """

    def __init__(self, g, b, p : np.ndarray, domain : Domain = Domain()) -> None:
        """
        Initializes a Region object.

        Args:
            g (dict, optional): Dictionary of constraint gradients.
            b (dict, optional): Dictionary of constraint bounds.
            p (np.ndarray, optional): A point in the region.
        """
        
        self._g = {}
        self._b = {}
        self._ch : ConvexHull = None
        self._p : np.ndarray = None
        self._domain : Domain = None

        self.AssignConstraints(g, b, domain)
        self.AssignPoint(p)      

    def copy(self):
        return CPRegion(self.g(), self.b(), self.domain())

    def g(self) -> Dict[int, np.ndarray]:
        return self._g
    
    def b(self) -> Dict[int, float]:
        return self._b

    def p(self) -> np.ndarray:
        return self._p
    
    def domain(self) -> Domain:
        return self._domain

    def GetConvexHull(self) -> ConvexHull:
        if self._ch is None:
            self.AssignConvexHull()
        return self._ch
        
    def GetOrthogonalConstraintNodes(self, dx):
        nodes = self.GetConvexHull().vertices
        
        for i in range(len(nodes)):
            p = self.GetConvexHull().points[nodes[i]]
            q = self.GetConvexHull().points[nodes[i-1]]

            a = q - p
            normal = np.array([-a[1], a[0]])
            if abs(np.dot(a,dx)) < 1e-3 and np.dot(normal,dx) > 0:
                return p, q

        plt.plot(self.p()[0], self.p()[1], 'bo')            
        plt.quiver(self.p()[0], self.p()[1], dx[0], dx[1], angles='xy', color='blue')
        for i in range(len(nodes)):
            p = self.GetConvexHull().points[nodes[i]]
            q = self.GetConvexHull().points[nodes[i-1]]
            a = q - p

            plt.plot(p[0], p[1], 'bo')            
            plt.quiver(p[0], p[1], a[0], a[1], angles='xy', color='green')
        raise Exception("No orthogonal constraint found")
    
    def DistToBoundary(self, p) -> float:
        # Project p onto g*x = b
        min_dist = np.inf
        for i in self.g().keys():
            g = self.g()[i]
            b = self.b()[i]
            dist = np.abs(np.dot(g,p) - b)/np.linalg.norm(g)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def RandomBoundaryPointOrig(self) -> np.ndarray:
        """
        Generates a random boundary point of the region which.
                
        Returns:
            Tuple[np.ndarray, int]: A tuple containing the random boundary point and the index of the constraint it belongs to.
        """
        
        p = self.p()
        assert(p is not None)


        # This tolerance is used to identify
        tol = 1e-10

        # Draw a random ray passing through p and check for intersections with the region's boundary
        # Return the nearest intersection point along with the corresponding boundary constraint
        while True:
            
            alpha = np.random.uniform(0, np.pi)
            a = np.array([np.cos(alpha), np.sin(alpha)])
            b = np.dot(a, p)
            a_normal = np.array([-a[1], a[0]])

            G = np.zeros((2,2))
            h = np.zeros((2,1))
            G[0,0] = a[0]; G[0,1] = a[1]
            h[0] = b

            min_dist = np.inf
            min_dist_key = None
            min_dist_point = None
            
            for i in self.g.keys():  
                G[1,0] = self.g[i][0]; G[1,1] = self.g[i][1]
                h[1] = self.b[i]

                # Skip constraints that are parallel to the ray                
                if(np.abs(np.linalg.det(G)) < tol):
                    Warning("Parallel constraints detected, skipping boundary segment ...")
                    continue

                # solve the linear program and compute distance to p
                x = np.linalg.solve(G, h).flatten()
                dist = np.linalg.norm(x - p)

                # Store the closest boundary point
                if (dist < min_dist):
                    min_dist = dist
                    min_dist_key = i
                    min_dist_point = x

            if (min_dist_key != None):
                return min_dist_point

            ValueError("Warning: No boundary point found. Retrying...")  

    def RandomBoundaryPoint(self) -> np.ndarray:        
        i = np.random.randint(0, len(self.GetConvexHull().vertices))
        alpha = np.random.uniform(0, 1)
        p = (1 - alpha) * self._ch.points[self._ch.vertices[i]] + alpha * self._ch.points[self._ch.vertices[i-1]]
        return p

    def ProjectToBoundary(self, x0, xf):
        bdp = None

        if np.linalg.norm(xf - x0) < np.finfo(float).eps:
            return None
        
        # compute the normal vector of xf - x0
        n = np.array([-(xf[1] - x0[1]), xf[0] - x0[0]])/np.linalg.norm(xf - x0)
        h = np.dot(n, x0)
        closest = np.inf
        for i in self.g().keys():
            q = np.array([[self.g()[i][0], self.g()[i][1]], [n[0], n[1]]])
            r = np.array([self.b()[i], h])
            
            # Skip parallel constraints
            if (np.abs(np.linalg.det(q)) < np.finfo(float).eps):
                continue
            
            x = np.linalg.solve(q, r)

            # Skip non-forward directions
            if np.dot(x - x0, xf - x0) < np.finfo(float).eps:
                continue

            dist = np.linalg.norm(x - x0)
            if dist < closest:
                closest = dist
                bdp = x
        return bdp

    def AssignConstraints(self, g, b, domain : Domain) -> None:
        """
        Assigns the constraint gradients and bounds of the region.

        Args:
            g: Constraint gradients.
            b: Constraint bounds.
        """
        assert(isinstance(g, dict))
        assert(isinstance(b, dict))
        assert(isinstance(domain, Domain))
        assert(len(g) == len(b))
        for i in g.keys():
            assert(i in b.keys())
            assert(isinstance(g[i], np.ndarray))
            assert(isinstance(b[i], float))

        self._g = g.copy()
        self._b = b.copy()

        self.AddConstraint(np.array([-1,0]), domain.xmin())
        self.AddConstraint(np.array([1,0]), domain.xmax())
        self.AddConstraint(np.array([0,-1]), domain.ymin())
        self.AddConstraint(np.array([0,1]), domain.ymax())  

    def AssignPoint(self, p : np.ndarray) -> None:
        assert(isinstance(p, np.ndarray))
        assert(self.Contains(p))
        self._p = p

    def AssignConvexHull(self) -> None:
        ip = self.ComputeIntersections()
        vertices = []
        for p in ip:
            if self.Contains(p,tol=1e-10):
                vertices.append(p)
        if len(vertices) < 3:
            plt.plot(self.p()[0], self.p()[1], 'yo', markersize=5)
        self._ch = ConvexHull(vertices)

    def AddConstraint(self, g : np.ndarray, b) -> None:
        """
        Adds a new constraint to the region.

        Args:
            g: Constraint gradient.
            b: Constraint bound.
        """
        self._g[len(self._g)+1] = g
        self._b[len(self._b)+1] = b

    def Contains(self, x: np.ndarray, tol : float = 0) -> bool:
        for i in self.g().keys():
            g = self.g()[i]
            b = self.b()[i]
            if np.dot(g, x) > b + tol:
                return False
        return True
        
    def Violates(self, x: np.ndarray, tol : float = 0) -> List[int]:
        violated = []
        for i in self.g().keys():
            g = self.g()[i]
            b = self.b()[i]
            viol = np.dot(g, x) - b + tol
            if viol > 0:
                violated.append([i, viol])
        return violated

    def DistToBoundary(self, x: np.ndarray) -> float:
        dist = np.inf
        for i in self.g().keys():
            g = self.g()[i]
            b = self.b()[i]
            d = np.abs(np.dot(g, x)-b)/np.dot(g,g)
            if d < dist:
                dist = d
        return dist

    def ComputeIntersections(self) -> List[np.ndarray]:
        intersections = []
        for i in self.g().keys():
            for j in self.g().keys():
                if i >= j:
                    continue
                q = np.array([[self.g()[i][0], self.g()[i][1]], [self.g()[j][0], self.g()[j][1]]])
                r = np.array([self.b()[i], self.b()[j]])
                
                # Skip parallel constraints
                if (np.abs(np.linalg.det(q)) < 1e-14):
                    continue
                
                x = np.linalg.solve(q, r)
                intersections.append(x)
        return intersections
        
    def TravelCost(self, x0 : np.ndarray, xf : np.ndarray) -> float:
        return np.linalg.norm(xf - x0)

    def planPath(self, x0 : np.ndarray, xf : np.ndarray) -> List[np.ndarray]:
        return [x0, xf]

    def Plot(self, ax : plt.Axes, **kwargs) -> PlotObject:

        xs : list = self.GetConvexHull().points[self.GetConvexHull().vertices,0].tolist()
        ys : list = self.GetConvexHull().points[self.GetConvexHull().vertices,1].tolist()
        xs.append(xs[0])
        ys.append(ys[0])
        if ax is None:
            ax = plt.gca()
        return PlotObject(ax.plot(xs, ys, **kwargs))
    
    def PlotPoint(self, ax : plt.Axes = plt, **kwargs) -> PlotObject:
        return PlotObject(ax.plot(self.p()[0], self.p()[1], **kwargs))

    def Fill(self, ax : plt.Axes = plt, **kwargs) -> PlotObject:
        return PlotObject(ax.fill(self.GetConvexHull().points[self.GetConvexHull().vertices,0], self.GetConvexHull().points[self.GetConvexHull().vertices,1], **kwargs))

class Dynamics:
    def __init__(self, nx : int, nz : int, nu : int):
        self._nx : np.ndarray = None
        self._nz : np.ndarray = None
        self._nu : np.ndarray = None
        self.zx : np.ndarray = None
        self.zz : np.ndarray = None
        self.zu : np.ndarray = None

        self.SetNX(nx)
        self.SetNZ(nz)
        self.SetNX(nu)

    def dynamics(self):
        return self

    def nx(self) -> int:
        return self._nx
    
    def nz(self) -> int:
        return self._nz
    
    def nu(self) -> int:
        return self._nu
    
    def SetNX(self, nx : int) -> None:
        assert(isinstance(nx, int))
        assert(nx >= 0)
        self._nx = nx
        self.zx = self.ZeroVec(nx)

    def SetNZ(self, nz : int) -> None:
        assert(isinstance(nz, int))
        assert(nz >= 0)
        self._nz = nz
        self.zz = self.ZeroVec(nz)

    def SetNU(self, nu : int) -> None:
        assert(isinstance(nu, int))
        assert(nu >= 0)
        self._nu = nu
        self.zu = self.ZeroVec(nu)

    def ZeroVec(self, n):
        return np.zeros(n)

    @abstractclassmethod
    def __call__(self, x, z, u):
        pass

    def eval(self,  x = None, z = None, u = None):
        if x is None:
            x = self.zx
        if z is None:
            z = self.zz
        if u is None:
            u = self.zu
        return self(x,z,u)
            
    def PlotVectorField(self, XY : list, ax : plt.Axes, scale = 1, **kwargs):
        assert(len(XY) == 2)
        X = XY[0]; Y = XY[1]
        
        DX = np.zeros(X.shape)
        DY = np.zeros(X.shape)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x = X[i,j]
                y = Y[i,j]
                state = np.array([x,y])
                if x is np.nan or y is np.nan:
                    DX[i,j]= np.nan
                    DY[i,j]= np.nan
                    continue
                F = self.eval(state)
                DX[i,j]= F[0]
                DY[i,j]= F[1]

        eka = extendKeywordArgs(plotAttributes.vector_field.getAttributes(), **kwargs)
        return ax.quiver(X, Y, scale*DX, scale*DY, pivot='mid', **eka)

class ConstantDynamics(Dynamics):
    def __init__(self, nx : int, nz : int, nu : int, v : np.ndarray):
        super().__init__(nx,nz,nu)
        self._v : np.ndarray = None
        self.SetV(v)

    def __call__(self, x, z, u):
        return self.v()

    def SetV(self, v : np.ndarray) -> None:
        assert(isinstance(v, np.ndarray))
        self._v = v

    def v(self) -> np.ndarray:
        return self._v

class DynamicCPRegion(CPRegion):
    def __init__(self, g,b,p,domain,dynamics : Dynamics):
        super().__init__(g,b,p,domain)
        self._dynamics : Dynamics = None
        self.AssignDynamics(dynamics)

    def AssignDynamics(self, dynamics : Dynamics) -> None:
        assert(isinstance(dynamics, Dynamics))
        self._dynamics = dynamics

    def AssignRegion(self, region : Region) -> None:
        assert(isinstance(region, Region))
        self._region = region
    
    def dynamics(self) -> Dynamics:
        return self._dynamics

class ConstantDCPRegion(DynamicCPRegion):
    def __init__(self, g,b,p,domain,dynamics : ConstantDynamics):
        super().__init__(g,b,p,domain,dynamics)

    def dynamics(self) -> ConstantDynamics:
        return self._dynamics
    
    def TravelCostAD(self, x0 : cad.SX, xf : cad.SX) -> cad.SX:
        '''
        This function determines the optimal travel time between two points in the region.
        This is done by solving a quadratic equation obtained from the root finding problem
            u'*u = 1, u = (xf - x0)/t - v.
        Plugging the second equation into the first yields
            (xf - x0)'(xf - x0)/t^2 - 2*(xf - x0)'v/t + v'*v - 1 = 0
        and multiplying by t^2 yields
            (xf - x0)'(xf - x0) - 2*(xf - x0)'v*t + v'*v*t^2 - t^2 = 0.
        Organizing the terms yields
            (v'*v - 1)*t^2 - 2*(xf - x0)'v*t + (xf - x0)'(xf - x0) = 0.
        This is a quadratic equation of the form
            at^2 + bt + c = 0
        with solution 
            t = (-b +/- sqrt(b^2 - 4ac))/(2a).

        Args:
            x0: Initial point.
            xf: Final point.

        Returns:
            float: The optimal travel cost between x0 and xf.
        '''
        v = self.dynamics().v()
        a = cad.dot(v,v) - 1
        b = -2 * cad.dot(xf - x0, v)
        c = cad.dot(xf - x0, xf - x0)
        delta = cad.power(b,2) - 4 * a * c
        return (-b - cad.sqrt(delta)) / (2 * a)

    def TravelCost(self, x0 : np.ndarray, xf : np.ndarray) -> float:
        '''
        This function determines the optimal travel time between two points in the region.
        This is done by solving a quadratic equation obtained from the root finding problem
            u'*u = 1, u = (xf - x0)/t - v.
        Plugging the second equation into the first yields
            (xf - x0)'(xf - x0)/t^2 - 2*(xf - x0)'v/t + v'*v - 1 = 0
        and multiplying by t^2 yields
            (xf - x0)'(xf - x0) - 2*(xf - x0)'v*t + v'*v*t^2 - t^2 = 0.
        Organizing the terms yields
            (v'*v - 1)*t^2 - 2*(xf - x0)'v*t + (xf - x0)'(xf - x0) = 0.
        This is a quadratic equation of the form
            at^2 + bt + c = 0
        with solution 
            t = (-b +/- sqrt(b^2 - 4ac))/(2a).

        Args:
            x0: Initial point.
            xf: Final point.

        Returns:
            float: The optimal travel cost between x0 and xf.
        '''
        v = self.dynamics().v()
        a = np.dot(v,v) - 1
        b = -2 * np.dot(xf - x0, v)
        c = np.dot(xf - x0, xf - x0)

        # final point is equal to initial point
        if c < np.finfo(float).eps:
            return 0
        
        # missing affine part (allows for factoring out t)
        if abs(a) < np.finfo(float).eps:
            if b >= 0:
                return np.inf
            else:
                return -c/b

        # general solution
        delta = pow(b,2) - 4 * a * c
        t_star = 0
        if (delta < 0):
            return np.inf
        
        t1 = (-b + np.sqrt(delta)) / (2 * a)
        t2 = (-b - np.sqrt(delta)) / (2 * a)
        if t1 < 0 and t2 < 0:
            return np.inf

        if t1 > 0 and t2 > 0:
            t_star = min(t1,t2)
        else:
            t_star = max(t1,t2)
        u_star = (xf - x0)/t_star - v
        return t_star

class Target:
    
    def __init__(self, pos : np.ndarray, region : Region = None, phi0 : np.ndarray = None, A : np.ndarray = None, Q : np.ndarray = None) -> None:
        
        self._r : Region = None
        self._p : np.ndarray = None
        self._phi : np.ndarray = None
        self.A : np.ndarray = None
        self.Q : np.ndarray = None
        self.name : str = None

        self.assignRegion(region)
        self.assignPosition(pos)
        self.assignInternalState(phi0)
        self.assignStateMatrix(A)
        self.assignCovariance(Q)
        
    def p(self) -> np.ndarray:
        return self._p

    def region(self) -> Region:
        return self._r
    
    def internalState(self) -> np.ndarray:
        return self._phi

    def getNumberOfStates(self) -> int:
        return self.A.shape[0]

    def assignPosition(self, p : np.ndarray) -> None:
        assert(isinstance(p, np.ndarray))
        
        if (self.region() is not None):
            assert(self.region().Contains(p))
            self._p = p

    def assignRegion(self, r : Region) -> None:
        assert(isinstance(r, Region) or r is None)
        self._r = r

    def assignInternalState(self, phi0 : np.ndarray) -> None:
        assert(isinstance(phi0, np.ndarray))
        self._phi = phi0

    def assignStateMatrix(self, A : np.ndarray) -> None:
        assert(isinstance(A, np.ndarray))
        if (A.ndim == 1 and A.shape[0] == 1):
            self.A = A.reshape((1,1))
            return
        else:
            assert(A.ndim == 2)
        self.A = A

    def assignCovariance(self, Q : np.ndarray) -> None:
        assert(isinstance(Q, np.ndarray))
        if (Q.ndim == 1 and Q.shape[0] == 1):
            self.Q = Q.reshape((1,1))
            return
        else:
            assert(Q.ndim == 2)
        self.Q = Q

    def plot(self, ax : plt.Axes = plt, annotate=True, **kwargs) -> PlotObject:
        eka = extendKeywordArgs(plotAttributes.target.getAttributes(), **kwargs)
        po = PlotObject(ax.plot(self._p[0], self._p[1], **eka))

        if annotate:
            dx = 0; dy = 0.05
            if self.name == '4':
                dx = 0.05
                dy = -0.025
            if self.name == '3':
                dx = -0.025
                dy = -0.075
            if self.name == '2':
                dx = -0.07
                dy = -0.05
            if self.name == '1':
                dx = 0.0
                dy = 0.04
            po.add(ax.text(self._p[0] + dx, self._p[1] + dy, self.name))

        return po

class World:
    
    def __init__(self, objs = [], domain : Domain = Domain()) -> None:
        
        self._regions : Set[Region] = []
        self._targets : List[Target] = []
        self._region_to_target : Dict[Region, Target] = {}
        self._target_to_region : Dict[Target, Region] = {}
        self._partition : Partition = None
        self._domain : Domain = domain

        self.SetRegions(objs)
        self.SetPartition()
        
    def SetRegions(self, objs) -> None:
        for obj in objs:
            try:
                self.AddRegion(obj.region())
            finally:
                pass
    
    def SetTargets(self, objs) -> None:
        for obj in objs:
            try:
                self.AddTarget(obj.target())
            finally:
                pass

    def AddRegion(self, region : Region) -> None:
        assert(isinstance(region, Region))
        if region not in self._regions:
            self._regions.append(region)

    def AddTarget(self, target : Target) -> None:
        assert(isinstance(target, Target))
        if target not in self._targets:
            self._targets.append(target)

    def SetPartition(self) -> None:
        self._partition = Partition(self.regions()) 

    def add_target_region(self, target : Target, region : Region) -> None:
        self._targets.append(target)
        self._regions.append(region)
        self._target_to_region[target] = region
        self._region_to_target[region] = target
    
    # getters
    def regions(self) -> Set[Region]:
        return self._regions
    
    def targets(self) -> List[Target]:
        return self._targets
        
    def getTarget(self, name : str) -> Target:
        for t in self._targets:
            if t.name == name:
                return t
        return None

    def partition(self) -> Partition:
        return self._partition
    
    def target(self, i) -> Target:
        assert(i < self.NT() and i >= 0)
        return self._targets[i]
    
    def NT(self) -> int:
        return len(self._targets)
    
    def NR(self) -> int:
        return len(self._regions)
    
    def domain(self) -> Domain:
        return self._domain
    
    def GetRegions(self, p : np.ndarray, tol = 1e-10) -> Set[Region]:
        regions = set()
        for r in self._regions:
            assert(isinstance(r, Region))
            if r.Contains(p, tol=tol):
                regions.add(r)
        return regions

    def get_meshgrid(self, dx = 0.005, dy = 0.005):
        x = np.arange(self.domain().xmin()-0.5*dx, self.domain().xmax()+0.5*dx, dx)
        y = np.arange(self.domain().ymin()-0.5*dy, self.domain().ymax()+0.5*dy, dy)
        X, Y = np.meshgrid(x, y)
        Z = np.nan*np.ones(X.shape)
        return X, Y, Z

    # plotters
    def PlotMissionSpace(self, ax, add_target_labels=False, fill_empty_regions=True) -> PlotObject:
        po = PlotObject()
        
        po.add(self.partition().Plot(ax))
        eka = extendKeywordArgs(plotAttributes.partition_background.getAttributes())
        for region in self.regions():
            has_target = False
            for target in self.targets():
                if target.region() == region:
                    has_target = True
                    break
            if not has_target and fill_empty_regions:
                po.add(region.Fill(ax, **eka))
        
        for target in self._targets:
            assert(isinstance(target, Target))
            po.add(target.plot(ax, annotate=add_target_labels))

        for region in self.regions():
            dynamics = region.dynamics()
            assert(isinstance(dynamics, Dynamics))
            d = 0.033
            dynamics.PlotVectorField(region.GetGrid(self.domain(),dx=d,dy=d,mdtb=d), ax, 0.6)

    def plotDistToBoundary(self, ax : plt.Axes = plt) -> PlotObject:
        X, Y, Z = self.get_meshgrid()
        for i in range(X.shape[0]):
            for j in range(Y.shape[1]):
                p = np.array((X[i,j], Y[i,j]))
                for region in self.regions():
                    if region.Contains(p):
                        Z[i,j] = region.DistToBoundary(p)
        return PlotObject(ax.contourf(X, Y, Z, antialiased=True, alpha=0.5))

    def plotTravelCostPerRegion(self, ax : plt.Axes = plt) -> PlotObject:
        X, Y, Z = self.get_meshgrid()
        for i in range(X.shape[0]):
            for j in range(Y.shape[1]):
                p = np.array((X[i,j], Y[i,j]))
                for region in self.regions():
                    assert(isinstance(region, CPRegion))
                    if region.Contains(p):
                        Z[i,j] = region.TravelCost(p, region.p())
                        if Z[i,j] == np.inf:
                            Z[i,j] = -region.TravelCost(region.p(), p)
        return PlotObject(ax.contourf(X, Y, Z, antialiased=True, alpha=0.5))
