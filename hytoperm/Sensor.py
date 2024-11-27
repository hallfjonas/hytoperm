
# external imports
import casadi as cad

# internal imports
from .World import *


'''
SensingQualityFunction: Abstract class definition for any sensing function.

Created on: Mar 28 2024

@author: Jonas Hall
'''
class SensingQualityFunction:
    def __init__(self):
        pass

    def __call__(self, p, q):
        pass


class ConstantQualityFunction(SensingQualityFunction):
    def __init__(self, c = 1):
        self._c : float = None
        self.assignConstant(c)

    def assignConstant(self, c : float) -> None:
        if c < 0 or c > 1:
            raise ValueError("Constant must be in the interval [0, 1].")
        self._c = c

    def __call__(self, p, q):
        return self._c


class GaussianQualityFunction(SensingQualityFunction):
    def __init__(self, c1: float = 50, c2: float = 1):
        self._c1 : float = None
        self._c2 : float = None
        self.assignConstants(c1, c2)

    def assignConstants(self, c1 : float, c2 : float) -> None:
        if c1 <= 0 or c2 <= 0:
            raise ValueError("Constants must be positive.")
        try:
            self._c1 = float(c1)
            self._c2 = float(c2)
        except ValueError:
            raise ValueError("Constants must be a number.")            

    def __call__(self, p, q):
        delta = p - q
        sqr_dist = cad.dot(delta, delta)
        return self._c2*cad.exp(-self._c1*sqr_dist)


class SinusoidalQualityFunction(SensingQualityFunction):
    def __init__(self, c1 = 3.0, c2 = 20.0, c3 = 40.0):
        self._c1 : float = None
        self._c2 : float = None
        self._c3 : float = None
        self.assignConstants(c1, c2, c3)

    def assignConstants(self, c1 : float, c2 : float, c3 : float) -> None:
        if c3 <= 0:
            raise ValueError("Constant c3 must be positive.")
        try:
            self._c1 = float(c1)
            self._c2 = float(c2)
            self._c3 = float(c3)
        except ValueError:
            raise ValueError("Constants must be numbers.")

    def __call__(self, p, q):
        delta = p - q
        expArg = -self._c3*cad.dot(delta,delta)
        sinArg = self._c1*delta[0]
        cosArg = self._c2*delta[1]
        return 0.5*cad.exp(expArg)*(cad.sin(sinArg)**2 + cad.cos(cosArg)**2)


'''
Sensor: A collection of sensing quality functions and measurement models 
        for each target. 

Created on: Mar 28 2024

@author: Jonas Hall
'''
class Sensor:
    def __init__(self, **kwargs):
        self._p : np.ndarray = None

        p = kwargs.get('p', None)
        if p is not None:
            self.setPosition(p)
        
    def getPosition(self) -> np.ndarray:
        return self._p
         
    # getters
    def getQualityFunction(self, **kwargs) -> SensingQualityFunction:
        pass
    
    def getSensingQuality(self, target: Target) -> float:
        return self.getQualityFunction(target)(self.getPosition(), target.p())
        
    def getMeasurementMatrix(self, target: Target) -> np.ndarray:
        pass

    def getMeasurementNoiseMatrix(self, target: Target) -> np.ndarray:
        pass

    def getMeasurementNoiseInverse(self, target: Target) -> np.ndarray:
        pass

    def getMeasurement(self, target: Target):
        p = self.getPosition()
        q = target.p()
        quality = self.getSensingQuality(target=target)(p, q)
        H = self.getMeasurementMatrix(target=target)
        targetState = target.internalState()
        return quality * H @ targetState + self.drawNoise(target=target) 
        
    def drawNoise(self, target: Target) -> np.ndarray:
        R = self.getMeasurementNoiseMatrix(target=target)
        return np.random.multivariate_normal(0, R)
    
    # setters
    def setPosition(self, p : np.ndarray) -> None:
        self.checkPosition(p)
        self._p = p

    def setTargetQualityFunction(
            self, 
            sqf : SensingQualityFunction,
            **kwargs
            ) -> None:
        pass
      
    def setNoiseMatrix(self, R : np.ndarray, **kwargs) -> None:
        pass

    def setMeasurementMatrix(self, H : np.ndarray, **kwargs) -> None:
        pass

    def checkMeasurementMatrix(self, H : np.ndarray) -> None:
        if not isinstance(H, np.ndarray):
            raise ValueError("H must be a numpy array.")
        if H.shape[0] != H.shape[1]:
            raise ValueError("H must be square.")
        
    def checkNoiseMatrix(self, R : np.ndarray) -> None:
        if not isinstance(R, np.ndarray):
            raise ValueError("R must be a numpy array.")
        try:
            return np.linalg.inv(R)
        except np.linalg.LinAlgError:
            raise ValueError(f"Failed to invert noise matrix R.")
        
    def checkSensorQualityFunction(self, sqf : SensingQualityFunction) -> None:
        if not isinstance(sqf, SensingQualityFunction):
            raise ValueError("sqf must be of type SensingQualityFunction.")
        
    def checkPosition(self, p : np.ndarray) -> None:
        if not isinstance(p, np.ndarray):
            raise ValueError("Position must be a numpy array.")
        
    def getTarget(self, kwargs) -> Target:
        target = kwargs.get('target', None)
        if target is None:
            raise ValueError("Target must be provided.")
        if not isinstance(target, Target):
            raise ValueError("Target must be of type Target.")
        return target

class HomogeneousSensor(Sensor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._sqf : SensingQualityFunction = None
        self._H : np.ndarray = None
        self._R : np.ndarray = None
        self._Rinv : np.ndarray = None

    def getQualityFunction(self, target: Target) -> SensingQualityFunction:
        return self._sqf
    
    def getMeasurementMatrix(self, target: Target) -> np.ndarray:
        return self._H

    def getMeasurementNoiseMatrix(self, target: Target) -> np.ndarray:
        return self._R

    def getMeasurementNoiseInverse(self, target: Target) -> np.ndarray:
        return self._Rinv
    
    def setTargetQualityFunction(
            self, 
            sqf : SensingQualityFunction,
            **kwargs
            ) -> None:
        self.checkSensorQualityFunction(sqf)
        self._sqf = sqf
      
    def setNoiseMatrix(self, R : np.ndarray, **kwargs) -> None:
        Rinv = self.checkNoiseMatrix(R)
        self._R = R
        self._Rinv = Rinv
        try:
            self._Rinv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            raise ValueError("R must be invertible.")

    def setMeasurementMatrix(self, H : np.ndarray, **kwargs) -> None:
        self.checkMeasurementMatrix(H)
        self._H = H

class HeterogeneousSensor(Sensor):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._ttsqm : Dict[Target, SensingQualityFunction] = {}                 # target to sensing quality function mapper
        self._ttHm : Dict[Target, np.ndarray] = {}                              # target to measurement matrix mapper
        self._ttRm : Dict[Target, np.ndarray] = {}                              # target to measurement noise mapper
        self._ttRinvm : Dict[Target, np.ndarray] = {}                           # target to measurement noise inverse mapper
        
    # getters
    def targetToSQFMapper(self) -> Dict[Target, SensingQualityFunction]:
        return self._ttsqm
    
    def getQualityFunction(self, **kwargs) -> SensingQualityFunction:
        return self.targetToSQFMapper()[self.getTarget(kwargs)]
    
    def getSensingQuality(self, **kwargs) -> float:
        return self.getQualityFunction(self.getTarget(kwargs))(
            self.getPosition(), self.getTarget(kwargs).p()
        )
    
    def getMeasurementMatrix(self, target: Target) -> np.ndarray:
        return self._ttHm[self.getTarget(target)]

    def getMeasurementNoiseMatrix(self, target: Target) -> np.ndarray:
        return self._ttRm[self.getTarget(target)]

    def getMeasurementNoiseInverse(self, target: Target) -> np.ndarray:
        return self._ttRinvm[self.getTarget(target)]

    def setTargetQualityFunction(
            self, 
            sqf : SensingQualityFunction,
            **kwargs
            ) -> None:
        self.checkSensorQualityFunction(sqf)
        self.targetToSQFMapper()[self.getTarget(kwargs)] = sqf
      
    def setNoiseMatrix(self, R : np.ndarray, **kwargs) -> None:
        target = self.getTarget(kwargs)
        Rinv = self.checkNoiseMatrix(R)
        self._ttRm[target] = R
        self._ttRinvm[target] = Rinv

    def setMeasurementMatrix(self, H : np.ndarray, **kwargs) -> None:
        target = self.getTarget(kwargs)
        self.checkMeasurementMatrix(H)
        self._ttHm[target] = H
