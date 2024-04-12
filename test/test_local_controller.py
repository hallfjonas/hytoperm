
# external imports
import unittest

# internal imports
from hytoperm import *


class TestLocalController(unittest.TestCase):
    
    def testLocalController(self):
        n_sets = 20
        ex = Experiment.generate(n_sets=n_sets)
        assert(isinstance(ex, Experiment))
        
        target = ex._world.targets()[0]
        sensor = ex.agent().sensor()

        phi = SwitchingPoint(target.region().randomBoundaryPoint())
        psi = SwitchingPoint(target.region().randomBoundaryPoint())
        tf = 10
        Omega0 = {}
        for t in ex._world.targets():
            Omega0[t] = np.eye(1)
        lmp = SwitchingParameters(phi=phi,psi=psi,tf=tf,Omega0=Omega0)
        mc = MonitoringController(target, sensor)

        mc.buildOptimalMonitoringSolver(target, sensor)
        tp, tmse, tomega, tu, cost, lam = mc.optimalMonitoringControl(lmp)

        ex, ax = ex.plotWorld(with_sensor_quality=True)
        tp.plotStateVsState(0,1, ax)


if __name__ == "__main__":
    unittest.main()
