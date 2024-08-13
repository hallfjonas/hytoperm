
from hytoperm import *
import matplotlib.pyplot as plt 

def testSegmentOrder():
    n_sets = 8
    ex = Experiment.generate(n_sets=n_sets)
    assert(isinstance(ex, Experiment))
    ex.agent().computeVisitingSequence()
    ex.agent().initializeCycle()
    ex.agent()._cycle.simulate()
    
    fig, ax = plt.subplots()
    for i, ss in enumerate(ex.agent()._switchingSegments):
        po = ss.plotInMissionSpace(ax, label=f"s {i}", linewidth=3)
        sp = ss.getStartPoint()
        ep = ss.getEndPoint()
        ax.plot([sp[0]], [sp[1]], label=f"s {i} start", color=po._objs[0].get_color(), marker='o', markersize=40, alpha=0.3)
        ax.plot([ep[0]], [ep[1]], label=f"s {i} end", color=po._objs[0].get_color(), marker='*', markersize=30, alpha=0.5)

    for i, ms in enumerate(ex.agent()._monitoringSegments):
        po = ms.plotInMissionSpace(ax, label=f"m {i}", linewidth=3)
        sp = ms.getStartPoint()
        ep = ms.getEndPoint()
        ax.plot([sp[0]], [sp[1]], label=f"m {i} start", color=po._objs[0].get_color(), marker='s', markersize=20, alpha=0.7)
        ax.plot([ep[0]], [ep[1]], label=f"m {i} end", color=po._objs[0].get_color(), marker='d', markersize=10, alpha=0.9)
        
    plt.legend()
    plt.show()

def testTrajectoryPointsVisually():
    n_sets = 8
    ex = Experiment.generate(n_sets=n_sets, homogeneous_agents=True)
    assert(isinstance(ex, Experiment))
    ex.agent().computeVisitingSequence()
    ex.agent().initializeCycle()
    ex.agent()._cycle.simulate()
    
    fig, ax = ex.plotWorld(with_sensor_quality=False)
    plt.ion()
    plt.show()
    
    for i in range(len(ex.agent()._switchingSegments)):
        ss = ex.agent()._switchingSegments[i]
        for k in range(ss.pTrajectory.x.shape[1]):
            p = ss.pTrajectory.x[:,k]
            ax.plot(p[0], p[1], 'o', color='r', markersize=10)
            plt.pause(0.05)
        plt.pause(0.25)
        if i < len(ex.agent()._monitoringSegments):
            ms = ex.agent()._monitoringSegments[i]
            for k in range(ms.pTrajectory.x.shape[1]):
                p = ms.pTrajectory.x[:,k]
                ax.plot(p[0], p[1], 'o', color='b')
                plt.pause(0.005)
        plt.pause(0.25)
    
    plt.pause(2)

if __name__ == "__main__":
    testTrajectoryPointsVisually()
    testSegmentOrder()
