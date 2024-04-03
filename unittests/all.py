from .Tests import *

if __name__ == "__main__":
    run("test_bilevel_optimization", test_bilevel_optimization)
    run("test_cycle", test_cycle)
    run("test_dist_to_boundary", test_dist_to_boundary)
    run("test_local_controller", test_local_controller)
    run("test_plot_world", test_plot_world)
    run("test_rrt", test_rrt)
    run("test_shift_time", test_shift_time)
    run("test_travel_cost", test_travel_cost)
    run("test_tsp", test_tsp)
    run("test_voronoi", test_voronoi)
