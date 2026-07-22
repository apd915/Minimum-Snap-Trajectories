import sys
print("Hello from start of file")
import os
import time
import numpy as np

# Add the rrt-astar_mavsim directory to the Python path for legacy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rrt-astar_mavsim')))
# Add the C++ build directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cpp', 'build')))

import min_snap_bindings
print("Hello after import")

def run_benchmark():
    print("Inside run_benchmark")
    # Build Map Bounds
    map_bounds = np.array([100.0, 100.0, 15.0])

    # Build floating blocks in Python to pass to C++
    obstacles = []
    num_blocks = 4
    block_width = 10.0
    x_inc = 100.0 / num_blocks
    y_inc = 100.0 / num_blocks
    z_inc = 15.0 / num_blocks
    
    x_start = x_inc / 2.0
    y_start = y_inc / 2.0
    z_start = z_inc / 2.0
    
    try:
        for i in range(num_blocks):
            for j in range(num_blocks):
                for k in range(num_blocks):
                    cx = x_start + i * x_inc
                    cy = y_start + j * y_inc
                    cz = z_start + k * z_inc
                    
                    min_pt = np.array([cx - block_width / 2.0, cy - block_width / 2.0, cz - block_width / 2.0])
                    max_pt = np.array([cx + block_width / 2.0, cy + block_width / 2.0, cz + block_width / 2.0])
                    
                    obs = min_snap_bindings.ObstacleBox(min_pt, max_pt)
                    obstacles.append(obs)
    except Exception as e:
        print("EXCEPTION: ", e)
    
    print("Obstacles built")

    # Configure Planner
    config = min_snap_bindings.FrontEndConfig()
    config.map_bounds = map_bounds
    config.sfc_height = 5.0
    config.sfc_width = 5.0
    config.sfc_start_ext = 5.0
    config.sfc_end_ext = 5.0
    config.spline_type = "clamped"
    config.aircraft_type = "fixed-wing"
    
    planner = min_snap_bindings.TrajectoryPlanner(config, obstacles, 3.0, 2.0)

    start = np.array([2.0, 2.0, 5.0])
    goal = np.array([100.0, 100.0, 15.0])
    
    print("============================================")
    print("  PYTHON RUNNING C++ BINDINGS (FLOATING BLOCKS) ")
    print("============================================")
    
    # Run the mission
    result = planner.plan_mission(start, goal)
    
    if result.success:
        print("Success! Trajectory Optimized.")
        print("Total Ctrl Points: ", result.total_control_points)
        print("SFC Time:          ", result.sfc_time_ms, "ms")
        print("Opt Time:          ", result.opt_time_ms, "ms")
        print("Overhead Time:     ", result.overhead_ms, "ms")
        print("Total Time:        ", result.total_time_ms, "ms")
    else:
        print("Optimization Failed.")

run_benchmark()
