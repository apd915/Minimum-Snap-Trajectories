import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import rrt_mavsim.parameters.plotter_parameters as PLOT
import rrt_mavsim.parameters.floatingBlocks_parameters as BLOCKS
from scipy.interpolate import BSpline
import sys
import os

# Force Python to add the parent 'python/' directory to its search path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from min_snap_natural import MinSnapEvalNatural
from min_snap_clamped import MinSnapEvalClamped

# ==========================================
# BENCHMARKING FUNCTIONS
# ==========================================
def run_batch_performance_test():
    """
    Tests execution time scalability by simulating a drone rapidly 
    calculating massive batches of trajectories.
    """
    batch_sizes = [10, 100, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    execution_times = []
    
    snap_degree = 4
    snap_ctrl_pts = 11

    print("\nPre-computing Q Matrix once for all batches...")
    evaluator = MinSnapEvalNatural(snap_ctrl_pts, snap_degree)
    Q_d4_M = evaluator.get_Q_matrix()
    
    print("\nRunning Batch Execution Test...")
    for num_trajectories in batch_sizes:
        print(f"Calculating {num_trajectories:,} random trajectories...")
        
        start_exec = time.perf_counter()
        
        for _ in range(num_trajectories):
            # Generate random boundaries
            p0, pf = np.random.rand(3, 1) * 10, np.random.rand(3, 1) * 10
            v0, vf = np.random.rand(3, 1) * 5 - 2.5, np.random.rand(3, 1) * 5 - 2.5
            a0, af = np.random.rand(3, 1) * 2 - 1, np.random.rand(3, 1) * 2 - 1
            
            S = np.hstack((p0, v0, a0))
            E = np.hstack((pf, vf, af))
            SE = np.hstack((S, E))

            # The real-time mapping math
            C_p_snap = SE @ Q_d4_M
            
        end_exec = time.perf_counter()
        execution_times.append(end_exec - start_exec)

    # Plot batch results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(batch_sizes, execution_times, 'g-o', linewidth=2, markersize=6)
    ax.set_title('Batch Processing Time for Natural Uniform Minimum Snap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Trajectories Computed', fontsize=12)
    ax.set_ylabel('Total Computation Time (seconds)', fontsize=12)
    ax.ticklabel_format(style='plain', axis='x')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.show()


def run_performance_benchmark(max_control_points=100, iterations=1000, degree=4):
    """
    Benchmarks the setup (offline) and execution (real-time) time of the solver 
    as the number of control points scales up.
    """
    print(f"\n🚀 Starting Benchmark: 7 to {max_control_points} Control Points")
    print(f"   Running {iterations} random trajectories per step...\n")
    
    ctrl_pts_range = range(7, max_control_points + 1, 2)
    setup_times_ms = []
    exec_times_us = []
    
    for num_pts in ctrl_pts_range:
        # 1. SETUP PHASE (Boot-up Math)
        start_setup = time.perf_counter()
        evaluator = MinSnapEvalNatural(num_pts, degree)
        Q_matrix = evaluator.get_Q_matrix()
        end_setup = time.perf_counter()
        
        setup_times_ms.append((end_setup - start_setup) * 1000)
        
        # 2. EXECUTION PHASE (Real-time Math)
        start_exec = time.perf_counter()
        for _ in range(iterations):
            p0, pf = np.random.rand(3, 1) * 10, np.random.rand(3, 1) * 10
            v0, vf = np.random.rand(3, 1) * 5 - 2.5, np.random.rand(3, 1) * 5 - 2.5
            a0, af = np.random.rand(3, 1) * 2 - 1, np.random.rand(3, 1) * 2 - 1
            
            S, E = np.hstack((p0, v0, a0)), np.hstack((pf, vf, af))
            SE = np.hstack((S, E))
            C_optimal = SE @ Q_matrix 
            
        end_exec = time.perf_counter()
        
        avg_exec_us = ((end_exec - start_exec) / iterations) * 1_000_000
        exec_times_us.append(avg_exec_us)
        
        print(f"Pts: {num_pts:3d} | Setup: {setup_times_ms[-1]:6.2f} ms | Exec: {avg_exec_us:6.3f} µs")

    # Plot Scaling Results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(ctrl_pts_range, setup_times_ms, 'r-o', linewidth=2)
    ax1.set_title('Boot-up Time (Q Matrix Generation)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Number of Control Points')
    ax1.set_ylabel('Time (Milliseconds)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax2.plot(ctrl_pts_range, exec_times_us, 'b-o', linewidth=2)
    ax2.set_title('Real-Time Execution (SE @ Q)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Number of Control Points')
    ax2.set_ylabel('Time (Microseconds)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    
    plt.suptitle('Minimum Snap B-Spline Performance Scaling', fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_sfc_wireframes(ax, corridors):
    """
    Reconstructs the 8 3D corners of each Safe Flight Corridor and 
    plots them as a transparent green wireframe.
    """
    for corridor in corridors:
        sfc = corridor.getSFC()
        # Ensure your dim/trans are in meters, not voxels!
        dim = sfc.dimensions 
        trans = sfc.translation 
        rot = sfc.rotation
        
        # 1. Create the 8 corners of a box centered at local origin
        dx, dy, dz = dim[0,0]/2, dim[1,0]/2, dim[2,0]/2
        corners_local = np.array([
            [-dx, -dy, -dz],
            [ dx, -dy, -dz],
            [-dx,  dy, -dz],
            [ dx,  dy, -dz],
            [-dx, -dy,  dz],
            [ dx, -dy,  dz],
            [-dx,  dy,  dz],
            [ dx,  dy,  dz]
        ]).T # Shape becomes (3, 8)
        
        # 2. Rotate and Translate into the Global Map Space
        corners_global = rot @ corners_local + trans
        
        # 3. Define the 12 edges connecting the 8 corners
        edges = [
            (0,1), (0,2), (0,4), (1,3), (1,5), (2,3),
            (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)
        ]
        
        # 4. Plot the edges as lime green lines
        for idx1, idx2 in edges:
            p1 = corners_global[:, idx1]
            p2 = corners_global[:, idx2]
            # Use real-world meter coordinates
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                    color='lime', alpha=0.8, linewidth=2)



def generate_random_city(bounds=(100, 100, 15), num_buildings=30, max_building_size=(15, 15, 15)):
    """
    Generates a list of random continuous bounding boxes mimicking a cityscape.
    Returns a list of dummy obstacle objects that your SparseVoxelGrid can read.
    """
    class DummyObstacle:
        def __init__(self, min_bounds, max_bounds):
            # Format to match what your voxel_grid.py expects: a 3xN array of vertices
            self.vertices_shifted_worldFrame_3D = np.array([
                [min_bounds[0], max_bounds[0]], 
                [min_bounds[1], max_bounds[1]], 
                [min_bounds[2], max_bounds[2]]
            ])

    obstacles = []
    for _ in range(num_buildings):
        # Randomly pick the lower-left-bottom corner of the building
        min_x = np.random.uniform(0, bounds[0] - max_building_size[0])
        min_y = np.random.uniform(0, bounds[1] - max_building_size[1])
        min_z = 0 # Buildings start on the ground
        
        # Randomly pick the building's size
        width = np.random.uniform(5, max_building_size[0])
        depth = np.random.uniform(5, max_building_size[1])
        height = np.random.uniform(5, max_building_size[2])
        
        obstacles.append(DummyObstacle(
            min_bounds=(min_x, min_y, min_z),
            max_bounds=(min_x + width, min_y + depth, height)
        ))
        
    return obstacles

#creates the list of points for each side
sideLists = [[0,1,2,3,0],#-75 3
             [0,3,7,4,0],#-30 1
             [0,1,5,4,0],#-75 2
             [4,5,6,7,4],#75 3
             [1,2,6,5,1],#730 1
             [2,3,7,6,2]]#-75 2

def visualize_random_city(obstacles, control_points, degree, knots, safeFlightCorridors_list):
    print("\n--- Rendering Last Benchmark Trial ---")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Draw the Random Buildings
    for obs in obstacles:
        bounds = obs.vertices_shifted_worldFrame_3D
        min_x, max_x = bounds[0][0], bounds[0][1]
        min_y, max_y = bounds[1][0], bounds[1][1]
        min_z, max_z = bounds[2][0], bounds[2][1]

        # Define the 8 vertices of the building
        vertices = np.array([
            [min_x, min_y, min_z], [max_x, min_y, min_z], 
            [max_x, max_y, min_z], [min_x, max_y, min_z],
            [min_x, min_y, max_z], [max_x, min_y, max_z], 
            [max_x, max_y, max_z], [min_x, max_y, max_z]
        ])

        # Define the 6 faces of the building
        faces = [
            [vertices[0], vertices[1], vertices[5], vertices[4]], # Front
            [vertices[1], vertices[2], vertices[6], vertices[5]], # Right
            [vertices[2], vertices[3], vertices[7], vertices[6]], # Back
            [vertices[3], vertices[0], vertices[4], vertices[7]], # Left
            [vertices[4], vertices[5], vertices[6], vertices[7]], # Top
            [vertices[0], vertices[1], vertices[2], vertices[3]]  # Bottom
        ]
        
        # Add the faces to the plot as semi-transparent blocks
        poly3d = Poly3DCollection(faces, facecolors='red', linewidths=1, edgecolors='darkred', alpha=0.15)
        ax.add_collection3d(poly3d)

    # 2. Draw the B-Spline Trajectory
    if control_points is not None:
        pts = control_points.T
        spline = BSpline(knots, pts, degree)
        t_smooth = np.linspace(knots[degree], knots[-degree-1], 200)
        curve = spline(t_smooth)
        
        # Plot the smooth flight path
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], 'b-', linewidth=3, label='Optimized Flight Path')
        
        # Plot the control points
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'ko--', alpha=0.3, markersize=4, label='Control Polygon')
        
        # Mark Start and End
        ax.scatter(*pts[0], c='green', s=150, marker='*', label='Start')
        ax.scatter(*pts[-1], c='purple', s=150, marker='*', label='Goal')

    # 3. Draw SFCs

    for safeFlightCorridor in safeFlightCorridors_list:
        verticesArray = safeFlightCorridor.sfc.getAllVertices_3D()
        numVertices = np.shape(verticesArray)[1]

        verticesList = [verticesArray[:,i:(i+1)] for i in range(numVertices)]

        #creates all of the sides
        sideVertexLists = []

        for side in sideLists:

            tempSide = []

            for index in side:

                tempVertex = verticesList[index]
                tempSide.append(tempVertex)

            sideVerticesArray = np.concatenate((tempSide), axis=1)

            #gets them rotated into the altitude frame
            sideVerticesArray_rotated = PLOT.R_NED_to_Altitude @ sideVerticesArray

            #plots the side
            # x_component = sideVerticesArray_rotated[0,:]
            # y_component = sideVerticesArray_rotated[1,:]
            # z_component = sideVerticesArray_rotated[2,:]

            x_component = sideVerticesArray[0,:]
            y_component = sideVerticesArray[1,:]
            z_component = sideVerticesArray[2,:]

            #plots this side out
            ax.plot(
                x_component,
                y_component,
                z_component,
                color="purple",
                linewidth=2,
                zorder=1,
            )

    # Set equal aspect ratio
    ax.set_box_aspect(BLOCKS.aspect_ratio)

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Altitude (m)')
    ax.set_xlim(BLOCKS.x_limits)
    ax.set_ylim(BLOCKS.y_limits)
    ax.set_zlim([0, 20])
    ax.view_init(elev=0.0,azim=0.0)
    ax.legend()
    
    plt.title("Randomized Benchmark Environment", fontsize=14, fontweight='bold')
    plt.show()



def run_benchmark_suite(num_trials=100):
    print(f"Starting Benchmark Suite: {num_trials} Randomized Environments...")
    
    results = []
    start_pos = np.array([0.0, 0.0, 0.0])
    end_pos = np.array([100.0, 100.0, 5.0])
    
    for i in range(num_trials):
        print(f"\n--- Running Trial {i+1}/{num_trials} ---")
        
        # 1. Generate the random environment (Using the helper from earlier)
        random_obstacles = generate_random_city()

        spline_type="clamped"

        sfc_height = 5.
        sfc_width = 5.

        sfc_start_ext = 2.
        sfc_end_ext = 2.
        
        # 2. Setup your Planner dynamically
        from trajectory_planner import TrajectoryPlanner
        planner = TrajectoryPlanner(map_config="RANDOM", map_bounds=(100.,100.,15.), spline_type=spline_type, 
                                    sfc_height =sfc_height , sfc_width =sfc_width , 
                                    sfc_start_ext =sfc_start_ext , sfc_end_ext =sfc_end_ext )
        
        # Inject our random map into your existing grid logic
        planner.front_end.discrete_grid.occupied_voxels_inflated.clear()
        planner.front_end.discrete_grid.continuous_inflated_bounds.clear()
        planner.front_end.discrete_grid.populate_from_continuous(
            obstacles=random_obstacles, 
            inflation_radius=planner.front_end.grid_inflation_radius # <-- FIXED TYPO
        )

        # --- THE SYNCHRONIZATION FIX ---
        # Update the raw points and rebuild the KD-Tree so the Box Builder can "see" the new random map
        res = planner.front_end.voxel_resolution
        planner.front_end.inflated_obstacle_meters = [
            np.array(idx) * res + (res / 2.0)
            for idx in planner.front_end.discrete_grid.occupied_voxels_inflated
        ]
        planner.front_end.raw_obstacle_meters = [
            np.array(idx) * res + (res / 2.0)
            for idx in planner.front_end.discrete_grid.occupied_voxels_raw
        ]
        
        from planning.dynamic_sfc import AsymmetricSFCManager
        max_cp_drift = (planner.front_end.sfc_width / 2.0) - planner.front_end.drone_physical_radius
        planner.front_end.sfc_manager = AsymmetricSFCManager(
            raw_uninflated_obstacle_points=planner.front_end.inflated_obstacle_meters, 
            drone_physical_radius=0.0,  
            max_cp_drift=max_cp_drift,
            voxel_resolution=res
        )

        # --- RING BUFFER SYNC ---
        # Rebuild the ring buffer so A* search and the LoS smoother see the new random obstacles
        from mapping.ring_buffer import RingBufferGrid
        size_x = int(np.ceil(100.0 / res)) + 1
        size_y = int(np.ceil(100.0 / res)) + 1
        size_z = int(np.ceil(15.0 / res)) + 1
        planner.front_end.discrete_grid.ring_buffer = RingBufferGrid(size_x, size_y, size_z)
        for (vx, vy, vz) in planner.front_end.discrete_grid.occupied_voxels_inflated:
            planner.front_end.discrete_grid.ring_buffer.set_occupied(vx, vy, vz)

        trial_data = {
            "trial_id": i,
            "success": False,
            "failure_reason": "None"
            # We will dynamically add the time metrics here!
        }

        # 3. Run the ENTIRE pipeline in one shot
        try:
            control_points, _, metrics = planner.plan_mission(start_pos, end_pos)
            
            # Save the times (A*, OSQP, Overhead, and Total Pipeline Time)
            trial_data.update(metrics)
            
            if control_points is not None:
                trial_data["success"] = True
            else:
                trial_data["failure_reason"] = "OSQP Kinodynamic Failure"
                
        except Exception as e:
            # If A* fails to find a path, your FrontEndSFC currently returns None, None, None, None
            # which will cause plan_mission to crash when it tries to unpack them. We catch that here!
            trial_data["failure_reason"] = f"Crash/No Path: {str(e)}"
             
        results.append(trial_data)

        # --- VISUALIZE EVERY ITERATION FOR DEBUGGING ---
        print(f"\n[Visual Check] Rendering Trial {i+1} Environment & SFCs...")
        
        knots = None
        if control_points is not None:
            # Dynamically import the correct evaluator if optimization succeeded
            if planner.spline_type == "natural":
                from min_snap_natural import MinSnapEvalNatural as SplineEvaluator
            else:
                from min_snap_clamped import MinSnapEvalClamped as SplineEvaluator

            knots = SplineEvaluator(
                num_segments=len(control_points[0]) - planner.degree, 
                degree=planner.degree
            ).knots
            
        # Safely fetch the corridors (returns empty list if A* crashed completely)
        corridors_to_plot = getattr(planner.front_end, 'last_corridors', [])
        
        visualize_random_city(
            obstacles=random_obstacles, 
            control_points=control_points, 
            degree=planner.degree, 
            knots=knots,
            safeFlightCorridors_list=corridors_to_plot
        )
        
    # 4. Export the Data
    df = pd.DataFrame(results)
    df.to_csv("trajectory_benchmarks.csv", index=False)
    
    print("\n=================================")
    print("      BENCHMARK COMPLETE         ")
    print("=================================")
    print(f"Total Trials: {num_trials}")
    print(f"Overall Success Rate: {df['success'].mean() * 100:.2f}%")
    
    # Only calculate average times for successful flights!
    success_df = df[df['success'] == True]
    if not success_df.empty:
        print(f"Avg A* Time:         {success_df['astar_time_ms'].mean():.2f} ms")
        print(f"Avg OSQP Time:       {success_df['osqp_time_ms'].mean():.2f} ms")
        print(f"Avg Total Pipeline:  {success_df['total_pipeline_ms'].mean():.2f} ms")

    if trial_data["success"]:
        # Dynamically import the correct evaluator based on the planner's active architecture
        if planner.spline_type == "natural":
            from min_snap_natural import MinSnapEvalNatural as SplineEvaluator
        else:
            from min_snap_clamped import MinSnapEvalClamped as SplineEvaluator

        # Instantiate the correct class to calculate the knot sequence
        knots = SplineEvaluator(
            num_segments=len(control_points[0]) - planner.degree, 
            degree=planner.degree
        ).knots
        
        visualize_random_city(
            obstacles=random_obstacles, 
            control_points=control_points, 
            degree=planner.degree, 
            knots=knots,
            safeFlightCorridors_list=planner.front_end.last_corridors
        )

        
    

def run_parameter_sweep(num_trials=50):
    print(f"Starting Parameter Sweep: {num_trials} Randomized Environments...")
    results = []
    start_pos = np.array([0.0, 0.0, 0.0])
    end_pos = np.array([100.0, 100.0, 5.0]) # Keeping the low-altitude constraint!
    
    # Define our 3 test profiles
    test_profiles = [
        {"profile_name": "5.0m SFC", "sfc_width": 5.0, "voxel_res": 1.0},
        {"profile_name": "3.0m SFC", "sfc_width": 3.0, "voxel_res": 1.0},
        {"profile_name": "1.7m SFC", "sfc_width": 1.7, "voxel_res": 0.5}
    ]
    
    for i in range(num_trials):
        print(f"\n--- Running Trial {i+1}/{num_trials} ---")
        
        # 1. Generate ONE random environment for all 3 profiles to solve
        # (30 buildings to guarantee the "urban canyon" effect)
        random_obstacles = generate_random_city(num_buildings=30)
        
        for profile in test_profiles:
            print(f"  -> Testing Profile: {profile['profile_name']}")
            
            from trajectory_planner import TrajectoryPlanner
            from mapping.voxel_grid import SparseVoxelGrid
            from planning.astar_sfc import AStar_SFC_Planner
            
            planner = TrajectoryPlanner(map_config="RANDOM")
            
            # --- DYNAMIC PARAMETER INJECTION ---
            planner.front_end.voxel_resolution = profile["voxel_res"]
            planner.front_end.inflation_radius = profile["sfc_width"] / 2.0
            
            # Rebuild the Grid and A* Engine with the new parameters
            planner.front_end.discrete_grid = SparseVoxelGrid(resolution=profile["voxel_res"])
            
            # We must pass the bounds again so A* calculates the new max_indices correctly
            bounds = [(0.0, 100.0), (0.0, 100.0), (0.0, 15.0)]
            planner.front_end.path_gen_astar = AStar_SFC_Planner(planner.front_end.discrete_grid, bounds)
            
            # Populate the buildings
            planner.front_end.discrete_grid.populate_from_continuous(
                obstacles=random_obstacles, 
                inflation_radius=planner.front_end.inflation_radius
            )

            trial_data = {
                "trial_id": i,
                "profile": profile["profile_name"],
                "success": False,
                "failure_reason": "None"
            }

            # 3. Run the Pipeline
            try:
                control_points, _, metrics = planner.plan_mission(start_pos, end_pos)
                trial_data.update(metrics)
                
                if control_points is not None:
                    trial_data["success"] = True
                else:
                    trial_data["failure_reason"] = "OSQP Kinodynamic Failure"
            except Exception as e:
                trial_data["failure_reason"] = f"Crash/No Path: {str(e)}"
                 
            results.append(trial_data)
        
    # Export the Data
    df = pd.DataFrame(results)
    df.to_csv("parameter_sweep_results.csv", index=False)
    
    # Print a nice summary table
    print("\n==============================================")
    print("           PARAMETER SWEEP COMPLETE           ")
    print("==============================================")
    for profile in test_profiles:
        prof_name = profile['profile_name']
        prof_data = df[df['profile'] == prof_name]
        success_rate = prof_data['success'].mean() * 100
        
        successful_runs = prof_data[prof_data['success'] == True]
        avg_total = successful_runs['total_pipeline_ms'].mean() if not successful_runs.empty else 0
        
        print(f"Profile: {prof_name:<10} | Success: {success_rate:>6.2f}% | Avg Time: {avg_total:>6.2f} ms")


def run_batch_performance_test_clamped():
        # The number of random trajectories we want to compute in each batch
        batch_sizes = [10, 100, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
        
        execution_times = []
        
        # Static physics parameters
        snap_degree = 4
        snap_ctrl_pts = 11

        print("\nPre-computing Q Matrix once for all batches...")
        evaluator = MinSnapEvalClamped(snap_ctrl_pts, snap_degree)
        Q_d4_M = evaluator.get_Q_matrix()
        
        print("\nRunning Batch Execution Test...")
        
        for num_trajectories in batch_sizes:
            print(f"Calculating {num_trajectories:,} random trajectories...")
            
            # --- START TIMER ---
            start_exec = time.perf_counter()
            
            # Simulate the drone rapidly calculating new paths
            for _ in range(num_trajectories):
                # Generate random physical states
                p0 = np.random.rand(3, 1) * 10 
                v0 = np.random.rand(3, 1) * 5 - 2.5
                a0 = np.random.rand(3, 1) * 2 - 1
                
                pf = np.random.rand(3, 1) * 10 
                vf = np.random.rand(3, 1) * 5 - 2.5
                af = np.random.rand(3, 1) * 2 - 1
                
                A_p = np.hstack((p0, v0, a0, af, vf, pf))
                
                # The core calculation
                C_p = A_p @ Q_d4_M
                
            # --- STOP TIMER ---
            end_exec = time.perf_counter()
            
            total_time = end_exec - start_exec
            execution_times.append(total_time)

        # ==========================================
        # PLOT THE RESULTS
        # ==========================================
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(batch_sizes, execution_times, 'g-o', linewidth=2, markersize=6)
        
        # Format the graph
        ax.set_title('Batch Processing Time for Clamped Uniform Minimum Snap Trajectories', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Trajectories Computed', fontsize=12)
        ax.set_ylabel('Total Computation Time (seconds)', fontsize=12)
        
        # Use a standard decimal format for the X-axis instead of scientific notation
        ax.ticklabel_format(style='plain', axis='x')
        
        # Add a grid and start axes at 0
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.show()

def run_clamped_performance_benchmark(max_control_points=100, iterations=1000, degree=4):
    """
    Benchmarks the setup and execution time of the CLAMPED Minimum Snap solver.
    """
    print(f"🚀 Starting Clamped Benchmark: 7 to {max_control_points} Control Points")
    print(f"   Running {iterations} random trajectories per number of Control Points...\n")
    
    # Start at 7 points due to the 6 boundary constraints
    ctrl_pts_range = range(7, max_control_points + 1, 2)
    
    setup_times_ms = []
    exec_times_us = []
    
    for num_pts in ctrl_pts_range:
        # ==========================================
        # 1. MEASURE SETUP TIME (SVD & Solver)
        # ==========================================
        start_setup = time.perf_counter()
        
        # ---> CHANGE THIS TO YOUR ACTUAL CLAMPED CLASS NAME <---
        evaluator = MinSnapEvalClamped(num_pts, degree) 
        Q_matrix = evaluator.get_Q_matrix()
        
        end_setup = time.perf_counter()
        
        # Convert to milliseconds
        setup_times_ms.append((end_setup - start_setup) * 1000)
        
        # ==========================================
        # 2. PRE-GENERATE RANDOM STATES
        # ==========================================

        start_exec = time.perf_counter()
        for _ in range(iterations):
            # p0 = np.random.rand(3, 1) * 10 
            # v0 = np.random.rand(3, 1) * 5 - 2.5
            # a0 = np.random.rand(3, 1) * 2 - 1
            # pf = np.random.rand(3, 1) * 10 
            # vf = np.random.rand(3, 1) * 5 - 2.5
            # af = np.random.rand(3, 1) * 2 - 1

            p0 = np.array([[0],[0],[0]])
            v0 = np.array([[0],[0],[0]])
            a0 = np.array([[0],[0],[0]])
            pf = np.array([[10],[10],[10]])
            vf = np.array([[0],[0],[0]])
            af = np.array([[0],[0],[0]])

            A_p = np.hstack((p0, v0, a0, af, vf, pf))
            C_optimal = A_p @ Q_matrix 
            
        end_exec = time.perf_counter()
        
        # Calculate average time per trajectory in MICROSECONDS
        total_exec_time = end_exec - start_exec
        avg_exec_us = (total_exec_time / iterations) * 1_000_000
        exec_times_us.append(avg_exec_us)
        
        print(f"Pts: {num_pts:3d} | Setup: {setup_times_ms[-1]:8.2f} ms | Exec: {avg_exec_us:6.3f} µs")

    # ==========================================
    # 4. PLOT THE RESULTS
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Setup Time
    ax1.plot(ctrl_pts_range, setup_times_ms, 'r-o', linewidth=2)
    ax1.set_title('Clamped Boot-up Time (Q Matrix Generation)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Number of Control Points')
    ax1.set_ylabel('Time (Milliseconds)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Execution Time
    ax2.plot(ctrl_pts_range, exec_times_us, 'b-o', linewidth=2)
    ax2.set_title('Clamped Real-Time Execution (A @ Q)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Number of Control Points')
    ax2.set_ylabel('Time (Microseconds)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    
    plt.suptitle('Clamped Minimum Snap Performance Scaling (1000 Iterations/Number of Control Points)', fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_benchmark_suite(num_trials=10)
    # run_parameter_sweep(100)