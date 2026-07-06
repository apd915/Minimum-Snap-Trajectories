import numpy as np
import random
import time

np.random.seed(42)
random.seed(42)

import matplotlib.pyplot as plt
from rrt_mavsim.viewers.plot_map_path import PlotMapPath
import rrt_mavsim.parameters.floatingBlocks_parameters as FLOATING_PARAM
from front_end import FrontEndSFC
from min_snap_natural import MinSnapEvalNatural
from min_snap_clamped import MinSnapEvalClamped
from core.optimize import run_qp_solver

def print_sfc_diagnostics(corridors):
    """
    Translates SFC bounds into physical dimensions and detects geometric anomalies.
    """
    print("\n" + "="*60)
    print("                 SFC GEOMETRY DIAGNOSTICS")
    print("="*60)
    
    for i, sfc in enumerate(corridors):
        pA = sfc.primaryPosition.flatten()
        pB = sfc.secondaryPosition.flatten()
        
        # 1. Physical Dimensions (Calculated from the bounds array)
        b = sfc.bounds
        L = b[0] + b[1]
        W = b[2] + b[3]
        H = b[4] + b[5]
        
        print(f"\n[ SFC {i} ]")
        print(f"Segment:    A: {np.round(pA, 2)}  -->  B: {np.round(pB, 2)}")
        print(f"Dimensions: Length: {L:>5.2f}m | Width: {W:>5.2f}m | Height: {H:>5.2f}m")
        
        # 2. Raw Constraint Bounds (Distance from the A* node to the wall)
        print(f"Bounds:     +X (Fwd): {b[0]:>5.2f}m   | -X (Back): {b[1]:>5.2f}m")
        print(f"            +Y (Lft): {b[2]:>5.2f}m   | -Y (Rgt):  {b[3]:>5.2f}m")
        print(f"            +Z (Top): {b[4]:>5.2f}m   | -Z (Btm):  {b[5]:>5.2f}m")
        
        # 3. 3D Corners (Extracting max/min global boundaries)
        x_corners, y_corners, z_corners = sfc.getAllVertices_3D()
        z_max = max(z_corners)
        z_min = min(z_corners)
        print(f"Corners:    Global Z-Max: {z_max:>5.2f}m | Global Z-Min: {z_min:>5.2f}m")
        
        # 4. Anomaly Detection!
        if W <= 0.0:
            print("  >>> [FATAL] LATERAL WALLS COLLAPSED! (Negative Width) <<<")
        if H <= 0.0:
            print("  >>> [FATAL] CEILING CRUSHED FLOOR! (Negative Height) <<<")
        if L <= 0.0:
            print("  >>> [FATAL] X-AXIS TRUNCATED TO ZERO! (Negative Length) <<<")
            
    print("="*60 + "\n")

class TrajectoryPlanner:
    def __init__(self, map_config, map_bounds=(100.,100.,15.), v_max=3.0, a_max=2.0, degree=4, spline_type="natural", sfc_height=1., sfc_width=1., sfc_start_ext=5., sfc_end_ext=5.):
        """
        Initializes the master trajectory planner.
        """
        self.degree = degree
        self.v_max = v_max
        self.a_max = a_max
        self.map_config = map_config
        self.spline_type = spline_type
        self.sfc_height = sfc_height
        self.sfc_width = sfc_width
        self.sfc_start_ext = sfc_start_ext
        self.sfc_end_ext = sfc_end_ext
        self.map_bounds=map_bounds
        
        # Instantiate the Front-End
        self.front_end = FrontEndSFC(self.sfc_height, self.sfc_width, self.sfc_start_ext, self.sfc_end_ext, self.spline_type, map_config, degree)


    def plan_mission(self, start_pos, end_pos, start_vel=np.zeros(3), start_acc=np.zeros(3)):
        print("--- Starting Trajectory Planning Mission ---")
        mission_start_time = time.perf_counter()

        print("[Phase 1] Generating Safe Flight Corridors...")
        
        sfc_start_time = time.perf_counter()
        # --- THE FIX: Unpack the Constraint Pools ---
        corridors, constraint_pools, waypoints_smooth, waypoints_not_smooth = self.front_end.get_corridors_astar(start_pos, end_pos)
        sfc_duration = time.perf_counter() - sfc_start_time
        
        if not corridors:
            print("[Error] No valid path found through the environment.")
            return None, None
        
        # print_sfc_diagnostics(corridors)

        # print("\n[Visual Check] Displaying Complete Geometry (Raw Path, Smoothed Path, SFCs)...")
        # print("Close the Matplotlib window to begin OSQP optimization!")
        # self.visualize(waypoints_smooth, waypoints_not_smooth)

        max_stretches = 5
        stretch_count = 0
        optimal_control_points = None
        opt_duration = 0.0

        S = np.hstack((start_pos.reshape(3,1), start_vel.reshape(3,1), start_acc.reshape(3,1))) 

        if self.spline_type == "natural":
            E_standard = np.hstack((end_pos.reshape(3,1), np.zeros((3,1)), np.zeros((3,1))))
            SE = np.hstack((S, E_standard))
        elif self.spline_type == "clamped":
            E_reversed = np.hstack((np.zeros((3,1)), np.zeros((3,1)), end_pos.reshape(3,1)))
            SE = np.hstack((S, E_reversed))
        else:
            raise ValueError(f"Unknown spline_type: {self.spline_type}")
        
        while stretch_count <= max_stretches:
            print(f"\n--- Optimization Attempt {stretch_count + 1} ---")
            
            # --- THE FIX: Pass the unified pools to the matrix compiler ---
            A_sfc, b_sfc, total_control_points = self.front_end.compile_system_constraints(corridors, constraint_pools)
            
            print(f"[Phase 2] Translating {len(corridors)} SFCs for {total_control_points} points...")

            opt_start_time = time.perf_counter()
            num_segments = total_control_points - self.degree
            
            if self.spline_type == "natural":
                optimizer = MinSnapEvalNatural(num_segments=num_segments, degree=self.degree)
                D_vel = optimizer._get_fast_cascaded_D_matrix(num_segments, self.degree, 1).T
                D_accel = optimizer._get_fast_cascaded_D_matrix(num_segments, self.degree, 2).T
                SE_qp = SE
            else:
                optimizer = MinSnapEvalClamped(num_segments=num_segments, degree=self.degree)
                D_vel = optimizer._get_fast_cascaded_D_matrix(num_segments, self.degree, 1)
                D_accel = optimizer._get_fast_cascaded_D_matrix(num_segments, self.degree, 2)
                B_d3 = optimizer._get_B_d3_matrix(self.degree)
                SE_qp = SE @ B_d3

            W = optimizer.W
            A_eq = optimizer.B_combined.T

            print(f"[Phase 3] Running OSQP Solver...")
            try:
                optimal_control_points = run_qp_solver(
                    objective_matrix=W,
                    equality_constraints=SE_qp,
                    inequality_constraints=(D_vel, D_accel, self.v_max, self.a_max, A_sfc, b_sfc), 
                    A_eq=A_eq,
                    degree=self.degree,
                    spline_type=self.spline_type,
                    use_minvo=True
                )

                opt_duration = time.perf_counter() - opt_start_time 
                
                print(f"[Phase 3] Optimization Successful in {opt_duration:.4f}s!")
                break  
                
            except Exception as e:
                print(f"[Phase 4] Solver failed (Kinematically impossible): {e}")
                
                if stretch_count < max_stretches:
                    print("[Phase 4] Stretching time allocation (+points to straights and intersections)...")
                    # --- THE FIX: Stretch the Unified Pools ---
                    for pool in constraint_pools:
                        if pool['type'] == 'exclusive':
                            pool['pts'] += self.degree
                        else:
                            pool['pts'] += 2
                else:
                    print("[Error] Max stretching attempts reached.")
                
                stretch_count += 1

        total_time = time.perf_counter() - mission_start_time
        overhead_duration = total_time - (sfc_duration + opt_duration)

        print("\n" + "="*50)
        print("          TRAJECTORY PLANNER BENCHMARKS")
        print("="*50)
        print(f"SFC Generation (Front-End):   {sfc_duration * 1000:.2f} ms")
        print(f"Path Generation (Back-End):   {opt_duration * 1000:.2f} ms")
        print(f"Matrix & System Overhead:     {overhead_duration * 1000:.2f} ms")
        print("-" * 50)
        print(f"TOTAL PLANNING TIME:          {total_time * 1000:.2f} ms")
        print("="*50 + "\n")

        metrics = {
            "astar_time_ms": sfc_duration * 1000.0,
            "osqp_time_ms": opt_duration * 1000.0,
            "overhead_ms": overhead_duration * 1000.0,
            "total_pipeline_ms": total_time * 1000.0
        }
        
        return optimal_control_points, waypoints_smooth, metrics
    

    def _build_overlap_constraints(self, sfc_constraints, num_pts_list, total_num_points):
        """
        Converts Dean's A/b matrices and point allocations into a single massive 
        A and b matrix pair for the QP solver, overlapping the indices to 
        mathematically force the spline through the intersections.
        """
        A_ineq_list = []
        b_ineq_list = []
        start_idx = 0 
        num_dimensions = 3 
        
        # 1. Unpack the dynamic class variable
        north_end, east_end, alt_end = self.map_bounds

        # ---------------------------------------------------------
        # NEW: Define the absolute map boundaries (100x100x15)
        # Hyperplanes: [+x, -x, +y, -y, +z, -z]
        # ---------------------------------------------------------
        A_map = np.array([
            [ 1.0,  0.0,  0.0], # +x (North)
            [-1.0,  0.0,  0.0], # -x (South)
            [ 0.0,  1.0,  0.0], # +y (East)
            [ 0.0, -1.0,  0.0], # -y (West)
            [ 0.0,  0.0,  1.0], # +z (Up/Alt)
            [ 0.0,  0.0, -1.0]  # -z (Down/Ground)
        ])
        b_map = np.array([north_end, 0.0, east_end, 0.0, alt_end, 0.0])
        
        for i, sfc in enumerate(sfc_constraints):
            # ---------------------------------------------------------
            # NEW: Intersect the SFC with the Map Bounding Box
            # ---------------------------------------------------------
            A_mat = np.vstack((sfc['A'], A_map))
            b_vec = np.concatenate((np.array(sfc['b']).flatten(), b_map))

            # ---------------------------------------------------------
            # VIRTUAL RUNWAY LOGIC
            # ---------------------------------------------------------
            if getattr(self, 'spline_type', 'natural') == "natural":
                # Only apply to the takeoff and landing corridors
                if i == 0 or i == len(sfc_constraints) - 1:
                    runway_length = 30.0
                    
                    # Scan every wall in the combined matrix
                    for k in range(len(b_vec)):
                        normal = A_mat[k]
                        val = b_vec[k]
                        
                        # Relax X Map Boundaries
                        if np.allclose(normal, [1, 0, 0]) and np.isclose(val, north_end, atol=1e-2):
                            b_vec[k] += runway_length
                        elif np.allclose(normal, [-1, 0, 0]) and np.isclose(val, 0.0, atol=1e-2):
                            b_vec[k] += runway_length
                            
                        # Relax Y Map Boundaries
                        elif np.allclose(normal, [0, 1, 0]) and np.isclose(val, east_end, atol=1e-2):
                            b_vec[k] += runway_length
                        elif np.allclose(normal, [0, -1, 0]) and np.isclose(val, 0.0, atol=1e-2):
                            b_vec[k] += runway_length
                            
                        # Relax Z Map Boundaries (Ceiling and Floor)
                        elif np.allclose(normal, [0, 0, 1]) and np.isclose(val, alt_end, atol=1e-2):
                            b_vec[k] += runway_length
                        elif np.allclose(normal, [0, 0, -1]) and np.isclose(val, 0.0, atol=1e-2):
                            b_vec[k] += runway_length
            # ---------------------------------------------------------
            
            num_pts_in_box = num_pts_list[i]
            # This automatically adjusts to the new size (original + 6 map walls)
            num_inequalities = A_mat.shape[0] 
            
            for j in range(num_pts_in_box):
                global_cp_index = start_idx + j
                A_padded = np.zeros((num_inequalities, total_num_points * num_dimensions))
                
                col_start = global_cp_index * num_dimensions
                col_end = col_start + num_dimensions
                A_padded[:, col_start:col_end] = A_mat
                
                A_ineq_list.append(A_padded)
                b_ineq_list.append(b_vec)
                
            start_idx += (num_pts_in_box - self.degree)
            
        A_sfc_total = np.vstack(A_ineq_list)
        b_sfc_total = np.concatenate(b_ineq_list)
        
        return A_sfc_total, b_sfc_total
    

    def visualize(self, waypoints_smooth, waypoints_not_smooth=None, optimal_control_points=None):
        """
        Renders the 3D environment, the glass boxes, and (optionally) the B-spline.
        Code execution will PAUSE until you close the plot window.
        """
        import matplotlib.pyplot as plt
        from rrt_mavsim.viewers.plot_map_path import PlotMapPath
        import rrt_mavsim.parameters.floatingBlocks_parameters as FLOATING_PARAM

        print("--- Rendering Visualization ---")
        
        # Only pass the list if we actually have control points to plot
        cp_list = [optimal_control_points] if optimal_control_points is not None else None
        
        plotter = PlotMapPath(
            map=self.front_end.worldMap,
            waypoints_not_smooth=waypoints_not_smooth,
            waypoints_smooth=waypoints_smooth,
            controlPoints_not_smooth_list=None,
            controlPoints_smooth_list=cp_list,
        )

        plotter.plot(
            x_limits=FLOATING_PARAM.x_limits,
            y_limits=FLOATING_PARAM.y_limits,
            z_limits=FLOATING_PARAM.z_limits,
            aspectRatio=FLOATING_PARAM.aspect_ratio,
        )
        
        # This will block the code from continuing until you close the window!
        plt.show()

# ==========================================
# Execution Test
# ==========================================
if __name__ == "__main__":
    mock_map = "FLOATING_BLOCKS"

    from rrt_mavsim.parameters import floatingBlocks_parameters as FLOATING_PARAM
    northEnd = FLOATING_PARAM.northEnd
    eastEnd = FLOATING_PARAM.eastEnd
    altitudeEnd = -FLOATING_PARAM.altitudeEnd
    downEnd = FLOATING_PARAM.downEnd

    map_bounds = (northEnd, eastEnd, downEnd)

    startPosition_3D = np.array([[2.0],[2.0],[5.0]])
    endPosition_3D = np.array([[northEnd-5],[eastEnd-5],[downEnd]])

    start = startPosition_3D
    goal = FLOATING_PARAM.endPosition_3D

    spline_type="clamped"

    sfc_height = 10.
    sfc_width = 10.

    sfc_start_ext = 5.
    sfc_end_ext = 5.

    planner = TrajectoryPlanner(map_config=mock_map, map_bounds=map_bounds, 
                                spline_type=spline_type, sfc_height=sfc_height, 
                                sfc_width=sfc_width, sfc_start_ext=sfc_start_ext, 
                                sfc_end_ext=sfc_end_ext)
    controlPointsList, waypoints_smooth, _ = planner.plan_mission(start, goal)


    from rrt_mavsim.message_types.msg_world_map import MsgWorldMap, FloatingBlocksParams, MapTypes
    worldMap = MsgWorldMap(
            obstacleFieldType=MapTypes.FLOATING_BLOCKS,
            numDimensions_algorithm=FLOATING_PARAM.numDimensions,
            floatingBlocksParams=FloatingBlocksParams()
        )
    
    # --- THE FIX: Only pass the list if the optimizer actually succeeded! ---
    valid_control_points = [controlPointsList] if controlPointsList is not None else None
    
    plotter = PlotMapPath(
        map=worldMap,
        waypoints_smooth=waypoints_smooth,
        controlPoints_not_smooth_list=None,
        controlPoints_smooth_list=valid_control_points,
    )

    plotter.plot(
        x_limits=FLOATING_PARAM.x_limits,
        y_limits=FLOATING_PARAM.y_limits,
        z_limits=FLOATING_PARAM.z_limits,
        aspectRatio=FLOATING_PARAM.aspect_ratio,
    )

    print("[Matplotlib] Rendering 3D Polygons...")
    
    plt.show()