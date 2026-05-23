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

class TrajectoryPlanner:
    def __init__(self, map_config, v_max=3.0, a_max=2.0, degree=4, spline_type="natural"):
        """
        Initializes the master trajectory planner.
        """
        self.degree = degree
        self.v_max = v_max
        self.a_max = a_max
        self.map_config = map_config
        self.spline_type = spline_type
        
        # Instantiate the Front-End
        self.front_end = FrontEndSFC(map_config, degree)


    def plan_mission(self, start_pos, end_pos, start_vel=np.zeros(3), start_acc=np.zeros(3)):
        """
        The main orchestration loop with Dynamic Time Stretching.
        """
        print("--- Starting Trajectory Planning Mission ---")
        mission_start_time = time.perf_counter()

        # =========================================================
        # PHASE 1: The Front-End 
        # =========================================================
        print("[Phase 1] Generating Safe Flight Corridors...")
        
        sfc_start_time = time.perf_counter()
        sfc_constraints, num_pts_list, waypoints_smooth, waypoints_not_smooth = self.front_end.get_corridors_astar(start_pos, end_pos)
        sfc_duration = time.perf_counter() - sfc_start_time
        
        # Override the point allocation! 
        # 0.003 points per meter * 5000 meters = ~15 control points total
        # custom_point_density = 0.003 
        
        # sfc_constraints, num_pts_list, waypoints_smooth, waypoints_not_smooth = self.front_end.get_corridors(
        #     start_pos, 
        #     end_pos, 
        #     num_points_per_unit=custom_point_density
        # )
        
        if not sfc_constraints:
            print("[Error] No valid path found through the environment.")
            return None, None

        # [Visual Check] - Uncomment to pause and view SFCs before solving
        # print("[Visual Check] Displaying SFCs. Close the plot window to begin optimization...")
        # self.visualize(waypoints_smooth, waypoints_not_smooth)

        # =========================================================
        # PHASE 2 & 3 & 4: The Optimization and Stretching Loop
        # =========================================================
        max_stretches = 5
        stretch_count = 0
        optimal_control_points = None
        opt_duration = 0.0

        # 1. Format initial states for the boundary matrices
        S = np.hstack((start_pos.reshape(3,1), start_vel.reshape(3,1), start_acc.reshape(3,1))) 

        # 2. Route the Boundary Logic
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
            
            # 1. Calculate points and build matrices
            total_control_points = sum(num_pts_list) - self.degree * (len(num_pts_list) - 1)
            print(f"[Phase 2] Translating {len(sfc_constraints)} SFCs for {total_control_points} points...")
            A_sfc, b_sfc = self._build_overlap_constraints(sfc_constraints, num_pts_list, total_control_points)

            # 2. Initialize Backend Math
            opt_start_time = time.perf_counter()
            num_segments = total_control_points - self.degree
            
            if self.spline_type == "natural":
                optimizer = MinSnapEvalNatural(num_segments=num_segments, degree=self.degree)

                # Natural natively returns transposed blocks, so we MUST flip them for OSQP
                D_vel = optimizer._get_fast_cascaded_D_matrix(num_segments, self.degree, 1).T
                D_accel = optimizer._get_fast_cascaded_D_matrix(num_segments, self.degree, 2).T
                SE_qp = SE
            else:
                optimizer = MinSnapEvalClamped(num_segments=num_segments, degree=self.degree)

                # Clamped natively returns the correct shape. DO NOT flip them!
                D_vel = optimizer._get_fast_cascaded_D_matrix(num_segments, self.degree, 1)
                D_accel = optimizer._get_fast_cascaded_D_matrix(num_segments, self.degree, 2)
                
                # Clamped requires mapping the boundary targets through B^{d,3}
                B_d3 = optimizer._get_B_d3_matrix(self.degree)
                SE_qp = SE @ B_d3

            W = optimizer.get_W_matrix()
            Q = optimizer.Q
            A_eq = optimizer.B_combined.T

            # 3. Initial Guess
            C_p_guess = SE @ Q

            # 4. Run the QP Solver
            print(f"[Phase 3] Running OSQP Solver...")
            try:
                optimal_control_points = run_qp_solver(
                    objective_matrix=W,
                    equality_constraints=SE_qp,
                    inequality_constraints=(D_vel, D_accel, self.v_max, self.a_max, A_sfc, b_sfc), 
                    initial_guess=C_p_guess,
                    A_eq=A_eq,
                    degree=self.degree,
                    use_minvo=False
                )

                opt_duration = time.perf_counter() - opt_start_time 
                
                print(f"[Phase 3] Optimization Successful in {opt_duration:.4f}s!")
                break  # Exit the while loop!
                
            except Exception as e:
                # =========================================================
                # PHASE 4: Kinodynamic Check & Time Stretching
                # =========================================================
                print(f"[Phase 4] Solver failed (Kinematically impossible): {e}")
                
                if stretch_count < max_stretches:
                    print("[Phase 4] Stretching time allocation (+1 point to all SFCs)...")
                    num_pts_list = [pts + 1 for pts in num_pts_list]
                else:
                    print("[Error] Max stretching attempts reached.")
                
                stretch_count += 1

        total_time = time.perf_counter() - mission_start_time

        # Calculate overhead (matrix formatting, RRT overhead, etc.)
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

        # --- NEW: Package the metrics for the benchmarking script ---
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
        
        for i, sfc in enumerate(sfc_constraints):
            A_mat = sfc['A'] 
            b_vec = np.array(sfc['b']).flatten() 
            
            num_pts_in_box = num_pts_list[i]
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
    downEnd = -FLOATING_PARAM.downEnd

    start = FLOATING_PARAM.startPosition_3D
    goal = FLOATING_PARAM.endPosition_3D

    planner = TrajectoryPlanner(map_config=mock_map, spline_type="clamped")
    controlPointsList, waypoints_smooth, _ = planner.plan_mission(start, goal)


    from rrt_mavsim.message_types.msg_world_map import MsgWorldMap, FloatingBlocksParams, MapTypes
    worldMap = MsgWorldMap(
            obstacleFieldType=MapTypes.FLOATING_BLOCKS,
            numDimensions_algorithm=FLOATING_PARAM.numDimensions,
            floatingBlocksParams=FloatingBlocksParams()
        )
    
    plotter = PlotMapPath(
        map=worldMap,
        waypoints_smooth=waypoints_smooth,
        controlPoints_not_smooth_list=None,
        controlPoints_smooth_list=[controlPointsList],
    )

    plotter.plot(
        x_limits=FLOATING_PARAM.x_limits,
        y_limits=FLOATING_PARAM.y_limits,
        z_limits=FLOATING_PARAM.z_limits,
        aspectRatio=FLOATING_PARAM.aspect_ratio,
    )
    plt.show()