import open3d as o3d
import numpy as np
import math
import time
from planning.astar_sfc import AStar_SFC_Planner
from min_snap_clamped import MinSnapEvalClamped
from core.optimize import run_qp_solver

FILE_PATH = "data/vicon_hard 1.ply"  

class VoxelGridMap:
    def __init__(self, resolution, occupied_set):
        self.voxel_resolution = resolution
        self.occupied_voxels_inflated = occupied_set

def inflate_obstacles(occupied_voxels, resolution, inflation_radius_meters=1):
    print(f"Inflating {len(occupied_voxels)} obstacles for physical safety buffer...")
    inflated_set = set()

    # Create a new, distinctly named variable and strictly round UP
    inflation_voxels = math.ceil(inflation_radius_meters / resolution)
    print(f"Calculated voxel padding: {inflation_voxels}")
    
    for (vx, vy, vz) in occupied_voxels:
        for dx in range(-inflation_voxels, inflation_voxels + 1):
            for dy in range(-inflation_voxels, inflation_voxels + 1):
                for dz in range(-inflation_voxels, inflation_voxels + 1):
                    inflated_set.add((vx + dx, vy + dy, vz + dz))
                    
    print(f"Inflation complete! New safe obstacle count: {len(inflated_set)}")
    return inflated_set

def meters_to_grid(coord_meters, voxel_size):
    # Converts (0.0, 0.0, 0.5m) -> Grid Index (0, 0, 2)
    return tuple(int(np.floor(c / voxel_size)) for c in coord_meters)

def grid_to_meters(path_indices, voxel_size):
    # Converts Grid Index (0, 0, 2) back to (0.0, 0.0, 0.5m)
    # We add (voxel_size / 2) to force the trajectory exactly through the CENTER of the safe voxel
    return [(x * voxel_size + (voxel_size / 2), 
             y * voxel_size + (voxel_size / 2), 
             z * voxel_size + (voxel_size / 2)) for (x, y, z) in path_indices]

def get_sfc_open3d_geometries(corridors):
    """
    Translates MsgFlightCorridor objects directly into Open3D 3D wireframes.
    """
    
    sfc_geometries = []
    
    for sfc in corridors:
        # Access the underlying 3D properties natively from your class!
        center = sfc.translation_world.flatten()
        R = sfc.R_SFCToWorld
        extent = sfc.dimensions.flatten()
        
        # 1. Create the Oriented Bounding Box (OBB)
        obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
        
        # 2. Convert to a LineSet to guarantee it renders as a transparent wireframe 
        # (This prevents solid box faces from hiding the green A* trajectory)
        obb_lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
        obb_lines.paint_uniform_color([111/255, 66/255, 245/255]) # Paint it bright purple
        
        sfc_geometries.append(obb_lines)
        
    return sfc_geometries

def build_overlap_constraints(sfc_constraints, num_pts_list, degree, total_num_points):
    """
    Constructs the massive Ax <= b inequality matrix for the QP solver.
    """
    import numpy as np
    A_ineq_list = []
    b_ineq_list = []
    start_idx = 0 
    num_dimensions = 3 
    
    for i, sfc in enumerate(sfc_constraints):
        A_mat = np.array(sfc['A'])
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
            
        start_idx += (num_pts_in_box - degree)
        
    A_sfc_total = np.vstack(A_ineq_list)
    b_sfc_total = np.concatenate(b_ineq_list)
    
    return A_sfc_total, b_sfc_total

def run_planner_and_visualize():
    voxel_size = 0.1
    
    # --- 1. INGESTION ---
    print(f"Loading map from {FILE_PATH}...")
    pcd = o3d.io.read_point_cloud(FILE_PATH)
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    downsampled_pcd.paint_uniform_color([0.5, 0.5, 0.5]) # Gray walls
    
    points = np.asarray(downsampled_pcd.points)
    grid_indices = np.floor(points / voxel_size).astype(int)
    occupied_voxels = set(tuple(idx) for idx in grid_indices)
    
    # --- 2. CONFIGURATION SPACE INFLATION ---
    # Inflate by 1 voxel (0.2m) so the drone center stays away from walls
    safe_occupied_voxels = inflate_obstacles(occupied_voxels, voxel_size, inflation_radius_meters=0.2)

    # --- 3. CALCULATE MAP BOUNDS ---
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    bounds = [
        [min_bound[0], max_bound[0]],
        [min_bound[1], max_bound[1]],
        [min_bound[2], max_bound[2]]
    ]

    # --- 4. RUN A* PLANNER ---
    voxel_map = VoxelGridMap(voxel_size, safe_occupied_voxels)
    planner = AStar_SFC_Planner(voxel_map, bounds)

    start_meters = (-2.0, -2.6, 0.2) 
    goal_meters = (2.0, 3.6, 0.3)   

    print(f"\nRouting from {start_meters}m to {goal_meters}m...")
    
    start_idx = meters_to_grid(start_meters, voxel_size)
    goal_idx = meters_to_grid(goal_meters, voxel_size)

    start_time = time.perf_counter()
    path_indices = planner.search(start_idx, goal_idx)
    path_indices = planner.sfc_smoother()
    end_time = time.perf_counter()

    # --- 5. EXTRACT SFCS (Ported from front_end.py) ---
    sfc_constraints = []
    
    if path_indices:
        print(f"[ SUCCESS ] Path found in {(end_time - start_time)*1000:.2f} ms!")
        
        # 1. Convert to continuous 3x1 columns
        continuous_path = []
        for idx in path_indices:
            # Crucial: Use idx * voxel_size, NOT grid_to_meters, so raycast math matches
            pos_meters = (np.array(idx) * voxel_size).reshape(3, 1)
            continuous_path.append(pos_meters)
            
        # 2. Build the SFCs
        from rrt_mavsim.message_types.msg_waypoints import MsgWaypoints_SFC
        from rrt_mavsim.message_types.msg_flight_corridors import MsgFlightCorridor
        
        waypoints_smooth = MsgWaypoints_SFC(numDimensions=3)
        waypoints_smooth.add(position=continuous_path[0], parent=np.inf, cost=0.0, connectsToGoal=False)
        
        # Adjust these bounds based on how much wiggle room your drone needs
        sfc_height = 0.2 
        sfc_width = 0.2
        sfc_start_ext = 0.1
        sfc_end_ext = 0.1
        spline_type = 'clamped' 
        
        for i in range(1, len(continuous_path)):
            prev_pos = continuous_path[i-1]
            curr_pos = continuous_path[i]
            
            is_goal = (i == len(continuous_path) - 1)
            waypoints_smooth.add(position=curr_pos, parent=i-1, cost=0.0, connectsToGoal=is_goal)
            
            # Convert back to indices for the dynamic capping raycast
            idx_a = tuple(int(x) for x in np.ravel(prev_pos) / voxel_size)
            idx_b = tuple(int(x) for x in np.ravel(curr_pos) / voxel_size)
            
            safe_end_ext = planner.get_safe_extension_length(idx_a, idx_b, sfc_end_ext)
            safe_start_ext = planner.get_safe_extension_length(idx_b, idx_a, sfc_start_ext)
            
            if spline_type == 'natural' and (i-1) == 0:
                sfc = MsgFlightCorridor(
                    primaryPosition=prev_pos, secondaryPosition=curr_pos, primaryPosition_index=i-1,
                    numDimensions=3, height=sfc_height, width=sfc_width,
                    startExtension_length=30., endExtension_length=safe_end_ext
                )
            elif spline_type == 'natural' and is_goal:
                sfc = MsgFlightCorridor(
                    primaryPosition=prev_pos, secondaryPosition=curr_pos, primaryPosition_index=i-1,
                    numDimensions=3, height=sfc_height, width=sfc_width,
                    startExtension_length=safe_start_ext, endExtension_length=30.
                )
            else:
                sfc = MsgFlightCorridor(
                    primaryPosition=prev_pos, secondaryPosition=curr_pos, primaryPosition_index=i-1,
                    numDimensions=3, height=sfc_height, width=sfc_width,
                    startExtension_length=safe_start_ext, endExtension_length=safe_end_ext
                )
            waypoints_smooth.addSFC(sfc)
            
        corridors = waypoints_smooth.getAllFlightCorridors()
        for sfc in corridors:
            # Use the class method to calculate and return the matrices!
            A_mat, b_vec = sfc.getAbMatrices()
            sfc_constraints.append({'A': A_mat, 'b': b_vec})
            
        print(f"[ SUCCESS ] Extracted {len(sfc_constraints)} Safe Flight Corridors via A*.")

        # --- 6. OPTIMIZATION (MIN-SNAP B-SPLINE) ---
        if sfc_constraints:
            print("\n--- Starting Trajectory Optimization ---")
            degree = 4
            v_max = 3.0
            a_max = 2.0
            spline_type = 'clamped'
            
            # Initial Point Allocation (Give each box degree + 2 points to start)
            num_pts_list = [degree + 2] * len(sfc_constraints) 
            
            max_stretches = 5
            stretch_count = 0
            optimal_control_points = None
            
            # Define Start and End State constraints (Pos, Vel, Accel)
            start_pos_np = np.array(start_meters)
            end_pos_np = np.array(goal_meters)
            zero_vec = np.zeros(3)
            
            S = np.hstack((start_pos_np.reshape(3,1), zero_vec.reshape(3,1), zero_vec.reshape(3,1))) 
            E_reversed = np.hstack((zero_vec.reshape(3,1), zero_vec.reshape(3,1), end_pos_np.reshape(3,1)))
            SE = np.hstack((S, E_reversed))

            while stretch_count <= max_stretches:
                print(f"Optimization Attempt {stretch_count + 1}...")
                
                total_control_points = sum(num_pts_list) - degree * (len(num_pts_list) - 1)
                A_sfc, b_sfc = build_overlap_constraints(sfc_constraints, num_pts_list, degree, total_control_points)

                opt_start_time = time.perf_counter()
                num_segments = total_control_points - degree
                
                # Initialize Kinodynamic Clamped Backend
                optimizer = MinSnapEvalClamped(num_segments=num_segments, degree=degree)
                D_vel = optimizer._get_fast_cascaded_D_matrix(num_segments, degree, 1)
                D_accel = optimizer._get_fast_cascaded_D_matrix(num_segments, degree, 2)
                B_d3 = optimizer._get_B_d3_matrix(degree)
                SE_qp = SE @ B_d3

                W = optimizer.W
                A_eq = optimizer.B_combined.T

                try:
                    optimal_control_points = run_qp_solver(
                        objective_matrix=W,
                        equality_constraints=SE_qp,
                        inequality_constraints=(D_vel, D_accel, v_max, a_max, A_sfc, b_sfc), 
                        A_eq=A_eq,
                        degree=degree,
                        spline_type=spline_type,
                        use_minvo=True
                    )
                    
                    opt_duration = time.perf_counter() - opt_start_time 
                    print(f"[ SUCCESS ] OSQP Solved in {opt_duration*1000:.2f} ms!")
                    break 
                    
                except Exception as e:
                    print(f"[Phase 4] Solver failed (Kinematically tight): {e}")
                    if stretch_count < max_stretches:
                        print("          Stretching time allocation (+1 point to all SFCs)...")
                        num_pts_list = [pts + 1 for pts in num_pts_list]
                    else:
                        print("[ ERROR ] Max stretching attempts reached. Cannot solve.")
                    stretch_count += 1
    else:
        print("\n[ FAILED ] No path found.")

    # --- 6. OPEN3D VISUALIZATION ---
    geometries_to_draw = [downsampled_pcd]

    if safe_occupied_voxels:
        inflated_points = np.array(list(safe_occupied_voxels)) * voxel_size
        inflated_pcd = o3d.geometry.PointCloud()
        inflated_pcd.points = o3d.utility.Vector3dVector(inflated_points)
        inflated_pcd.paint_uniform_color([0.5, 0.5, 0.5]) 
        # geometries_to_draw.append(inflated_pcd) # Uncomment to see solid orange LiDAR walls

    if path_indices:
        # A. Add the A* Path (Green Line)
        path_meters = grid_to_meters(path_indices, voxel_size)
        path_array = np.array(path_meters)
        
        lines = [[i, i + 1] for i in range(len(path_meters) - 1)]
        colors = [[0, 1, 0] for _ in range(len(lines))]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(path_array)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        geometries_to_draw.append(line_set)
        
        # B. Add the SFC Boxes (Cyan Wireframes)
        sfc_boxes = get_sfc_open3d_geometries(corridors)
        geometries_to_draw.extend(sfc_boxes)

    print("\nOpening Open3D Visualizer...")
    if optimal_control_points is not None:
        # --- THE FIX: Unpack the Cascaded C-Array ---
        num_cp = len(optimal_control_points) // 3
        cp_x = optimal_control_points[0 : num_cp]
        cp_y = optimal_control_points[num_cp : 2 * num_cp]
        cp_z = optimal_control_points[2 * num_cp : 3 * num_cp]
        cp_array = np.vstack((cp_x, cp_y, cp_z)).T

        # 1. Plot the Red Control Polygon
        lines = [[i, i + 1] for i in range(len(cp_array) - 1)]
        colors = [[0, 1, 0] for _ in range(len(lines))] # Green
        
        optimized_line_set = o3d.geometry.LineSet()
        optimized_line_set.points = o3d.utility.Vector3dVector(cp_array)
        optimized_line_set.lines = o3d.utility.Vector2iVector(lines)
        optimized_line_set.colors = o3d.utility.Vector3dVector(colors)
        # geometries_to_draw.append(optimized_line_set)

        # 2. Evaluate the Smooth B-Spline Trajectory
        import scipy.interpolate as si
        
        degree = 4
        num_points = len(cp_array)
        
        # Generate a uniform clamped knot vector
        knots = np.concatenate((
            [0] * degree, 
            np.linspace(0, 1, num_points - degree + 1), 
            [1] * degree
        ))
        
        # Interpolate the control points mathematically
        spline_x = si.BSpline(knots, cp_array[:, 0], degree)
        spline_y = si.BSpline(knots, cp_array[:, 1], degree)
        spline_z = si.BSpline(knots, cp_array[:, 2], degree)
        
        # Sample the trajectory at 500 discrete time steps
        t = np.linspace(0, 1, 500)
        smooth_path = np.vstack((spline_x(t), spline_y(t), spline_z(t))).T
        
        # 3. Plot the Smooth Trajectory (Bright Green)
        lines_smooth = [[i, i + 1] for i in range(len(smooth_path) - 1)]
        colors_smooth = [[1, 0, 0] for _ in range(len(lines_smooth))] # Red
        
        smooth_line_set = o3d.geometry.LineSet()
        smooth_line_set.points = o3d.utility.Vector3dVector(smooth_path)
        smooth_line_set.lines = o3d.utility.Vector2iVector(lines_smooth)
        smooth_line_set.colors = o3d.utility.Vector3dVector(colors_smooth)
        
        geometries_to_draw.append(smooth_line_set)

    o3d.visualization.draw_geometries(geometries_to_draw, window_name="A* Path with SFC Boxes")

if __name__ == "__main__":
    run_planner_and_visualize()