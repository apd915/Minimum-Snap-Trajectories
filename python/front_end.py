import time
import numpy as np
import matplotlib.pyplot as plt

# rrt_mavsim imports
from rrt_mavsim.message_types.msg_world_map import MsgWorldMap, FloatingBlocksParams, CityParams, PlanarMazeParam, MapTypes
from rrt_mavsim.planners.rrt_sfc_bspline import RRT_SFC_BSpline
from rrt_mavsim.tools.waypointsTools import getNumCntPts_list
from rrt_mavsim.viewers.plot_map_path import PlotMapPath
from rrt_mavsim.message_types.msg_waypoints import MsgWaypoints_SFC

# Parameter imports
import rrt_mavsim.parameters.planner_parameters as PLAN
import rrt_mavsim.parameters.floatingBlocks_parameters as FLOATING_PARAM
import rrt_mavsim.parameters.city_parameters as CITY_PARAM
import rrt_mavsim.parameters.planar_maze_parameters as MAZE_PARAM
import rrt_mavsim.parameters.flightCorridor_parameters as FLIGHT

# Discretization and A* imports
from mapping.voxel_grid import SparseVoxelGrid
from planning.astar_sfc import AStar_SFC_Planner
from planning.dynamic_sfc import AsymmetricSFCManager, StandaloneWaypointsSFC

class FrontEndSFC:
    def __init__(self, sfc_height, sfc_width, sfc_start_ext, sfc_end_ext, spline_type="natural", map_type="FLOATING_BLOCKS", degree=4):
        """
        Initializes the environment and the RRT planner.
        """
        self.degree = degree
        self.sfc_height = sfc_height
        self.sfc_width = sfc_width
        self.sfc_start_ext = sfc_start_ext
        self.sfc_end_ext = sfc_end_ext
        self.spline_type = spline_type
        
        # 1. Initialize the Map
        # Note: We are defaulting to floating blocks based on your test, 
        # but you can easily swap this logic to accept the CITY map later.
        self.worldMap = MsgWorldMap(
            obstacleFieldType=MapTypes.FLOATING_BLOCKS,
            numDimensions_algorithm=FLOATING_PARAM.numDimensions,
            floatingBlocksParams=FloatingBlocksParams()
        )

        # self.worldMap = MsgWorldMap(
        #     obstacleFieldType=MapTypes.CITY,
        #     numDimensions_algorithm=CITY_PARAM.numDimensions,
        #     CityParams=CityParams()
        # )

        # self.worldMap = MsgWorldMap(
        #     obstacleFieldType=MapTypes.PLANAR_MAZE,
        #     numDimensions_algorithm=MAZE_PARAM.numDimensions,
        #     planarMazeParams=PlanarMazeParam()
        # )


        # 1.5. Transform map into voxel grid
        # Define your resolution (e.g., 2.5 meters per voxel)
        self.voxel_resolution = 0.5 
        
        # Define your drone's physical radius
        self.drone_physical_radius = 0.5 

        # Round up to the nearest whole voxel to prevent quantization loss!
        inflation_voxels = np.ceil(self.drone_physical_radius / self.voxel_resolution)
        self.grid_inflation_radius = inflation_voxels * self.voxel_resolution

        # Initialize your discrete grid (Assuming you build a SparseVoxelGrid class)
        self.discrete_grid = SparseVoxelGrid(resolution=self.voxel_resolution)


        # Populate the grid using the continuous obstacles
        continuous_obstacles = self.worldMap.get_obstacles()
        self.discrete_grid.populate_from_continuous(
            obstacles=continuous_obstacles, 
            inflation_radius=self.grid_inflation_radius
        )
        # ---------------------------------------------------------

        # --- THE FIX: Pass INFLATED discrete voxel indices back to continuous meters for the KD-Tree! ---
        # This perfectly aligns the continuous math engine with the discrete A* pathfinder.
        self.inflated_obstacle_meters = [
            np.array(idx) * self.voxel_resolution 
            for idx in self.discrete_grid.occupied_voxels_inflated
        ]
        
        raw_obstacle_meters = [
            np.array(idx) * self.voxel_resolution 
            for idx in self.discrete_grid.occupied_voxels_raw
        ]
        
        # Define the physical constraints perfectly matching sfc_width
        max_cp_drift = (self.sfc_width / 2.0) - self.drone_physical_radius
        
        # Initialize the Manager with RAW points
        self.sfc_manager = AsymmetricSFCManager(
            raw_uninflated_obstacle_points=raw_obstacle_meters, 
            drone_physical_radius=self.drone_physical_radius,  
            max_cp_drift=max_cp_drift,
            voxel_resolution=self.voxel_resolution
        )

        occupied_inflated = self.discrete_grid.occupied_voxels_inflated
        occupied_raw = self.discrete_grid.occupied_voxels_raw


        self.bounds = [(0,FLOATING_PARAM.northEnd), (0,FLOATING_PARAM.eastEnd), (0,FLOATING_PARAM.downEnd)]

        self.path_gen_astar = AStar_SFC_Planner(self.discrete_grid, self.bounds)



    def get_corridors_astar(self, start_pos, end_pos):
        # 1. Discretize and Search
        start_discretized = tuple(int(x) for x in np.ravel(start_pos) // self.voxel_resolution)
        goal_discretized = tuple(int(x) for x in np.ravel(end_pos) // self.voxel_resolution)

        print(f"[Front-End] Starting A* Search from {start_discretized} to {goal_discretized}...")
        t_start = time.perf_counter()
        
        self.path_gen_astar.search(start_discretized, goal_discretized)
        astar_path_indices = self.path_gen_astar.sfc_smoother()
        
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        print(f"[Front-End] A* Search & Smoothing Completed in {elapsed_ms:.2f} ms")

        if not astar_path_indices:
            print("[Error] A* failed to find a path.")
            return None, None, None, None

        # 2. Convert discrete grid indices back to continuous 3x1 column vectors
        continuous_path = []
        for idx in astar_path_indices:
            # .reshape(3, 1) is CRITICAL to match your existing matrix math!
            pos_meters = (np.array(idx) * self.voxel_resolution).reshape(3, 1)
            continuous_path.append(pos_meters)


        # --- MATHEMATICAL CLIPPING DIAGNOSTIC ---
        print("\n[Diagnostic] Checking A* Path against Inflated KD-Tree...")
        path_is_safe = True
        for i in range(1, len(continuous_path)):
            pA = continuous_path[i-1].flatten()
            pB = continuous_path[i].flatten()
            
            # Calculate segment length and direction
            v = pB - pA
            seg_len = np.linalg.norm(v)
            if seg_len == 0: continue
            u = v / seg_len
            
            # Check every inflated obstacle against the segment
            for obs in self.inflated_obstacle_meters:
                # 1. Project obstacle onto the line segment (dot product)
                t = np.dot(obs - pA, u)
                
                # 2. Only care about obstacles directly adjacent to the segment
                if 0 <= t <= seg_len:
                    # 3. Calculate perpendicular distance from segment to obstacle
                    proj_point = pA + t * u
                    dist_to_line = np.linalg.norm(obs - proj_point)
                    
                    # If the distance is exactly 0, the line goes directly through the voxel center
                    # If the distance is < the voxel resolution, it's clipping the physical block!
                    if dist_to_line < self.voxel_resolution:
                        print(f"  -> [WARNING] Segment {i} is clipping an inflated voxel!")
                        print(f"     Obstacle at {obs} is only {dist_to_line:.3f}m from the line.")
                        path_is_safe = False
                        
        if path_is_safe:
            print("  -> [PASS] Mathematical path completely clears all inflated voxel centers.")
        print("------------------------------------------------------\n")

        # 3. Build the NEW StandaloneWaypointsSFC object
        waypoints_smooth = StandaloneWaypointsSFC(numDimensions=FLOATING_PARAM.numDimensions)
        
        # Add the Start point
        waypoints_smooth.add(
            position=continuous_path[0], 
            parent=np.inf, 
            cost=0.0, 
            connectsToGoal=False
        )

        # 4. Generate the Safe Flight Corridors (SFCs) along the path
        for i in range(1, len(continuous_path)):
            prev_pos = continuous_path[i-1]
            curr_pos = continuous_path[i]
            
            is_goal = (i == len(continuous_path) - 1)
            waypoints_smooth.add(
                position=curr_pos, parent=i-1, cost=0.0, connectsToGoal=is_goal
            )

            # Determine massive extensions only if natural splines are used
            ext_start = self.sfc_start_ext
            ext_end = self.sfc_end_ext
            if self.spline_type == 'natural':
                if i == 1: ext_start = 30.0
                if is_goal: ext_end = 30.0
            
            # The Manager runs Layer 1, 2, and 3, and returns the finished duck-typed object!
            sfc = self.sfc_manager.generate_sfc(
                pA=prev_pos, 
                pB=curr_pos, 
                W=self.sfc_width, 
                ext_start=ext_start, 
                ext_end=ext_end
            )
            
            waypoints_smooth.addSFC(sfc)
            
        corridors = waypoints_smooth.getAllFlightCorridors()
        print(f"[Front-End] Extracted {len(corridors)} Safe Flight Corridors via A*.")

        # 5. Calculate Control Point Allocation (Dynamic Kinematic & Local Support)
        exclusive_pts_list, int_pts_list = self.allocate_dynamic_control_points(
            corridors=corridors,
            degree=self.degree,
            v_max=3.0,       
            a_max=2.0,       
            pts_per_sec=1.0  
        )

        self.last_corridors = corridors
        return corridors, exclusive_pts_list, int_pts_list, waypoints_smooth, None
    

    def get_corridors(self, start_pos, end_pos, num_points_per_unit=FLIGHT.numPoints_perUnit):
        """
        Runs the RRT search and extracts the raw math for the Back-End.
        
        Returns:
            sfc_constraints: List of dicts [{'A': A_mat, 'b': b_vec}, ...]
            num_pts_list: List of integers denoting how many points belong in each box
        """
        print("[Front-End] Running RRT Path Search...")
        start_time = time.time()
        
        # 1. Run the mathematical search
        self.path_gen.generateSFCPaths(
            startPosition_3D=start_pos,
            endPosition_3D=end_pos,
            worldMap=self.worldMap,
            segmentLength=FLIGHT.segmentLength,
        )
        
        print(f"[Front-End] Path found in {time.time() - start_time:.3f} seconds.")
        
        # 2. Extract the smooth waypoints and the corridor objects
        waypoints_not_smooth = self.path_gen.getWaypointsNotSmooth()
        waypoints_smooth = self.path_gen.getWaypointsSmooth()
        corridors = waypoints_smooth.getAllFlightCorridors()
        
        if not corridors:
            return None, None
            
        print(f"[Front-End] Extracted {len(corridors)} Safe Flight Corridors.")
        
        # 3. Calculate the Control Point Allocation
        # We use Dean's exact tool to calculate how many points each box gets 
        # based on the box's physical length.
        num_pts_list = getNumCntPts_list(
            waypoints=waypoints_smooth, 
            numPointsPerUnit=num_points_per_unit
        )
        
        # 4. Extract the A and b constraint matrices
        sfc_constraints = []
        for sfc in corridors:
            A_mat, b_vec = sfc.getAbMatrices()
            sfc_constraints.append({
                'A': A_mat, 
                'b': b_vec
            })
            
        return sfc_constraints, num_pts_list, waypoints_smooth, waypoints_not_smooth
    

    def allocate_dynamic_control_points(self, corridors, degree, v_max=3.0, a_max=2.0, pts_per_sec=1.0):
        num_corridors = len(corridors)
        
        # 1. Cleanly separate the lists!
        exclusive_pts_list = [0] * num_corridors
        int_pts_list = [degree] * max(0, num_corridors - 1)
        
        # PASS 1: The Straightaway Baseline
        for i in range(num_corridors):
            L = getattr(corridors[i], 'length', 1.0)
            t_target = max(L / v_max, 2.0 * np.sqrt(L / a_max))
            N_kinematic = int(np.ceil(t_target * pts_per_sec))
            
            # The straightaway gets its own points independently
            exclusive_pts_list[i] = max(N_kinematic, degree)

        # PASS 2: The Apex Injector (The 3rd Entity)
        for i in range(num_corridors - 1):
            sfc_in = corridors[i]
            sfc_out = corridors[i+1]
            
            v_in = np.ravel(sfc_in.secondaryPosition) - np.ravel(sfc_in.primaryPosition)
            v_out = np.ravel(sfc_out.secondaryPosition) - np.ravel(sfc_out.primaryPosition)
            
            norm_in, norm_out = np.linalg.norm(v_in), np.linalg.norm(v_out)
            
            if norm_in > 0.001 and norm_out > 0.001:
                cos_theta = np.clip(np.dot(v_in, v_out) / (norm_in * norm_out), -1.0, 1.0)
                momentum_shed_factor = 1.0 - cos_theta
                
                if momentum_shed_factor > 0.1: 
                    apex_pool_size = int(np.ceil(momentum_shed_factor * (degree * 3)))
                    
                    # THE FIX: Inject the flexibility DIRECTLY into the intersection overlap!
                    int_pts_list[i] += apex_pool_size

        return exclusive_pts_list, int_pts_list
    

    def compile_system_constraints(self, corridors, exclusive_pts_list, int_pts_list):
        import numpy as np
        A_ineq_list, b_ineq_list = [], []
        num_dimensions = 3
        num_corridors = len(corridors)
        
        # Total points is just the clean sum of the two arrays
        total_num_points = sum(exclusive_pts_list) + sum(int_pts_list)
        global_cp_index = 0
        
        for i in range(num_corridors):
            A_curr, b_curr = corridors[i].getAbMatrices()
            b_curr = b_curr.flatten()
            num_ineq_curr = A_curr.shape[0]
            
            # 1. Add EXCLUSIVE points for the straightaway
            pts_exclusive = exclusive_pts_list[i]
            for _ in range(pts_exclusive):
                A_padded = np.zeros((num_ineq_curr, total_num_points * num_dimensions))
                col_start = global_cp_index * num_dimensions
                A_padded[:, col_start:col_start + num_dimensions] = A_curr
                A_ineq_list.append(A_padded)
                b_ineq_list.append(b_curr)
                global_cp_index += 1
                
            # 2. Add INTERSECTION points
            if i < num_corridors - 1:
                A_next, b_next = corridors[i+1].getAbMatrices()
                b_next = b_next.flatten()
                
                A_int = np.vstack((A_curr, A_next))
                b_int = np.concatenate((b_curr, b_next))
                num_ineq_int = A_int.shape[0]
                
                # Retrieve the dynamically sized overlap pool!
                current_int_pts = int_pts_list[i]
                for _ in range(current_int_pts):
                    A_padded = np.zeros((num_ineq_int, total_num_points * num_dimensions))
                    col_start = global_cp_index * num_dimensions
                    A_padded[:, col_start:col_start + num_dimensions] = A_int
                    A_ineq_list.append(A_padded)
                    b_ineq_list.append(b_int)
                    global_cp_index += 1
                    
        A_sfc_total = np.vstack(A_ineq_list)
        b_sfc_total = np.concatenate(b_ineq_list)
        
        return A_sfc_total, b_sfc_total, total_num_points


    def visualize_debugging(self, waypoints_smooth=None):
        """
        Plots the continuous map, inflated discrete voxels, and the smoothed A* path.
        Fully renders the 3D grid to expose clipping anomalies.
        """
        print("[Debug] Launching Map Visualization...")
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # 1. Draw the Full Inflated Voxel Aura (Red)
        occupied_inflated = self.discrete_grid.occupied_voxels_inflated
        if len(occupied_inflated) > 0:
            ix, iy, iz = zip(*occupied_inflated)
            
            ix_m = np.array(ix) * self.voxel_resolution
            iy_m = np.array(iy) * self.voxel_resolution
            iz_m = np.array(iz) * self.voxel_resolution
            
            # Render the entire map with low opacity to see the path inside
            ax.scatter(ix_m, iy_m, iz_m, 
                       color='red', marker='s', s=20, alpha=0.10, label='Inflated Voxel Buffer')

        # 2. Plot the Continuous Smoothed A* Path (Blue Line)
        if waypoints_smooth is not None:
            path_coords = waypoints_smooth.getAllPositions()
            if len(path_coords) > 0:
                px = [pos[0,0] for pos in path_coords]
                py = [pos[1,0] for pos in path_coords]
                pz = [pos[2,0] for pos in path_coords]
                
                # Plot the line on top of the voxels
                ax.plot(px, py, pz, color='blue', linewidth=3, zorder=10, label='Smoothed A* Line')
                
                # Plot the physical corner nodes to see exactly where the smoother anchors
                ax.scatter(px, py, pz, color='cyan', s=60, zorder=11, edgecolors='black', label='A* Anchor Nodes')
            
        # 3. Format and Render
        ax.set_xlim(FLOATING_PARAM.x_limits[0], FLOATING_PARAM.x_limits[1])
        ax.set_ylim(FLOATING_PARAM.y_limits[0], FLOATING_PARAM.y_limits[1])
        ax.set_zlim(FLOATING_PARAM.z_limits[0], FLOATING_PARAM.z_limits[1])
        ax.set_box_aspect(FLOATING_PARAM.aspect_ratio)
        
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (Altitude)')
        ax.legend()
        
        # Block=True pauses the python script so you can rotate and inspect the clipping
        plt.show(block=True)