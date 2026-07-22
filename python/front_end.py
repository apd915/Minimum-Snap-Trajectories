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
from planning.static_sfc import StaticSFCManager

class FrontEndSFC:
    def __init__(self, sfc_height, sfc_width, sfc_start_ext, sfc_end_ext, spline_type="natural", map_type="FLOATING_BLOCKS", degree=4, aircraft_type="multi-rotor", drone_radius=0.2):
        """
        Initializes the environment and the RRT planner.
        """
        self.degree = degree
        self.sfc_height = sfc_height
        self.sfc_width = sfc_width
        self.sfc_start_ext = sfc_start_ext
        self.sfc_end_ext = sfc_end_ext
        self.spline_type = spline_type
        self.aircraft_type = aircraft_type
        
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
        self.drone_physical_radius = drone_radius 

        # Calculate inflation radius based on aircraft type
        if self.aircraft_type == "fixed-wing":
            # Fixed-wing: inflate by sfc_width/2 so pre-sized SFC boxes never clip obstacles
            self.grid_inflation_radius = (self.sfc_width / 2.0) + self.drone_physical_radius
        else:
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

        # --- THE FIX: Add half-resolution to shift from Voxel Corners to Voxel Centers! ---
        # --- THE FIX: Pass INFLATED discrete voxel indices back to continuous meters for the KD-Tree! ---
        # This perfectly aligns the continuous math engine with the discrete A* pathfinder.
        self.inflated_obstacle_meters = [
            np.array(idx) * self.voxel_resolution + (self.voxel_resolution / 2.0)
            for idx in self.discrete_grid.occupied_voxels_inflated
        ]
        
        self.raw_obstacle_meters = [
            np.array(idx) * self.voxel_resolution + (self.voxel_resolution / 2.0)
            for idx in self.discrete_grid.occupied_voxels_raw
        ]
        
        # Define the physical constraints perfectly matching sfc_width
        max_cp_drift = (self.sfc_width / 2.0) - self.drone_physical_radius
        
        # We pass the INFLATED points, and tell the builder the drone radius is 0.0 
        # (since the inflation already contains the drone!).
        self.sfc_manager = AsymmetricSFCManager(
            raw_uninflated_obstacle_points=self.inflated_obstacle_meters, 
            drone_physical_radius=0.0,  
            max_cp_drift=max_cp_drift,
            voxel_resolution=self.voxel_resolution
        )
        self.static_sfc_manager = StaticSFCManager(self.sfc_height, self.sfc_width)

        occupied_inflated = self.discrete_grid.occupied_voxels_inflated
        occupied_raw = self.discrete_grid.occupied_voxels_raw


        self.bounds = [(0,FLOATING_PARAM.northEnd), (0,FLOATING_PARAM.eastEnd), (0,FLOATING_PARAM.downEnd)]
        
        from mapping.ring_buffer import RingBufferGrid
        size_x = int(np.ceil(FLOATING_PARAM.northEnd / self.voxel_resolution)) + 1
        size_y = int(np.ceil(FLOATING_PARAM.eastEnd / self.voxel_resolution)) + 1
        size_z = int(np.ceil(FLOATING_PARAM.downEnd / self.voxel_resolution)) + 1
        
        self.discrete_grid.ring_buffer = RingBufferGrid(size_x, size_y, size_z)
        for (vx, vy, vz) in occupied_inflated:
            self.discrete_grid.ring_buffer.set_occupied(vx, vy, vz)

        self.path_gen_astar = AStar_SFC_Planner(
            voxel_grid=self.discrete_grid, 
            bounds=self.bounds, 
            drone_radius=self.drone_physical_radius,
            aircraft_type=self.aircraft_type,
            use_ring_buffer=True
        )



    def get_corridors_astar(self, start_pos, end_pos):
        # 1. Discretize and Search
        start_discretized = tuple(int(x) for x in np.ravel(start_pos) // self.voxel_resolution)
        goal_discretized = tuple(int(x) for x in np.ravel(end_pos) // self.voxel_resolution)

        print(f"[Front-End] Starting A* Search from {start_pos} to {end_pos} continuous, {start_discretized} to {goal_discretized} discretized...")
        t_start = time.perf_counter()
        
        self.path_gen_astar.search(start_discretized, goal_discretized)
        astar_path_indices = self.path_gen_astar.sfc_smoother()
        
        # Extract the saved raw indices
        raw_astar_indices = getattr(self.path_gen_astar, 'raw_path', [])

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        print(f"[Front-End] A* Search & Smoothing Completed in {elapsed_ms:.2f} ms")

        if not astar_path_indices:
            print("[Error] A* failed to find a path.")
            return None, None, None, None, None

        # 2. Convert BOTH discrete paths back to continuous meters (using Voxel Centers!)
        continuous_path = []
        for idx in astar_path_indices:
            pos_meters = (np.array(idx) * self.voxel_resolution + (self.voxel_resolution / 2.0)).reshape(3, 1)
            continuous_path.append(pos_meters)
            
        waypoints_not_smooth = StandaloneWaypointsSFC(numDimensions=FLOATING_PARAM.numDimensions)
        for idx in raw_astar_indices:
            pos_meters = (np.array(idx) * self.voxel_resolution + (self.voxel_resolution / 2.0)).reshape(3, 1)
            waypoints_not_smooth.add(position=pos_meters, parent=np.inf, cost=0.0, connectsToGoal=False)

        # 2.5 Anchor the exact continuous Start and Goal positions!
        if len(continuous_path) >= 2:
            continuous_path[0] = start_pos.reshape(3, 1)
            continuous_path[-1] = end_pos.reshape(3, 1)
            
            waypoints_not_smooth.positions[0] = start_pos.reshape(3, 1)
            waypoints_not_smooth.positions[-1] = end_pos.reshape(3, 1)

        # raw_path_points = [pos for pos in waypoints_not_smooth.positions]
        # run_clipping_diagnostic(raw_path_points, "Raw A* Path")
        # run_clipping_diagnostic(continuous_path, "Smoothed A* Path")

        # 3. Build the NEW StandaloneWaypointsSFC object
        waypoints_smooth = StandaloneWaypointsSFC(numDimensions=FLOATING_PARAM.numDimensions)
        
        waypoints_smooth.add(position=continuous_path[0], parent=np.inf, cost=0.0, connectsToGoal=False)

        for i in range(1, len(continuous_path)):
            prev_pos = continuous_path[i-1]
            curr_pos = continuous_path[i]
            
            is_goal = (i == len(continuous_path) - 1)
            waypoints_smooth.add(position=curr_pos, parent=i-1, cost=0.0, connectsToGoal=is_goal)

            ext_start = self.sfc_start_ext
            ext_end = self.sfc_end_ext
            if self.spline_type == 'natural':
                if i == 1: ext_start = 30.0
                if is_goal: ext_end = 30.0
            
            if self.aircraft_type == "fixed-wing":
                sfc = self.static_sfc_manager.generate_sfc(
                    pA=prev_pos, pB=curr_pos, ext_start=ext_start, ext_end=ext_end
                )
            else:
                sfc = self.sfc_manager.generate_sfc(
                    pA=prev_pos, pB=curr_pos, W=self.sfc_width, ext_start=ext_start, ext_end=ext_end
                )
            waypoints_smooth.addSFC(sfc)
            
        corridors = waypoints_smooth.getAllFlightCorridors()
        print(f"[Front-End] Extracted {len(corridors)} Safe Flight Corridors via A*.")

        # run_sfc_volume_diagnostic(corridors)

        # --- THE FIX: Generate the Unified Constraint Pools ---
        if self.aircraft_type == "fixed-wing":
            allocation_data = self.allocate_dynamic_control_points_fixed_wing(
                corridors=corridors, degree=self.degree, v_max=3.0, a_max=2.0, pts_per_sec=1.5
            )
        else:
            allocation_data = self.allocate_dynamic_control_points(
                corridors=corridors, degree=self.degree, v_max=3.0, a_max=2.0, pts_per_sec=1.0  
            )

        self.last_corridors = corridors
        
        # Return the allocation_data instead of the split lists!
        return corridors, allocation_data, waypoints_smooth, waypoints_not_smooth


    def allocate_dynamic_control_points_fixed_wing(self, corridors, degree, v_max=3.0, a_max=2.0, pts_per_sec=1.0):
        """
        Apex-Centric Dynamic Allocation Manager for Fixed-Wing.
        Treats intersections as independent geometric entities and returns num_pts_list.
        """
        num_corridors = len(corridors)
        num_pts_list = [0] * num_corridors
        
        # ==========================================
        # PASS 1: The Straightaway Baseline
        # ==========================================
        for i in range(num_corridors):
            L = corridors[i].getDistancePrimaryToSecondary()
            
            # Pure straight-line kinematics
            t_cruise = L / v_max
            t_accel = 2.0 * np.sqrt(L / a_max)
            t_target = max(t_cruise, t_accel)
            
            # Assign baseline points
            N_kinematic = int(np.ceil(t_target * pts_per_sec))
            
            # Enforce the mathematical Local Support Floor
            num_pts_list[i] = max(N_kinematic, 2 * degree)

        # ==========================================
        # PASS 2: The Apex Injector (The 3rd Entity)
        # ==========================================
        # We loop through the joints BETWEEN the corridors
        for i in range(num_corridors - 1):
            sfc_in = corridors[i]
            sfc_out = corridors[i+1]
            
            # Grab the 3 intersection waypoints
            p0 = np.ravel(sfc_in.primaryPosition)
            p1 = np.ravel(sfc_in.secondaryPosition) # The Apex
            p2 = np.ravel(sfc_out.secondaryPosition)
            
            v_in = p1 - p0
            v_out = p2 - p1
            
            norm_in = np.linalg.norm(v_in)
            norm_out = np.linalg.norm(v_out)
            
            if norm_in > 0.001 and norm_out > 0.001:
                # Calculate the turn angle
                cos_theta = np.dot(v_in, v_out) / (norm_in * norm_out)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                
                # The momentum shedding factor (0 for straight, 1.0 for 90-deg)
                momentum_shed_factor = 1.0 - cos_theta
                
                # If it's a real turn (e.g., more than a ~25 degree bend)
                if momentum_shed_factor > 0.1: 
                    # 1. Create the Apex Pool
                    # A 90-deg turn creates a pool of exactly deg*3 (start of turn,curve,end of turn) extra control points
                    apex_pool_size = int(np.ceil(momentum_shed_factor * (degree*3)))
                    
                    # 2. Split the pool in half
                    half_pool = apex_pool_size // 2
                    
                    # 3. Inject the shared load!
                    # SFC A gets extra points at its tail to brake
                    num_pts_list[i] += half_pool
                    # SFC B gets extra points at its nose to accelerate out
                    num_pts_list[i+1] += half_pool

        return num_pts_list


    def allocate_dynamic_control_points(self, corridors, degree, v_max=3.0, a_max=2.0, pts_per_sec=1.0):
        """
        Sequential Pairwise Allocator.
        Guarantees geometric volume by ONLY ever intersecting two adjacent SFCs.
        Automatically 'shaves down' tiny SFCs to 0 exclusive points to prevent bunching.
        """
        num_corridors = len(corridors)
        constraint_pools = []

        for i in range(num_corridors):
            sfc = corridors[i]
            L_base = sfc.getDistancePrimaryToSecondary()

            # 1. EXCLUSIVE POOL (The Main Body)
            # Determine true physical overlap from neighbors
            actual_ext_prev = corridors[i-1].bounds[0] - corridors[i-1].getDistancePrimaryToSecondary() if i > 0 else 0.0
            actual_ext_next = corridors[i+1].bounds[1] if i < num_corridors - 1 else 0.0

            # Subsumption Heuristic: Is this SFC a tiny joint, or a travel hallway?
            if L_base <= (actual_ext_prev + actual_ext_next):
                exclusive_pts = 0 # Subsumed by bridges! Shave down to 0 to prevent bunching.
            else:
                t_target = max(L_base / v_max, 2.0 * np.sqrt(L_base / a_max))
                N_kinematic = int(np.ceil(t_target * pts_per_sec))
                
                # --- THE FIX: Remove the 'degree' minimum! ---
                # Allow tiny 1m or 2m travel corridors to have just 1 or 2 points.
                # Enforcing max(degree, ...) was cramming 4 points into 1m gaps!
                exclusive_pts = N_kinematic

            # Edge case: If there is literally only 1 SFC in the whole map
            if num_corridors == 1:
                exclusive_pts = max(exclusive_pts, degree * 2)

            if exclusive_pts > 0:
                constraint_pools.append({'pts': exclusive_pts, 'sfcs': [i], 'type': 'exclusive'})

            # 2. BRIDGE POOL (The Intersection to the Next Box)
            if i < num_corridors - 1:
                # Pairwise intersection GUARANTEES 3D geometric volume!
                # No 3-way null-sets!
                constraint_pools.append({'pts': degree, 'sfcs': [i, i+1], 'type': 'bridge'})

        return constraint_pools
    

    def compile_system_constraints(self, corridors, constraint_pools):
        import numpy as np
        A_ineq_list, b_ineq_list = [], []
        num_dimensions = 3
        
        # Total points is the clean sum of all pools
        total_num_points = sum(pool['pts'] for pool in constraint_pools)
        global_cp_index = 0
        
        for pool in constraint_pools:
            pts = pool['pts']
            sfc_indices = pool['sfcs']
            
            # 1. Mathematically overlap all SFC matrices in this pool (The N-way Intersection)
            A_combined_list = []
            b_combined_list = []
            for sfc_idx in sfc_indices:
                A_curr, b_curr = corridors[sfc_idx].getAbMatrices()
                A_combined_list.append(A_curr)
                b_combined_list.append(b_curr.flatten())
                
            A_pool = np.vstack(A_combined_list)
            b_pool = np.concatenate(b_combined_list)
            num_ineq_pool = A_pool.shape[0]
            
            # 2. Lock the control points strictly inside this overlapping volume!
            for _ in range(pts):
                A_padded = np.zeros((num_ineq_pool, total_num_points * num_dimensions))
                col_start = global_cp_index * num_dimensions
                A_padded[:, col_start:col_start + num_dimensions] = A_pool
                
                A_ineq_list.append(A_padded)
                b_ineq_list.append(b_pool)
                global_cp_index += 1
                
        A_sfc_total = np.vstack(A_ineq_list)
        b_sfc_total = np.concatenate(b_ineq_list)
        
        return A_sfc_total, b_sfc_total, total_num_points


    def visualize_debugging(self, waypoints_smooth=None, waypoints_not_smooth=None, corridors=None):
        """
        Master Diagnostic Plotter (High Performance):
        - Actual Simulation Objects (via native PlotMapPath)
        - Gray Dashed Line: Raw Unsmoothed A* Path
        - Blue Solid Line: Smoothed A* Path
        - Green Wireframes: SFC Bounding Boxes
        """
        import matplotlib.pyplot as plt
        from rrt_mavsim.viewers.plot_map_path import PlotMapPath
        import rrt_mavsim.parameters.floatingBlocks_parameters as FLOATING_PARAM

        print("[Debug] Launching Master Geometry Visualization with Actual Objects...")
        
        # 1. Use the optimized native renderer to draw the ACTUAL blocks
        plotter = PlotMapPath(
            map=self.worldMap,
            waypoints_not_smooth=waypoints_not_smooth,
            waypoints_smooth=None, # We draw paths manually below to customize colors
            controlPoints_smooth_list=None
        )
        
        # This builds the environment polygons and sets the physical limits, 
        # but importantly, it leaves the figure open for us to draw on!
        plotter.plot(
            x_limits=FLOATING_PARAM.x_limits,
            y_limits=FLOATING_PARAM.y_limits,
            z_limits=FLOATING_PARAM.z_limits,
            aspectRatio=FLOATING_PARAM.aspect_ratio,
        )
        
        # Hijack the active 3D axis
        # ax = plt.gca()

        # # 2. Plot the Raw Unsmoothed A* Path (Gray Dashed)
        # if waypoints_not_smooth is not None:
        #     raw_coords = waypoints_not_smooth.getAllPositions()
        #     if len(raw_coords) > 0:
        #         rx = [pos[0,0] for pos in raw_coords]
        #         ry = [pos[1,0] for pos in raw_coords]
        #         rz = [pos[2,0] for pos in raw_coords]
        #         ax.plot(rx, ry, rz, color='gray', linestyle='--', linewidth=2, zorder=8, label='Raw A* Path')
        #         ax.scatter(rx, ry, rz, color='gray', s=10, zorder=9)

        # # 3. Plot the Continuous Smoothed A* Path (Blue Line)
        # if waypoints_smooth is not None:
        #     path_coords = waypoints_smooth.getAllPositions()
        #     if len(path_coords) > 0:
        #         px = [pos[0,0] for pos in path_coords]
        #         py = [pos[1,0] for pos in path_coords]
        #         pz = [pos[2,0] for pos in path_coords]
        #         ax.plot(px, py, pz, color='blue', linewidth=3, zorder=10, label='Smoothed A* Segment')
        #         ax.scatter(px, py, pz, color='cyan', s=60, zorder=11, edgecolors='black', label='A* Anchor Nodes')

        # 4. Plot the Safe Flight Corridors (Green Wireframes)
        # if corridors is not None:
        #     edges = [
        #         (0, 1), (1, 2), (2, 3), (3, 0), # Bottom face
        #         (4, 5), (5, 6), (6, 7), (7, 4), # Top face
        #         (0, 4), (1, 5), (2, 6), (3, 7)  # Vertical pillars
        #     ]
        #     for idx, sfc in enumerate(corridors):
        #         vertices = sfc.getAllVertices_3D() 
        #         for edge in edges:
        #             p1, p2 = vertices[:, edge[0]], vertices[:, edge[1]]
        #             # Draw continuous green lines connecting the corners of the SFC
        #             ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
        #                     color='lime', linewidth=1.5, alpha=0.8, zorder=12)

        # ax.legend()
        plt.show(block=True)


# --- SFC VOLUME VS INFLATED OBSTACLE DIAGNOSTIC ---
        def run_sfc_volume_diagnostic(corridors):
            print("\n[Diagnostic] Checking SFC Volumes against INFLATED Obstacles...")
            all_safe = True
            
            for i, sfc in enumerate(corridors):
                pA = sfc.primaryPosition.flatten()
                ux, uy, uz = sfc.ux, sfc.uy, sfc.uz
                b = sfc.bounds
                
                max_x, min_x = b[0], -b[1]
                max_y, min_y = b[2], -b[3]
                max_z, min_z = b[4], -b[5]
                
                # --- THE FIX: Volume Expansion ---
                # Expand the SFC check to cover the physical projection of the voxel!
                half_res = self.voxel_resolution / 2.0
                r_x = half_res * (abs(ux[0]) + abs(ux[1]) + abs(ux[2]))
                r_y = half_res * (abs(uy[0]) + abs(uy[1]) + abs(uy[2]))
                r_z = half_res * (abs(uz[0]) + abs(uz[1]) + abs(uz[2]))

                collisions = 0
                for obs in self.inflated_obstacle_meters:
                    v = obs - pA
                    px = np.dot(v, ux)
                    py = np.dot(v, uy)
                    pz = np.dot(v, uz)
                    
                    eps = 1e-3 
                    
                    # If the distance to the center is within the bounds + the voxel projection, it overlaps!
                    if (min_x - r_x + eps <= px <= max_x + r_x - eps) and \
                       (min_y - r_y + eps <= py <= max_y + r_y - eps) and \
                       (min_z - r_z + eps <= pz <= max_z + r_z - eps):
                        
                        print(f"  -> [FATAL] SFC {i} mathematically OVERLAPS an INFLATED obstacle at {np.round(obs, 2)}!")
                        collisions += 1
                        all_safe = False
                        
                if collisions > 0:
                    print(f"     [Result] SFC {i} FAILED ({collisions} volume overlaps).")
                    
            if all_safe:
                print("  -> [PASS] All SFCs perfectly avoid inflated voxel volumes!")
            print("-" * 55)


# --- MATHEMATICAL CLIPPING DIAGNOSTIC ---
        def run_clipping_diagnostic(path_points, path_name):
            print(f"\n[Diagnostic] Checking {path_name} against RAW KD-Tree Reality...")
            path_is_safe = True
            
            # 0.5m Drone + 0.25m Voxel Volume
            safety_threshold = 0.75 
            
            for i in range(1, len(path_points)):
                pA = path_points[i-1].flatten()
                pB = path_points[i].flatten()
                
                v = pB - pA
                seg_len = np.linalg.norm(v)
                if seg_len == 0: continue
                u = v / seg_len
                
                for obs in self.raw_obstacle_meters: 
                    t = np.dot(obs - pA, u)
                    if 0 <= t <= seg_len:
                        proj_point = pA + t * u
                        dist_to_line = np.linalg.norm(obs - proj_point)
                        
                        if dist_to_line < safety_threshold:
                            print(f"  -> [WARNING] {path_name} Segment {i} is clipping!")
                            print(f"     Obstacle at {np.round(obs, 2)} is only {dist_to_line:.3f}m from the line.")
                            path_is_safe = False
                            
            if path_is_safe:
                print(f"  -> [PASS] {path_name} perfectly clears all physical bounds.")
            print("-" * 55)