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

class FrontEndSFC:
    def __init__(self, map_type="FLOATING_BLOCKS", degree=4):
        """
        Initializes the environment and the RRT planner.
        """
        self.degree = degree
        
        # 1. Initialize the Map
        # Note: We are defaulting to floating blocks based on your test, 
        # but you can easily swap this logic to accept the CITY map later.
        # self.worldMap = MsgWorldMap(
        #     obstacleFieldType=MapTypes.FLOATING_BLOCKS,
        #     numDimensions_algorithm=FLOATING_PARAM.numDimensions,
        #     floatingBlocksParams=FloatingBlocksParams()
        # )

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
        self.voxel_resolution = 1 
        
        # Define your drone's inflation radius (e.g., 2.5m for a 5m wide SFC)
        self.inflation_radius = FLIGHT.width/2

        # Initialize your discrete grid (Assuming you build a SparseVoxelGrid class)
        self.discrete_grid = SparseVoxelGrid(resolution=self.voxel_resolution)


        # Populate the grid using the continuous obstacles
        continuous_obstacles = self.worldMap.get_obstacles()
        self.discrete_grid.populate_from_continuous(
            obstacles=continuous_obstacles, 
            inflation_radius=self.inflation_radius
        )
        # ---------------------------------------------------------

        occupied_inflated = self.discrete_grid.occupied_voxels_inflated
        occupied_raw = self.discrete_grid.occupied_voxels_raw

        # beginning = time.perf_counter()

        if len(occupied_inflated) > 0:
            x_idx, y_idx, z_idx = zip(*occupied_inflated)
            
            x_meters = np.array(x_idx) * self.voxel_resolution
            y_meters = np.array(y_idx) * self.voxel_resolution
            z_meters = np.array(z_idx) * self.voxel_resolution
            
            self.visualize(x_meters, y_meters, z_meters, style='')

        # total = time.perf_counter() - beginning
        # print(f"Plannning took: {total}\n")

        self.bounds = [(0,FLOATING_PARAM.northEnd), (0,FLOATING_PARAM.eastEnd), (0,FLOATING_PARAM.downEnd)]

        self.path_gen_astar = AStar_SFC_Planner(self.discrete_grid, self.bounds)

        # # 2. Initialize Dean's RRT Planner
        # self.path_gen = RRT_SFC_BSpline(
        #     numDimensions=FLOATING_PARAM.numDimensions,
        #     degree=self.degree,
        #     M=FLIGHT.M,
        #     Va=PLAN.Va0,
        #     rho=FLIGHT.rho,
        #     step_length=FLIGHT.segmentLength,
        #     numDesiredInitPaths=FLIGHT.numInitialPaths,
        #     # THE QUADROTOR HACK: We set chiMax to infinity. 
        #     # This disables Dean's fixed-wing turn radius limitations!
        #     chiMax=np.inf, 
        # )

    def get_corridors_astar(self, start_pos, end_pos, num_points_per_unit=FLIGHT.numPoints_perUnit):
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

        # 3. Build the MsgWaypoints_SFC object
        waypoints_smooth = MsgWaypoints_SFC(numDimensions=FLOATING_PARAM.numDimensions)
        
        # Add the Start point
        waypoints_smooth.add(
            position=continuous_path[0], 
            parent=np.inf, 
            cost=0.0, 
            connectsToGoal=False
        )

        # 4. Generate the Safe Flight Corridors (SFCs) along the path
        from rrt_mavsim.message_types.msg_flight_corridors import MsgFlightCorridor
        
        for i in range(1, len(continuous_path)):
            prev_pos = continuous_path[i-1]
            curr_pos = continuous_path[i]
            
            # Add the waypoint
            is_goal = (i == len(continuous_path) - 1)
            waypoints_smooth.add(
                position=curr_pos, 
                parent=i-1, 
                cost=0.0, 
                connectsToGoal=is_goal
            )
            
            # Create the SFC object. (Your system automatically applies the FLIGHT width here!)
            sfc = MsgFlightCorridor(
                primaryPosition=prev_pos,
                secondaryPosition=curr_pos,
                primaryPosition_index=i-1,
                numDimensions=FLOATING_PARAM.numDimensions
            )
            waypoints_smooth.addSFC(sfc)
            
        corridors = waypoints_smooth.getAllFlightCorridors()
        print(f"[Front-End] Extracted {len(corridors)} Safe Flight Corridors via A*.")

        # 5. Calculate Control Point Allocation (Using your existing tools!)
        from rrt_mavsim.tools.waypointsTools import getNumCntPts_list
        num_pts_list = getNumCntPts_list(
            waypoints=waypoints_smooth, 
            numPointsPerUnit=num_points_per_unit
        )

        # 6. Extract the A and b constraint matrices for the OSQP solver
        sfc_constraints = []
        for sfc in corridors:
            A_mat, b_vec = sfc.getAbMatrices()
            sfc_constraints.append({
                'A': A_mat, 
                'b': b_vec
            })
            
        # Return the exact same 4-variable tuple that RRT did!
        # (We return None for waypoints_not_smooth because A* doesn't need to keep the jagged path)
        return sfc_constraints, num_pts_list, waypoints_smooth, None
    

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


    def visualize(self, x_meters, y_meters, z_meters, style):
        if style == 'continuous':
            plotter_noWaypoints = PlotMapPath(
            map=self.worldMap,
            controlPoints_not_smooth_list=None,
            )

            plotter_noWaypoints.plot(
                x_limits=FLOATING_PARAM.x_limits,
                y_limits=FLOATING_PARAM.y_limits,
                z_limits=FLOATING_PARAM.z_limits,
                aspectRatio=FLOATING_PARAM.aspect_ratio,
            )
        elif style == 'discrete':
            # 1. Create the base figure (the window)
            fig = plt.figure()
            # 2. Add a 3D axis to the figure. This generates the 'ax' object!
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x_meters, y_meters, z_meters, color='red', marker='s')
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
            ax.set_zlabel('Z (Altitude)')
        else:
            return

        # 4. Render the window! (The code will pause here until you close the plot)
        plt.show()
