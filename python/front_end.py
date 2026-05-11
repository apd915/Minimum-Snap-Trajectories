import time
import numpy as np

# rrt_mavsim imports
from rrt_mavsim.message_types.msg_world_map import MsgWorldMap, FloatingBlocksParams, MapTypes
from rrt_mavsim.planners.rrt_sfc_bspline import RRT_SFC_BSpline
from rrt_mavsim.tools.waypointsTools import getNumCntPts_list
from rrt_mavsim.viewers.plot_map_path import PlotMapPath

# Parameter imports
import rrt_mavsim.parameters.planner_parameters as PLAN
import rrt_mavsim.parameters.floatingBlocks_parameters as FLOATING_PARAM
import rrt_mavsim.parameters.flightCorridor_parameters as FLIGHT

class FrontEndSFC:
    def __init__(self, map_type="FLOATING_BLOCKS", degree=4):
        """
        Initializes the environment and the RRT planner.
        """
        self.degree = degree
        
        # 1. Initialize the Map
        # Note: We are defaulting to floating blocks based on your test, 
        # but you can easily swap this logic to accept the CITY map later.
        self.worldMap = MsgWorldMap(
            obstacleFieldType=MapTypes.FLOATING_BLOCKS,
            numDimensions_algorithm=FLOATING_PARAM.numDimensions,
            floatingBlocksParams=FloatingBlocksParams()
        )

        # plotter_noWaypoints = PlotMapPath(
        #     map=self.worldMap,
        #     controlPoints_not_smooth_list=None,
        # )

        # plotter_noWaypoints.plot(
        #     x_limits=FLOATING_PARAM.x_limits,
        #     y_limits=FLOATING_PARAM.y_limits,
        #     z_limits=FLOATING_PARAM.z_limits,
        #     aspectRatio=FLOATING_PARAM.aspect_ratio,
        # )
        
        # 2. Initialize Dean's RRT Planner
        self.path_gen = RRT_SFC_BSpline(
            numDimensions=FLOATING_PARAM.numDimensions,
            degree=self.degree,
            M=FLIGHT.M,
            Va=PLAN.Va0,
            rho=FLIGHT.rho,
            step_length=FLIGHT.segmentLength,
            numDesiredInitPaths=FLIGHT.numInitialPaths,
            # THE QUADROTOR HACK: We set chiMax to infinity. 
            # This disables Dean's fixed-wing turn radius limitations!
            chiMax=np.inf, 
        )

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