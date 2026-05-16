import numpy as np
import matplotlib.pyplot as plt
import itertools
from .node import Node    
import heapq
import math

class AStar_SFC_Planner:
    def __init__(self, voxel_grid, bounds):
        self.voxel_grid = voxel_grid
        res = self.voxel_grid.voxel_resolution

        # 1. Convert the entire list to an array, divide it all at once, and cast to integers!
        discrete_bounds = (np.array(bounds) // res).astype(int)

        # 2. Unpack the arrays cleanly into your class variables
        self.min_x_idx, self.max_x_idx = discrete_bounds[0]
        self.min_y_idx, self.max_y_idx = discrete_bounds[1]
        self.min_z_idx, self.max_z_idx = discrete_bounds[2]
        
        # Generate the 26 3D movement directions (-1, 0, 1 for X, Y, Z)
        self.directions = []
        for dx, dy, dz in itertools.product([-1, 0, 1], repeat=3):
            if dx == 0 and dy == 0 and dz == 0:
                continue # Skip the center point (we are already here!)
            
            # Pre-calculate the exact distance to this specific neighbor
            cost = math.sqrt(dx**2 + dy**2 + dz**2)
            self.directions.append((dx, dy, dz, cost)) # Store the cost in the tuple!

    
    def search(self, start_pos, goal_pos):
        # Force the inputs to be standard Python tuples
        start_pos = tuple(int(x) for x in np.ravel(start_pos))
        goal_pos = tuple(int(x) for x in np.ravel(goal_pos))

        # 1. Initialize Lists
        open_list = []
        closed_set = set() # O(1) lookups!
        
        # 2. Create the Start Node (f_cost, g_cost, h_cost, position, parent)
        start_node = Node(0, 0, 0, start_pos, None)
        heapq.heappush(open_list, start_node)
        
        nodes_expanded = 0
        # 3. Main Loop
        while len(open_list) > 0:
            nodes_expanded += 1 # Count every node we pop
            
            # a & b) Pop the node with the lowest F-Cost
            current_node = heapq.heappop(open_list)
            
            # --- THE LAZY DELETION CHECK ---
            if current_node.position in closed_set:
                continue 
                
            # d.i) Check if Goal is reached (Do this on POP, not generation!)
            if current_node.position == goal_pos:
                print(f"Path found! Explored {nodes_expanded} nodes.") # Print the result
                return self.reconstruct_path(current_node)
                
            # e) Push to CLOSED list
            closed_set.add(current_node.position)
            
            # c) Generate 26 Successors
            for dx, dy, dz, move_cost in self.directions:
                neighbor_pos = (
                    current_node.position[0] + dx,
                    current_node.position[1] + dy,
                    current_node.position[2] + dz
                )

                # --- 1. GEOFENCE BOUNDARY CHECK ---
                # Drop neighbors that exist outside our defined physical/LiDAR bounds
                if not (self.min_x_idx <= neighbor_pos[0] <= self.max_x_idx):
                    continue
                if not (self.min_y_idx <= neighbor_pos[1] <= self.max_y_idx):
                    continue
                if not (self.min_z_idx <= neighbor_pos[2] <= self.max_z_idx):
                    continue
                
                # --- COLLISION CHECK ---
                if neighbor_pos in self.voxel_grid.occupied_voxels_inflated:
                    continue # It's a building! Skip.
                    
                # d.iv) Skip if already fully evaluated
                if neighbor_pos in closed_set:
                    continue
                    
                # d.ii) Compute G, H, and F
                # 1. Use the instantly pre-calculated move_cost
                new_g = current_node.g_cost + move_cost
                
                # --- THE MAGIC NUMBER: WEIGHTED A* ---
                new_h = math.dist(neighbor_pos, goal_pos) * 2.0 
                new_f = new_g + new_h
                
                # Create the neighbor node and push it to the heap!
                neighbor_node = Node(new_f, new_g, new_h, neighbor_pos, current_node)
                heapq.heappush(open_list, neighbor_node)
                
        return None # Open list emptied, no path found

    def reconstruct_path(self, end_node):
        current_node = end_node
        self.path = []
        while current_node != None:
            self.path.append(current_node.position)
            current_node = current_node.parent
        
        self.path.reverse()
        return self.path
    
    def sfc_smoother(self):
        # If the path is only 2 points (Start and Goal), it's already a straight line!
        if not hasattr(self, 'path') or len(self.path) <= 2:
            return self.path 
            
        smoothed_path = [self.path[0]] # 1. Set the Anchor (Start Node)
        anchor_idx = 0
        
        # 2. The Look-Ahead Loop
        for i in range(2, len(self.path)):
            
            # 3. The Continuous Raycast Check 
            if not self.is_line_of_sight_clear(self.path[anchor_idx], self.path[i]):
                # The line hit a building. The previous node (i-1) is our new corner.
                smoothed_path.append(self.path[i-1])
                anchor_idx = i-1 
                
        # 4. Cap it off by adding the final Goal node
        smoothed_path.append(self.path[-1])
        
        # Overwrite the jagged path with our new, minimal waypoint list
        self.path = smoothed_path 
        return self.path
    
    def is_line_of_sight_clear(self, idx_a, idx_b):
        """
        Uses the continuous 3D Slab Method to check for intersections 
        between a line segment and all inflated bounding boxes.
        """
        res = self.voxel_grid.voxel_resolution
        
        # Convert integer indices back to continuous real-world meters
        p0 = np.array(idx_a) * res
        p1 = np.array(idx_b) * res
        
        # Direction vector of the ray
        d = p1 - p0
        
        # Calculate inverse direction for fast multiplication.
        # np.errstate suppresses warnings if d contains a zero (perfectly horizontal/vertical lines).
        # A division by zero results in 'inf', which mathematically works perfectly in the Slab Method!
        with np.errstate(divide='ignore'):
            inv_d = 1.0 / d
            
        for b_min, b_max in self.voxel_grid.continuous_inflated_bounds:
            # Calculate intersection 't' values for all 3 axes simultaneously
            t1 = (b_min - p0) * inv_d
            t2 = (b_max - p0) * inv_d
            
            # Find entry and exit times for each axis
            t_min = np.minimum(t1, t2)
            t_max = np.maximum(t1, t2)
            
            # t_enter is the latest time it enters a slab. t_exit is the earliest it exits.
            t_enter = np.max(t_min)
            t_exit = np.min(t_max)
            
            # Check for collision.
            # It hits the box IF it enters before it exits AND the collision happens 
            # between our two waypoints (t between 0.0 and 1.0).
            if t_enter <= t_exit and t_exit >= 0 and t_enter <= 1.0:
                return False # Collision detected!
                
        return True # Line of sight is completely clear

    def visualize_path(self):
        """
        Overlays the discrete A* path onto the continuous 3D Matplotlib plot.
        Must be called after search() has successfully found a path.
        """
        if not hasattr(self, 'path') or not self.path:
            print("No path to display. Run search() first.")
            return

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 1. Plot the Inflated Voxel Grid (C-Space)
        # occupied = self.voxel_grid.occupied_voxels_inflated
        # if len(occupied) > 0:
        #     x_idx, y_idx, z_idx = zip(*occupied)
            
        #     # Convert indices back to physical meters
        #     x_meters = np.array(x_idx) * res
        #     y_meters = np.array(y_idx) * res
        #     z_meters = np.array(z_idx) * res
            
        #     # alpha=0.15 makes the buildings slightly transparent
        #     ax.scatter(x_meters, y_meters, z_meters, color='red', marker='s', s=100, alpha=0.15)

        from rrt_mavsim.viewers.plot_map_path import PlotMapPath
        from rrt_mavsim.message_types.msg_world_map import MsgWorldMap, FloatingBlocksParams, MapTypes
        import rrt_mavsim.parameters.floatingBlocks_parameters as FLOATING_PARAM
        
        worldMap = MsgWorldMap(
            obstacleFieldType=MapTypes.FLOATING_BLOCKS,
            numDimensions_algorithm=FLOATING_PARAM.numDimensions,
            floatingBlocksParams=FloatingBlocksParams()
        )
        plotter = PlotMapPath(map=worldMap,
                      waypoints_smooth=None)

        plotter.plot_astar(ax, self.path, self.voxel_grid.voxel_resolution)



