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
        Checks for clear line of sight. Prioritizes perfect continuous math if available,
        falls back to discrete voxel raycasting for LiDAR point clouds.
        """
        import numpy as np
        
        # --- 1. CONTINUOUS FRONT-END (Perfect Math) ---
        # If the simulation provided perfect bounding boxes, we absolutely want to use them!
        if hasattr(self.voxel_grid, 'continuous_inflated_bounds') and len(self.voxel_grid.continuous_inflated_bounds) > 0:
            res = self.voxel_grid.voxel_resolution
            p0 = np.array(idx_a) * res
            p1 = np.array(idx_b) * res
            d = p1 - p0
            
            with np.errstate(divide='ignore'):
                inv_d = 1.0 / d
                
            for b_min, b_max in self.voxel_grid.continuous_inflated_bounds:
                t1 = (b_min - p0) * inv_d
                t2 = (b_max - p0) * inv_d
                t_min = np.minimum(t1, t2)
                t_max = np.maximum(t1, t2)
                t_enter = np.max(t_min)
                t_exit = np.min(t_max)
                
                if t_enter <= t_exit and t_exit >= 0 and t_enter <= 1.0:
                    return False
            return True

        # --- 2. DISCRETE LIDAR (Voxel Raycasting) ---
        # If we are flying in a raw LiDAR map, we must use the discrete raycast.
        elif hasattr(self.voxel_grid, 'occupied_voxels_inflated'):
            p0 = np.array(idx_a)
            p1 = np.array(idx_b)
            dist = np.linalg.norm(p1 - p0)
            
            if dist == 0:
                return True
                
            # Upgraded step resolution: 5 steps per voxel prevents skipping diagonals
            steps = int(np.ceil(dist * 5))
            for i in range(1, steps):
                t = i / steps
                point = p0 + t * (p1 - p0)
                voxel = tuple(np.round(point).astype(int))
                
                if voxel in self.voxel_grid.occupied_voxels_inflated:
                    return False
            return True
            
        else:
            return True
    
    def get_safe_extension_length(self, idx_a, idx_b, requested_extension):
        """
        Fires a ray forward from idx_b to dynamically cap the extension 
        so it never penetrates an inflated obstacle. Hybrid function.
        """
        import numpy as np
        if requested_extension <= 0.0:
            return 0.0
            
        res = self.voxel_grid.voxel_resolution
        
        # --- 1. CONTINUOUS FRONT-END (Perfect Math) ---
        if hasattr(self.voxel_grid, 'continuous_inflated_bounds') and len(self.voxel_grid.continuous_inflated_bounds) > 0:
            p0 = np.array(idx_a) * res
            p1 = np.array(idx_b) * res
            
            d = p1 - p0
            dist = np.linalg.norm(d)
            if dist == 0: return 0.0
                
            dir_unit = d / dist 
            
            with np.errstate(divide='ignore'):
                inv_d = 1.0 / dir_unit
                
            min_safe_distance = requested_extension
            
            for b_min, b_max in self.voxel_grid.continuous_inflated_bounds:
                t1 = (b_min - p1) * inv_d
                t2 = (b_max - p1) * inv_d
                
                t_enter = np.max(np.minimum(t1, t2))
                t_exit = np.min(np.maximum(t1, t2))
                
                if t_enter <= t_exit and t_exit >= 0:
                    if t_enter > 0 and t_enter < min_safe_distance:
                        min_safe_distance = max(0.0, t_enter - 0.01)
            return min_safe_distance

        # --- 2. DISCRETE LIDAR (Voxel Raycasting) ---
        elif hasattr(self.voxel_grid, 'occupied_voxels_inflated'):
            p0 = np.array(idx_a) * res
            p1 = np.array(idx_b) * res
            
            d = p1 - p0
            dist = np.linalg.norm(d)
            if dist == 0: return 0.0
            
            dir_unit = d / dist
            
            # Step size of half a voxel for guaranteed collision detection
            step_size = res / 2.0
            num_steps = int(np.ceil(requested_extension / step_size))
            
            min_safe_distance = requested_extension
            
            for i in range(1, num_steps + 1):
                t_dist = i * step_size
                point = p1 + dir_unit * t_dist
                # Convert back to grid index
                voxel = tuple(np.round(point / res).astype(int))
                
                if voxel in self.voxel_grid.occupied_voxels_inflated:
                    # Collision detected! Cap the extension just before we hit the block
                    min_safe_distance = max(0.0, t_dist - step_size)
                    break
            return min_safe_distance
            
        else:
            return requested_extension

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
        occupied = self.voxel_grid.occupied_voxels_inflated
        if len(occupied) > 0:
            x_idx, y_idx, z_idx = zip(*occupied)
            
            # Convert indices back to physical meters
            x_meters = np.array(x_idx) * self.voxel_grid.voxel_resolution
            y_meters = np.array(y_idx) * self.voxel_grid.voxel_resolution
            z_meters = np.array(z_idx) * self.voxel_grid.voxel_resolution
            
            # alpha=0.15 makes the buildings slightly transparent
            ax.scatter(x_meters, y_meters, z_meters, color='red', marker='s', s=100, alpha=0.15)

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



