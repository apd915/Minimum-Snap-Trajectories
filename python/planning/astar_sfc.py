import numpy as np
import matplotlib.pyplot as plt
import itertools
from .node import Node    
import heapq
import math

class AStar_SFC_Planner:
    def __init__(self, voxel_grid):
        self.voxel_grid = voxel_grid
        
        # Generate the 26 3D movement directions (-1, 0, 1 for X, Y, Z)
        self.directions = []
        for dx, dy, dz in itertools.product([-1, 0, 1], repeat=3):
            if dx == 0 and dy == 0 and dz == 0:
                continue # Skip the center point (we are already here!)
            self.directions.append((dx, dy, dz))

    
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
        
        # 3. Main Loop
        while len(open_list) > 0:
            
            # a & b) Pop the node with the lowest F-Cost
            current_node = heapq.heappop(open_list)
            
            # --- THE LAZY DELETION CHECK ---
            if current_node.position in closed_set:
                continue 
                
            # d.i) Check if Goal is reached (Do this on POP, not generation!)
            if current_node.position == goal_pos:
                return self.reconstruct_path(current_node)
                
            # e) Push to CLOSED list
            closed_set.add(current_node.position)
            
            # c) Generate 26 Successors
            for dx, dy, dz in self.directions:
                neighbor_pos = (
                    current_node.position[0] + dx,
                    current_node.position[1] + dy,
                    current_node.position[2] + dz
                )
                
                # --- COLLISION CHECK ---
                if neighbor_pos in self.voxel_grid.occupied_voxels_inflated:
                    continue # It's a building! Skip.
                    
                # d.iv) Skip if already fully evaluated
                if neighbor_pos in closed_set:
                    continue
                    
                # d.ii) Compute G, H, and F
                move_cost = math.dist(current_node.position, neighbor_pos)
                new_g = current_node.g_cost + move_cost
                new_h = math.dist(neighbor_pos, goal_pos) # 3D Euclidean
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

        plotter.plot_astar(ax,self.path,self.voxel_grid.voxel_resolution)



