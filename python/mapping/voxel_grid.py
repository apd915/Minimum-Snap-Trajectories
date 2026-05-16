import numpy as np

class SparseVoxelGrid:
    def __init__(self, resolution=1.0):
        self.voxel_resolution = resolution
        
        # A Python set gives O(1) lookup time for A* (Lightning fast)
        self.occupied_voxels_inflated = set()
        self.occupied_voxels_raw = set()

        # Store the continuous floating-point bounds for the LoS Smoother
        self.continuous_inflated_bounds = []
    
    def populate_from_continuous(self, obstacles, inflation_radius):
        for obstacle in obstacles:
            vertices = obstacle.vertices_shifted_worldFrame_3D
            min_x, min_y, min_z = np.min(vertices, axis=1)
            max_x, max_y, max_z = np.max(vertices, axis=1)

            inflated_min = np.array([min_x - inflation_radius, min_y - inflation_radius, min_z - inflation_radius])
            inflated_max = np.array([max_x + inflation_radius, max_y + inflation_radius, max_z + inflation_radius])

            # Save the continuous box before converting to integers
            self.continuous_inflated_bounds.append((inflated_min, inflated_max))

            voxel_coordinates_inflated = ((inflated_min//self.voxel_resolution).astype(int), (inflated_max//self.voxel_resolution).astype(int))

            raw_min = np.array([min_x, min_y, min_z])
            raw_max = np.array([max_x, max_y, max_z])
            voxel_coordinates_raw = ((raw_min//self.voxel_resolution).astype(int), (raw_max//self.voxel_resolution).astype(int))

            # Unpacking makes the loops much easier to read!
            inf_min_idx, inf_max_idx = voxel_coordinates_inflated
            raw_min_idx, raw_max_idx = voxel_coordinates_raw

            for X in range(inf_min_idx[0], inf_max_idx[0]+1):
                for Y in range(inf_min_idx[1], inf_max_idx[1]+1):
                    for Z in range(inf_min_idx[2], inf_max_idx[2]+1):
                        self.occupied_voxels_inflated.add((X,Y,Z))

            for X in range(raw_min_idx[0], raw_max_idx[0]+1):
                for Y in range(raw_min_idx[1], raw_max_idx[1]+1):
                    for Z in range(raw_min_idx[2], raw_max_idx[2]+1):
                        self.occupied_voxels_raw.add((X,Y,Z))




        

        


