import numpy as np

class SparseVoxelGrid:
    def __init__(self, resolution=1.0):
        self.voxel_resolution = resolution
        
        # A Python set gives O(1) lookup time for A* (Lightning fast)
        self.occupied_voxels = set()

    
    def populate_from_continuous(self, obstacles, inflation_radius):
        for obstacle in obstacles:
            vertices = obstacle.vertices_shifted_worldFrame_3D
            min_x, min_y, min_z = np.min(vertices, axis=1)
            max_x, max_y, max_z = np.max(vertices, axis=1)

            inflated_min = np.array([min_x - inflation_radius, min_y - inflation_radius, min_z - inflation_radius])
            inflated_max = np.array([max_x + inflation_radius, max_y + inflation_radius, max_z + inflation_radius])

            voxel_coordinates = ((inflated_min//self.voxel_resolution).astype(int), (inflated_max//self.voxel_resolution).astype(int))

            for X in range(voxel_coordinates[0][0], voxel_coordinates[1][0]+1):
                for Y in range(voxel_coordinates[0][1], voxel_coordinates[1][1]+1):
                    for Z in range(voxel_coordinates[0][2], voxel_coordinates[1][2]+1):
                        self.occupied_voxels.add((X,Y,Z))




        

        


