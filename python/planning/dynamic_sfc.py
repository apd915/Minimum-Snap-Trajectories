import numpy as np
from scipy.spatial import KDTree

# ==========================================
# 1. THE WAYPOINT TRACKER (Replaces msg_waypoints.py)
# ==========================================
class StandaloneWaypointsSFC:
    """
    Dependency-free tracker for waypoints and their associated corridors.
    Mirrors the exact method names of the old rrt_mavsim class.
    """
    def __init__(self, numDimensions: int):
        self.numDimensions = numDimensions
        self.positions = []
        self.parents = []
        self.costs = []
        self.flightCorridors = []
        self.connectsToGoal = []
        self.numPositions = 0

    def add(self, position=None, parent=None, cost=None, connectsToGoal=None):
        if position is not None:
            self.positions.append(position)
            self.numPositions += 1
        if parent is not None: self.parents.append(parent)
        if cost is not None: self.costs.append(cost)
        if connectsToGoal is not None: self.connectsToGoal.append(connectsToGoal)

    def addSFC(self, sfc):
        if sfc is None: raise TypeError("None Type Detected for SFC")
        if sfc.getNumDimensions() != self.numDimensions:
            raise ValueError("Number of Dimensions is inconsistent")
        self.flightCorridors.append(sfc)

    def getAllPositions(self): return self.positions
    def getAllFlightCorridors(self): return self.flightCorridors
    
    def getAllSFCs(self):
        # Calls the dummy getSFC() method to mimic the old nested architecture
        return [fc.getSFC() for fc in self.flightCorridors]

    def getPosition(self, index: int): return self.positions[index]
    def getCost(self, index: int): return self.costs[index]
    def getParent(self, index: int): return self.parents[index]
    def getFlightCorridor(self, index: int): return self.flightCorridors[index]
    def getNumNodes(self): return len(self.positions)

    

import numpy as np
from scipy.spatial import KDTree

class AsymmetricSFCManager:
    def __init__(self, raw_uninflated_obstacle_points, drone_physical_radius, max_cp_drift, voxel_resolution=1.0):
        self.r = drone_physical_radius
        self.scanner = SpatialScannerKD(raw_uninflated_obstacle_points)
        
        # Pass the voxel resolution down to the math engine!
        self.builder = AsymmetricBoxBuilder(drone_physical_radius, max_cp_drift, voxel_resolution)

    def generate_sfc(self, pA, pB, W, ext_start, ext_end):
        # LAYER 1: Fetch raw points
        obs_points = self.scanner.get_broad_phase_obstacles(pA, pB, W, ext_start, ext_end)
        
        # LAYER 2: Calculate local frame and shrink bounds
        ux, uy, uz, bounds = self.builder.build_bounds(pA, pB, obs_points, ext_start, ext_end)
        
        # LAYER 3: Translate to OSQP Matrices
        A_mat, b_vec = OBBAdapters.get_osqp_matrices(pA, ux, uy, uz, bounds)
        
        # Create the duck-typed object for front_end.py
        sfc = MsgAsymmetricOBB(
            pA, pB, A_mat, b_vec, bounds, ux, uy, uz, self.r, numDimensions=3
        )
        return sfc

class SpatialScannerKD:
    def __init__(self, raw_uninflated_obstacle_points):
        """
        Initializes the spatial environment. 
        In C++, this maps directly to an octree or PCL KD-Tree initialization.
        """
        points = np.array(list(raw_uninflated_obstacle_points))
        # Build the tree once for the entire map
        self.kd_tree = KDTree(points) if len(points) > 0 else None

    def get_broad_phase_obstacles(self, pA, pB, W, ext_start, ext_end):
        """
        Drops W-spaced spheres along the vector and returns a deduplicated list 
        of all obstacles caught within the 3D Pythagorean radius.
        """
        if self.kd_tree is None:
            return []

        pA = np.ravel(pA)
        pB = np.ravel(pB)
        dist = np.linalg.norm(pB - pA)
        
        # Calculate the unit directional vector
        if dist == 0:
            ux = np.array([1.0, 0.0, 0.0])
        else:
            ux = (pB - pA) / dist

        # ----------------------------------------------------
        # THE GEOMETRIC CONSTANTS
        # ----------------------------------------------------
        spacing = W
        # The 3D Pythagorean Fix to guarantee zero blind spots
        r_search = W * (np.sqrt(3.0) / 2.0) 

        sample_points = []

        # 1. Backward Extension Sampling
        # Start at pA, walk backwards
        backward_dist = 0.0
        while backward_dist <= ext_start:
            sample_points.append(pA - (ux * backward_dist))
            backward_dist += spacing

        # 2. Main Segment Sampling
        # Start at pA, walk forwards to pB
        segment_dist = 0.0
        while segment_dist <= dist:
            sample_points.append(pA + (ux * segment_dist))
            segment_dist += spacing
            
        # Guarantee the exact end node is sampled
        sample_points.append(pB)

        # 3. Forward Extension Sampling
        # Start at pB, walk forwards
        forward_dist = spacing 
        while forward_dist <= ext_end:
            sample_points.append(pB + (ux * forward_dist))
            forward_dist += spacing

        # ----------------------------------------------------
        # THE KD-TREE QUERY
        # ----------------------------------------------------
        obs_points_set = set()
        
        for pt in sample_points:
            # We use query_ball_point instead of k=100. 
            # This guarantees we fetch ALL obstacles in the radius, 
            # completely eliminating the risk of missing a dense cluster of trees.
            idxs = self.kd_tree.query_ball_point(pt, r=r_search)
            for idx in idxs:
                # Add to a set as a tuple to automatically remove duplicates
                obs_points_set.add(tuple(self.kd_tree.data[idx]))
                
        # Convert the deduplicated tuples back into numpy arrays for Layer 2
        return [np.array(pt) for pt in obs_points_set]
    

import numpy as np

class AsymmetricBoxBuilder:
    def __init__(self, drone_physical_radius, max_cp_drift, voxel_resolution=1.0):
        """
        Initializes the math engine with the physical constraints of the drone.
        """
        self.r = drone_physical_radius + (voxel_resolution / 2.0)
        self.max_drift = max_cp_drift

    def build_bounds(self, pA, pB, obs_points, ext_start, ext_end):
        """
        Calculates local axes and shrinks the maximum asymmetric boundaries.
        Returns: ux, uy, uz, bounds_array
        """
        pA = np.ravel(pA)
        pB = np.ravel(pB)
        dist = np.linalg.norm(pB - pA)

        # ----------------------------------------------------
        # 1. ESTABLISH LOCAL COORDINATE FRAME
        # ----------------------------------------------------
        ux = (pB - pA) / dist if dist > 0 else np.array([1.0, 0.0, 0.0])
        
        global_up = np.array([0.0, 0.0, 1.0])
        # Gimbal lock protection: if flying straight up/down, swap the reference vector
        if np.abs(np.dot(ux, global_up)) > 0.99:
            global_up = np.array([1.0, 0.0, 0.0])
            
        uy = np.cross(global_up, ux)
        uy = uy / np.linalg.norm(uy) # Normalize lateral vector
        uz = np.cross(ux, uy)        # Vertical vector is orthogonal to both

        # ----------------------------------------------------
        # 2. INITIALIZE MAXIMUM BOUNDS 
        # Array format: [ +X, -X, +Y, -Y, +Z, -Z ]
        # ----------------------------------------------------
        bounds = np.array([
            dist + ext_end,   # +X (Forward Runway)
            ext_start,        # -X (Backward Runway)
            self.max_drift,   # +Y (Left Wall)
            self.max_drift,   # -Y (Right Wall)
            self.max_drift,   # +Z (Ceiling)
            self.max_drift    # -Z (Floor)
        ])

        # ----------------------------------------------------
        # 3. NARROW-PHASE PROJECTION & SMART SHRINK
        # ----------------------------------------------------
        for obs in obs_points:
            v = obs - pA
            
            # Project global coordinate onto local axes via Dot Product
            proj_x = np.dot(v, ux)
            proj_y = np.dot(v, uy)
            proj_z = np.dot(v, uz)
            
            # Fast-fail: Is the obstacle totally outside our maximum possible length?
            if not (-(bounds[1] + self.r) <= proj_x <= (bounds[0] + self.r)):
                continue 

            # Check if obstacle is inside the current Y/Z cross-section 
            in_y_slice = -(bounds[3] + self.r) <= proj_y <= (bounds[2] + self.r)
            in_z_slice = -(bounds[5] + self.r) <= proj_z <= (bounds[4] + self.r)
                
            # SCENARIO A: Obstacle is adjacent to the main A* segment
            if 0 <= proj_x <= dist:
                # Shrink Lateral Walls (Y) if obstacle is within vertical slice
                if in_z_slice: 
                    if proj_y > 0:
                        bounds[2] = min(bounds[2], max(0.1, proj_y - self.r)) 
                    elif proj_y < 0:
                        bounds[3] = min(bounds[3], max(0.1, abs(proj_y) - self.r)) 
                
                # Shrink Vertical Walls (Z) if obstacle is within lateral slice
                if in_y_slice: 
                    if proj_z > 0:
                        bounds[4] = min(bounds[4], max(0.1, proj_z - self.r)) 
                    elif proj_z < 0:
                        bounds[5] = min(bounds[5], max(0.1, abs(proj_z) - self.r)) 
                        
            # SCENARIO B: Obstacle is in the start/end Extensions
            else:
                # Only truncate X-axis length if it is directly inside the tunnel
                if in_y_slice and in_z_slice:
                    if proj_x > dist: # Obstacle is ahead
                        bounds[0] = min(bounds[0], max(dist + 0.1, proj_x - self.r))
                    elif proj_x < 0:  # Obstacle is behind
                        bounds[1] = min(bounds[1], max(0.1, abs(proj_x) - self.r))

        return ux, uy, uz, bounds
    

import numpy as np

class OBBAdapters:
    @staticmethod
    def get_osqp_matrices(pA, ux, uy, uz, bounds):
        """
        Translates Layer 2 vectors into OSQP linear inequality constraints (Ax <= b).
        """
        # Formulate the A Matrix (The 6 Outward Normals)
        A = np.vstack([ux, -ux, uy, -uy, uz, -uz])
        
        # Calculate the b Vector (Global offset distances)
        b_base = A @ np.ravel(pA)
        b_offset = np.array([
            bounds[0], bounds[1], bounds[2], 
            bounds[3], bounds[4], bounds[5]
        ])
        
        b = b_base + b_offset
        return A, b

    @staticmethod
    def get_visual_vertices(pA, ux, uy, uz, bounds, drone_radius):
        """
        Translates Layer 2 vectors into 8 physical 3D corners for Matplotlib/Rviz.
        """
        pA = np.ravel(pA)
        
        # Expand the bounds back out to their physical dimensions for plotting
        xmax, xmin = bounds[0] + drone_radius, -(bounds[1] + drone_radius)
        ymax, ymin = bounds[2] + drone_radius, -(bounds[3] + drone_radius)
        zmax, zmin = bounds[4] + drone_radius, -(bounds[5] + drone_radius)
        
        # Hardcoded combinations to match Matplotlib's expected face indexing
        x_vals = [xmin, xmax, xmax, xmin, xmin, xmax, xmax, xmin]
        y_vals = [ymin, ymin, ymax, ymax, ymin, ymin, ymax, ymax]
        z_vals = [zmin, zmin, zmin, zmin, zmax, zmax, zmax, zmax]
        
        vertices = np.zeros((3, 8))
        for i in range(8):
            # Start at pA, walk along the local axes to reach the corner
            corner = pA + (x_vals[i] * ux) + (y_vals[i] * uy) + (z_vals[i] * uz)
            vertices[:, i] = corner
            
        return vertices
    

class MsgAsymmetricOBB:
    """
    Oriented Bounding Box that directly outputs OSQP matrices and 
    Matplotlib-compatible vertices for rendering.
    """
    def __init__(self, pA, pB, A_mat, b_vec, bounds, ux, uy, uz, drone_radius, numDimensions=3):
        self.numDimensions = numDimensions
        self.primaryPosition = pA.reshape(3, 1)
        self.secondaryPosition = pB.reshape(3, 1)
        self.A_mat = A_mat
        self.b_vec = b_vec
        self.bounds = bounds
        self.ux = ux
        self.uy = uy
        self.uz = uz
        self.r = drone_radius
        
        # Plotter Duck-Typing
        self.sfc = self 

    def getNumDimensions(self): return self.numDimensions
    def getSFC(self): return self 
    def getAbMatrices(self): return self.A_mat, self.b_vec
        
    def getDistancePrimaryToSecondary(self) -> float:
        return np.linalg.norm(self.secondaryPosition - self.primaryPosition)

    def getAllVertices_3D(self):
        """
        Returns the 8 corners of the Asymmetric OBB as a 3x8 numpy array.
        Mapped to the exact vertex connections the legacy Matplotlib viewer expects.
        """
        pA = self.primaryPosition.flatten()
        
        # Add the physical drone radius back so the plotted boxes represent the actual safe flight space
        xmax, xmin = self.bounds[0] + self.r, -(self.bounds[1] + self.r)
        ymax, ymin = self.bounds[2] + self.r, -(self.bounds[3] + self.r)
        zmax, zmin = self.bounds[4] + self.r, -(self.bounds[5] + self.r)
        
        x_vals = [xmin, xmax, xmax, xmin, xmin, xmax, xmax, xmin]
        y_vals = [ymin, ymin, ymax, ymax, ymin, ymin, ymax, ymax]
        z_vals = [zmin, zmin, zmin, zmin, zmax, zmax, zmax, zmax]
        
        vertices = np.zeros((3, 8))
        for i in range(8):
            # Scale the local axis vectors and add to origin pA
            pt = pA + (x_vals[i] * self.ux) + (y_vals[i] * self.uy) + (z_vals[i] * self.uz)
            vertices[:, i] = pt
            
        return vertices