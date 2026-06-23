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
        self.kd_tree = KDTree(points) if len(points) > 0 else None

    def get_broad_phase_obstacles(self, pA, pB, W, ext_start, ext_end):
        """
        A Vectorized Discrete Cylinder Search.
        Executes a single O(log N) KD-Tree query, followed by an O(1) vectorized NumPy filter 
        to perfectly isolate discrete LiDAR points within the flight corridor.
        """
        if self.kd_tree is None:
            return []

        pA = np.ravel(pA)
        pB = np.ravel(pB)
        v = pB - pA
        dist = np.linalg.norm(v)
        
        # Calculate unit directional vector
        u = v / dist if dist > 0 else np.array([1.0, 0.0, 0.0])

        # ----------------------------------------------------
        # 1. DEFINE THE MATHEMATICAL CYLINDER
        # ----------------------------------------------------
        # Calculate the true start and end points including the runway extensions
        p_start = pA - (u * ext_start)
        p_end = pB + (u * ext_end)
        total_len = dist + ext_start + ext_end

        # ----------------------------------------------------
        # 2. THE BROAD-PHASE KD-TREE QUERY
        # ----------------------------------------------------
        # Calculate the exact midpoint of the extended segment
        midpoint = p_start + (p_end - p_start) / 2.0
        
        # The bounding sphere radius must fully enclose the cylinder
        search_radius = np.sqrt((total_len / 2.0)**2 + W**2)
        # search_radius = np.sqrt((total_len / 2.0)**2 + (W/2)**2 + (W/2)**2)

        # Execute exactly ONE query against the LiDAR map
        candidate_idxs = self.kd_tree.query_ball_point(midpoint, r=search_radius)
        if not candidate_idxs:
            return []

        candidates = self.kd_tree.data[candidate_idxs]

        # ----------------------------------------------------
        # 3. VECTORIZED DISCRETE FILTER (The Narrow Phase)
        # ----------------------------------------------------
        # Vector from the start of the cylinder to every candidate point
        vecs_to_candidates = candidates - p_start
        
        # Project all points onto the unit vector 'u' via dot product
        # This gives us the distance along the length of the cylinder
        t_vals = np.dot(vecs_to_candidates, u)
        
        # Mask 1: Keep only points that fall within the physical length of the cylinder
        mask_length = (t_vals >= 0) & (t_vals <= total_len)
        
        valid_candidates = candidates[mask_length]
        valid_t = t_vals[mask_length]
        
        if len(valid_candidates) == 0:
            return []

        # Calculate the exact perpendicular distance to the flight line
        # proj_points = p_start + (t * u)
        proj_points = p_start + valid_t[:, np.newaxis] * u
        perp_dists = np.linalg.norm(valid_candidates - proj_points, axis=1)
        
        # Mask 2: Keep only points that penetrate the lateral width 'W'
        mask_radius = perp_dists <= W
        
        # Return the pure, filtered numpy array of discrete obstacles
        return valid_candidates[mask_radius]
    

import numpy as np

class AsymmetricBoxBuilder:
    def __init__(self, drone_physical_radius, max_cp_drift, voxel_resolution=1.0):
        """
        Initializes the math engine with the physical constraints of the drone.
        """
        self.drone_radius = drone_physical_radius
        self.voxel_resolution = voxel_resolution
        self.max_drift = max_cp_drift

    def build_bounds(self, pA, pB, obs_points, ext_start, ext_end):
        """
        Calculates local axes and uses Radial Surface Detection to 
        optimally shrink asymmetric boundaries without crushing perpendicular axes.
        """
        pA = np.ravel(pA)
        pB = np.ravel(pB)
        dist = np.linalg.norm(pB - pA)

        ux = (pB - pA) / dist if dist > 0 else np.array([1.0, 0.0, 0.0])
        global_up = np.array([0.0, 0.0, 1.0])
        if np.abs(np.dot(ux, global_up)) > 0.99: global_up = np.array([1.0, 0.0, 0.0])
            
        uy = np.cross(global_up, ux)
        uy = uy / np.linalg.norm(uy) 
        uz = np.cross(ux, uy)        

        bounds = np.array([
            dist + ext_end, ext_start,        
            self.max_drift, self.max_drift,   
            self.max_drift, self.max_drift    
        ])

        # --- THE FIX: Exact OBB-AABB Projection (Separating Axis Theorem) ---
        # Calculates exactly how far the voxel's sharp corners stick out towards each specific SFC wall.
        half_res = self.voxel_resolution / 2.0
        r_x = self.drone_radius + half_res * (abs(ux[0]) + abs(ux[1]) + abs(ux[2]))
        r_y = self.drone_radius + half_res * (abs(uy[0]) + abs(uy[1]) + abs(uy[2]))
        r_z = self.drone_radius + half_res * (abs(uz[0]) + abs(uz[1]) + abs(uz[2]))

        projected_points = []
        for obs in obs_points:
            v = obs - pA
            proj_x = np.dot(v, ux)
            proj_y = np.dot(v, uy)
            proj_z = np.dot(v, uz)
            
            # Use dynamic r_x to fail fast!
            if -(bounds[1] + r_x) <= proj_x <= (bounds[0] + r_x):
                r_dist = np.sqrt(proj_y**2 + proj_z**2)
                projected_points.append((r_dist, proj_x, proj_y, proj_z))
                
        projected_points.sort(key=lambda x: x[0])

        for r_dist, proj_x, proj_y, proj_z in projected_points:
            eps = 1e-4
            # Use dynamic r_y and r_z to test active slices
            in_y_slice = -(bounds[3] + r_y) + eps < proj_y < (bounds[2] + r_y) - eps
            in_z_slice = -(bounds[5] + r_z) + eps < proj_z < (bounds[4] + r_z) - eps
            
            if not (in_y_slice and in_z_slice): continue 
                
            if 0 <= proj_x <= dist:
                # Use dynamic r_y and r_z to shrink bounds safely!
                if abs(proj_y) >= abs(proj_z):
                    if proj_y >= 0: bounds[2] = min(bounds[2], max(0.0, proj_y - r_y)) 
                    else:           bounds[3] = min(bounds[3], max(0.0, abs(proj_y) - r_y)) 
                else:
                    if proj_z >= 0: bounds[4] = min(bounds[4], max(0.0, proj_z - r_z)) 
                    else:           bounds[5] = min(bounds[5], max(0.0, abs(proj_z) - r_z)) 
            else:
                # Use dynamic r_x to shrink endcaps safely!
                if proj_x > dist:  bounds[0] = min(bounds[0], max(dist + 0.0, proj_x - r_x))
                elif proj_x < 0:   bounds[1] = min(bounds[1], max(0.0, abs(proj_x) - r_x))

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
        
        # --- THE FIX: Remove drone_radius to plot true Center of Mass boundaries! ---
        xmax, xmin = bounds[0], -bounds[1]
        ymax, ymin = bounds[2], -bounds[3]
        zmax, zmin = bounds[4], -bounds[5]
        
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
        
        # --- THE FIX: Remove self.r to plot true Center of Mass boundaries! ---
        xmax, xmin = self.bounds[0], -self.bounds[1]
        ymax, ymin = self.bounds[2], -self.bounds[3]
        zmax, zmin = self.bounds[4], -self.bounds[5]
        
        x_vals = [xmin, xmax, xmax, xmin, xmin, xmax, xmax, xmin]
        y_vals = [ymin, ymin, ymax, ymax, ymin, ymin, ymax, ymax]
        z_vals = [zmin, zmin, zmin, zmin, zmax, zmax, zmax, zmax]
        
        vertices = np.zeros((3, 8))
        for i in range(8):
            # Scale the local axis vectors and add to origin pA
            pt = pA + (x_vals[i] * self.ux) + (y_vals[i] * self.uy) + (z_vals[i] * self.uz)
            vertices[:, i] = pt
            
        return vertices