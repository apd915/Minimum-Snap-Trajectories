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


# ==========================================
# 2. THE FLIGHT CORRIDOR (Replaces msg_flight_corridors.py)
# ==========================================
class MsgDynamicFlightCorridor:
    """
    Axis-Aligned corridor that directly outputs OSQP matrices.
    """
    def __init__(self, numDimensions, primaryPosition, secondaryPosition, box_min, box_max):
        self.numDimensions = numDimensions
        self.primaryPosition = primaryPosition
        self.secondaryPosition = secondaryPosition
        self.box_min = box_min
        self.box_max = box_max
        self.sfc = self

    def getNumDimensions(self): return self.numDimensions
    
    # Dummy method to prevent downstream chained calls from breaking
    def getSFC(self): return self 

    def getAbMatrices(self):
        A = np.array([
            [ 1.0,  0.0,  0.0],
            [-1.0,  0.0,  0.0],
            [ 0.0,  1.0,  0.0],
            [ 0.0, -1.0,  0.0],
            [ 0.0,  0.0,  1.0],
            [ 0.0,  0.0, -1.0]
        ])
        b = np.array([
            self.box_max[0], -self.box_min[0],
            self.box_max[1], -self.box_min[1],
            self.box_max[2], -self.box_min[2]
        ])
        return A, b
        
    def getDistancePrimaryToSecondary(self) -> float:
        return np.linalg.norm(self.secondaryPosition - self.primaryPosition)

    def getAllVertices_3D(self):
        """
        Returns the 8 corners of the axis-aligned bounding box as a 3x8 numpy array.
        The columns are specifically ordered to match the legacy PlotMapPath face indices.
        """
        xmin, ymin, zmin = self.box_min
        xmax, ymax, zmax = self.box_max

        # Columns correspond to vertices 0 through 7
        vertices = np.array([
            [xmin, xmax, xmax, xmin, xmin, xmax, xmax, xmin], # X Coordinates
            [ymin, ymin, ymax, ymax, ymin, ymin, ymax, ymax], # Y Coordinates
            [zmin, zmin, zmin, zmin, zmax, zmax, zmax, zmax]  # Z Coordinates
        ])
        return vertices


# ==========================================
# 3. THE GEOMETRIC GENERATOR
# ==========================================
class DynamicSFCGenerator:
    def __init__(self, raw_uninflated_obstacle_points, drone_radius=0.25, max_radius=2.0):
        self.r = drone_radius
        self.max_r = max_radius
        points = np.array(list(raw_uninflated_obstacle_points))
        self.kd_tree = KDTree(points) if len(points) > 0 else None

    def generate_sfc_for_segment(self, pA, pB, start_ext_override=None, end_ext_override=None):
        pA = np.ravel(pA)
        pB = np.ravel(pB)
        
        seg_min = np.minimum(pA, pB)
        seg_max = np.maximum(pA, pB)
        
        ext_start = start_ext_override if start_ext_override else self.max_r
        ext_end = end_ext_override if end_ext_override else self.max_r
        max_search_dist = max(self.max_r, ext_start, ext_end)
        
        box_min = seg_min - max_search_dist
        box_max = seg_max + max_search_dist

        if self.kd_tree is None: return box_min, box_max
            
        dist = np.linalg.norm(pB - pA)
        dir_vec = (pB - pA) / dist if dist > 0 else np.zeros(3)
        
        sample_points = []
        for backward_dist in np.arange(0, ext_start + 0.1, self.max_r):
            sample_points.append(pA - (dir_vec * backward_dist))
        for forward_dist in np.arange(0, ext_end + 0.1, self.max_r):
            sample_points.append(pB + (dir_vec * forward_dist))
            
        num_internal_samples = max(2, int(np.ceil(dist / self.max_r)) + 1)
        for t in np.linspace(0, 1, num_internal_samples):
            sample_points.append(pA + t * (pB - pA))
            
        obs_points = []
        for pt in sample_points:
            dists, idxs = self.kd_tree.query(pt, k=100, distance_upper_bound=self.max_r)
            if np.isscalar(dists): dists, idxs = [dists], [idxs]
            for d, idx in zip(dists, idxs):
                if d != float('inf') and idx < len(self.kd_tree.data):
                    obs_points.append(self.kd_tree.data[idx])
                    
        if not obs_points: return box_min, box_max
        obs_points = np.unique(obs_points, axis=0)
        
        for obs in obs_points:
            if (box_min[1] <= obs[1] <= box_max[1]) and (box_min[2] <= obs[2] <= box_max[2]):
                if obs[0] > seg_max[0]: box_max[0] = min(box_max[0], obs[0] - self.r)
                elif obs[0] < seg_min[0]: box_min[0] = max(box_min[0], obs[0] + self.r)
            if (box_min[0] <= obs[0] <= box_max[0]) and (box_min[2] <= obs[2] <= box_max[2]):
                if obs[1] > seg_max[1]: box_max[1] = min(box_max[1], obs[1] - self.r)
                elif obs[1] < seg_min[1]: box_min[1] = max(box_min[1], obs[1] + self.r)
            if (box_min[0] <= obs[0] <= box_max[0]) and (box_min[1] <= obs[1] <= box_max[1]):
                if obs[2] > seg_max[2]: box_max[2] = min(box_max[2], obs[2] - self.r)
                elif obs[2] < seg_min[2]: box_min[2] = max(box_min[2], obs[2] + self.r)

        return box_min, box_max