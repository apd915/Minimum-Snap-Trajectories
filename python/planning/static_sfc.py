import numpy as np

# Defines the list of edge vertices indices from Msg_SFC
edges_verticesIndices_3D = [[0,1], [0,3], [0,4], [1,2], [1,5], [2,3], [2,6], [3,7], [4,5], [4,7], [5,6], [6,7]]
edgeVectorPairsIndices_list = [[1,0], [0,2], [3,4], [5,6], [2,1], [8,9]]

def euler_to_rotation(phi, theta, psi):
    """Converts euler angles to rotation matrix"""
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_psi = np.cos(psi)
    s_psi = np.sin(psi)

    R_roll = np.array([[1, 0, 0],
                       [0, c_phi, -s_phi],
                       [0, s_phi, c_phi]])
    R_pitch = np.array([[c_theta, 0, s_theta],
                        [0, 1, 0],
                        [-s_theta, 0, c_theta]])
    R_yaw = np.array([[c_psi, -s_psi, 0],
                      [s_psi, c_psi, 0],
                      [0, 0, 1]])
    return R_yaw @ R_pitch @ R_roll


class StandaloneFixedSFC:
    """
    A 100% self-contained equivalent to Msg_SFC from rrt_mavsim.
    It builds the 6 physical boundaries using exact cross-products.
    """
    def __init__(self, pA, pB, dimensions, translation, rotation, bounds):
        self.primaryPosition = pA
        self.secondaryPosition = pB
        self.dimensions = dimensions
        self.translation = translation
        self.rotation = rotation
        self.bounds = bounds
        self.numDimensions = 3
        
        # Duck-typing for PlotMapPath which expects corridor.sfc.getAllVertices_3D()
        self.sfc = self
        
        # Add local frame unit vectors (columns of rotation matrix)
        self.ux = self.rotation[:, 0]
        self.uy = self.rotation[:, 1]
        self.uz = self.rotation[:, 2]
        
        self.generateAbMatrices()

    def generateAbMatrices(self):
        normals, vertices = self.getNormalsVertices_3d()
        self.A = np.ndarray((0, self.numDimensions))
        self.b = np.ndarray((0, 1))

        for normal, vertex in zip(normals, vertices):
            self.A = np.concatenate((self.A, normal.T), axis=0)
            b_value = normal.T @ vertex
            self.b = np.concatenate((self.b, b_value), axis=0)

    def getNumDimensions(self):
        return self.numDimensions

    def getAbMatrices(self):
        return self.A, self.b
        
    def getDistancePrimaryToSecondary(self):
        return np.linalg.norm(self.secondaryPosition - self.primaryPosition)
        
    def getFlightCorridorLength(self):
        return self.dimensions[0,0]

    def getNormalsVertices_3d(self):
        finalVertices = self.getAllVertices_3D()
        return self.generate3DNormalsVertices(finalVertices)

    def getAllVertices_3D(self):
        x_dimension = self.dimensions[0,0]
        y_dimension = self.dimensions[1,0]
        z_dimension = self.dimensions[2,0]

        x_min, x_max = -x_dimension/2.0, x_dimension/2.0
        y_min, y_max = -y_dimension/2.0, y_dimension/2.0
        z_min, z_max = -z_dimension/2.0, z_dimension/2.0

        initialVertices = np.array([
            [x_min, x_max, x_max, x_min, x_min, x_max, x_max, x_min],
            [y_min, y_min, y_max, y_max, y_min, y_min, y_max, y_max],
            [z_min, z_min, z_min, z_min, z_max, z_max, z_max, z_max]
        ])

        rotatedVertices = self.rotation @ initialVertices
        translation_World = self.rotation @ self.translation
        finalVertices = rotatedVertices + translation_World
        return finalVertices

    def generate3DNormalsVertices(self, vertices):
        self.edgeLengths = []
        self.edgeVectors = []
        for edgeVertices in edges_verticesIndices_3D:
            startVertexIndex = edgeVertices[0]
            endVertexIndex = edgeVertices[1]
            startVertex_pos = vertices[:,startVertexIndex:(startVertexIndex+1)]
            endVertex_pos = vertices[:,endVertexIndex:(endVertexIndex+1)]
            currentEdgeVector = endVertex_pos - startVertex_pos
            currentEdgeVector_length = np.linalg.norm(currentEdgeVector)
            currentEdgeVector_norm = currentEdgeVector / currentEdgeVector_length
            self.edgeVectors.append(currentEdgeVector_norm)
            self.edgeLengths.append(currentEdgeVector_length)

        normalVectorsList = []
        verticesForNormalVectorsList = []

        for edgeVectorPairIndices in edgeVectorPairsIndices_list:
            primaryVector_index = edgeVectorPairIndices[0]
            secondaryVector_index = edgeVectorPairIndices[1]
            primaryVector = self.edgeVectors[primaryVector_index]
            secondaryVector = self.edgeVectors[secondaryVector_index]
            primaryVector_shape = np.shape(primaryVector)
            
            primaryVector_flattened = primaryVector.flatten()
            secondaryVector_flattened = secondaryVector.flatten()
            normalVector_temp_flattened = np.cross(primaryVector_flattened, secondaryVector_flattened)
            normalVector_temp = np.reshape(normalVector_temp_flattened, primaryVector_shape)
            normalVectorsList.append(normalVector_temp)

            primaryVectorVertices_indices = edges_verticesIndices_3D[primaryVector_index]
            primaryVector_startPositionIndex = primaryVectorVertices_indices[0]
            primaryVector_startPosition = vertices[:,primaryVector_startPositionIndex:(primaryVector_startPositionIndex+1)]
            verticesForNormalVectorsList.append(primaryVector_startPosition)

        return normalVectorsList, verticesForNormalVectorsList


class StaticSFCManager:
    """
    Initializes a completely self-contained manager for fixed-wing 
    (pre-sized) Safe Flight Corridors. Matches old MsgFlightCorridor logic perfectly.
    """
    def __init__(self, sfc_height, sfc_width):
        self.H = sfc_height
        self.W = sfc_width

    def generate_sfc(self, pA, pB, ext_start, ext_end):
        pA_col = np.reshape(pA, (3,1))
        pB_col = np.reshape(pB, (3,1))
        
        centerVector = pB_col - pA_col
        centerLength = np.linalg.norm(centerVector)
        
        if centerLength == 0:
            centerVectorNorm = np.array([[1.0], [0.0], [0.0]])
        else:
            centerVectorNorm = centerVector / centerLength

        length = centerLength + ext_start + ext_end
        dimensions = np.array([[length], [self.W], [self.H]])
        
        shift_distance = -ext_start + (length / 2.0)
        centerPosition = pA_col + centerVectorNorm * shift_distance
        
        centerVectorNorm_north = centerVectorNorm.item(0)
        centerVectorNorm_east = centerVectorNorm.item(1)
        centerVectorNorm_down = centerVectorNorm.item(2)
        
        yaw_angle = np.arctan2(centerVectorNorm_east, centerVectorNorm_north)
        centerVectorNorm_NEProjection = np.array([[centerVectorNorm_north], [centerVectorNorm_east], [0.0]])
        centerVectorNorm_NEProjection_length = np.linalg.norm(centerVectorNorm_NEProjection)
        pitch_angle = -np.arctan2(centerVectorNorm_down, centerVectorNorm_NEProjection_length)
        
        R_SFCToWorld = euler_to_rotation(0.0, pitch_angle, yaw_angle)
        R_WorldToSFC = R_SFCToWorld.T
        
        translation_SFC = R_WorldToSFC @ centerPosition

        # Bounds expected by the front-end diagnostics
        bounds = np.array([
            centerLength + ext_end, ext_start,
            self.W / 2.0, self.W / 2.0,
            self.H / 2.0, self.H / 2.0
        ])
        
        sfc = StandaloneFixedSFC(
            pA, pB, dimensions, translation_SFC, R_SFCToWorld, bounds
        )
        return sfc
