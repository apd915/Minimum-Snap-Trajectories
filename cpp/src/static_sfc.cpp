#include "static_sfc.hpp"
#include <cmath>

namespace trajectory_planner {
namespace sfc {

StaticSFCManager::StaticSFCManager(double sfc_height, double sfc_width)
    : H_(sfc_height), W_(sfc_width) {}

SFCResult StaticSFCManager::generate_sfc(
    const Eigen::Vector3d& pA, const Eigen::Vector3d& pB,
    double ext_start, double ext_end) const {

    Eigen::Vector3d centerVector = pB - pA;
    double centerLength = centerVector.norm();

    Eigen::Vector3d centerVectorNorm;
    if (centerLength == 0.0) {
        centerVectorNorm = Eigen::Vector3d(1.0, 0.0, 0.0);
    } else {
        centerVectorNorm = centerVector / centerLength;
    }

    double length = centerLength + ext_start + ext_end;
    Eigen::Vector3d dimensions(length, W_, H_);

    double shift_distance = -ext_start + (length / 2.0);
    Eigen::Vector3d centerPosition = pA + centerVectorNorm * shift_distance;

    double north = centerVectorNorm(0);
    double east  = centerVectorNorm(1);
    double down  = centerVectorNorm(2);

    double yaw_angle = std::atan2(east, north);
    Eigen::Vector3d ne_proj(north, east, 0.0);
    double ne_proj_len = ne_proj.norm();
    double pitch_angle = -std::atan2(down, ne_proj_len);

    Eigen::Matrix3d R_SFCToWorld = euler_to_rotation(0.0, pitch_angle, yaw_angle);
    Eigen::Matrix3d R_WorldToSFC = R_SFCToWorld.transpose();

    Eigen::Vector3d translation_SFC = R_WorldToSFC * centerPosition;

    // Build the 8 initial vertices (local frame, centered at origin)
    double x_min = -length / 2.0, x_max = length / 2.0;
    double y_min = -W_ / 2.0,     y_max = W_ / 2.0;
    double z_min = -H_ / 2.0,     z_max = H_ / 2.0;

    Eigen::Matrix<double, 3, 8> initialVertices;
    // Exact same vertex ordering as Python
    initialVertices.col(0) = Eigen::Vector3d(x_min, y_min, z_min);
    initialVertices.col(1) = Eigen::Vector3d(x_max, y_min, z_min);
    initialVertices.col(2) = Eigen::Vector3d(x_max, y_max, z_min);
    initialVertices.col(3) = Eigen::Vector3d(x_min, y_max, z_min);
    initialVertices.col(4) = Eigen::Vector3d(x_min, y_min, z_max);
    initialVertices.col(5) = Eigen::Vector3d(x_max, y_min, z_max);
    initialVertices.col(6) = Eigen::Vector3d(x_max, y_max, z_max);
    initialVertices.col(7) = Eigen::Vector3d(x_min, y_max, z_max);

    // Rotate and translate to world frame
    Eigen::Matrix<double, 3, 8> rotatedVertices = R_SFCToWorld * initialVertices;
    Eigen::Vector3d translation_World = R_SFCToWorld * translation_SFC;
    Eigen::Matrix<double, 3, 8> finalVertices = rotatedVertices.colwise() + translation_World;

    // Generate A/b matrices from the cross-product normals
    Eigen::MatrixXd A_mat;
    Eigen::VectorXd b_vec;
    generateAbMatrices(finalVertices, A_mat, b_vec);

    // Build bounds
    Eigen::Matrix<double, 6, 1> bounds;
    bounds << centerLength + ext_end, ext_start,
              W_ / 2.0, W_ / 2.0,
              H_ / 2.0, H_ / 2.0;

    // Build the local frame vectors (columns of the rotation matrix)
    Eigen::Vector3d ux = R_SFCToWorld.col(0);
    Eigen::Vector3d uy = R_SFCToWorld.col(1);
    Eigen::Vector3d uz = R_SFCToWorld.col(2);

    SFCResult result;
    result.primaryPosition = pA;
    result.secondaryPosition = pB;
    result.A_mat = A_mat;
    result.b_vec = b_vec;
    result.bounds = bounds;
    result.ux = ux;
    result.uy = uy;
    result.uz = uz;

    return result;
}

void StaticSFCManager::generateAbMatrices(
    const Eigen::Matrix<double, 3, 8>& vertices,
    Eigen::MatrixXd& A_out, Eigen::VectorXd& b_out) {

    // Compute the 12 edge vectors
    std::array<Eigen::Vector3d, 12> edgeVectors;
    for (int i = 0; i < 12; ++i) {
        int s = edges_verticesIndices_3D[i][0];
        int e = edges_verticesIndices_3D[i][1];
        Eigen::Vector3d edgeVec = vertices.col(e) - vertices.col(s);
        double edgeLen = edgeVec.norm();
        edgeVectors[i] = edgeVec / edgeLen;
    }

    // Compute the 6 normals from the 6 edge vector pairs
    A_out.resize(6, 3);
    b_out.resize(6);

    for (int i = 0; i < 6; ++i) {
        int pi = edgeVectorPairsIndices_list[i][0];
        int si = edgeVectorPairsIndices_list[i][1];

        Eigen::Vector3d normal = edgeVectors[pi].cross(edgeVectors[si]);
        A_out.row(i) = normal.transpose();

        // The vertex used for the b value is the start vertex of the primary edge
        int startVertexIdx = edges_verticesIndices_3D[pi][0];
        Eigen::Vector3d vertex = vertices.col(startVertexIdx);

        b_out(i) = normal.dot(vertex);
    }
}

} // namespace sfc
} // namespace trajectory_planner
