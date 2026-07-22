#pragma once

#include <Eigen/Dense>
#include <vector>
#include <array>
#include <cmath>

namespace trajectory_planner {
namespace sfc {

// ==========================================
// Euler to Rotation Matrix
// ==========================================
inline Eigen::Matrix3d euler_to_rotation(double phi, double theta, double psi) {
    double c_phi = std::cos(phi),   s_phi = std::sin(phi);
    double c_theta = std::cos(theta), s_theta = std::sin(theta);
    double c_psi = std::cos(psi),   s_psi = std::sin(psi);

    Eigen::Matrix3d R_roll;
    R_roll << 1, 0, 0,
              0, c_phi, -s_phi,
              0, s_phi,  c_phi;

    Eigen::Matrix3d R_pitch;
    R_pitch << c_theta, 0, s_theta,
               0,       1, 0,
              -s_theta, 0, c_theta;

    Eigen::Matrix3d R_yaw;
    R_yaw << c_psi, -s_psi, 0,
             s_psi,  c_psi, 0,
             0,      0,     1;

    return R_yaw * R_pitch * R_roll;
}

// ==========================================
// SFC Result Struct (Output of all SFC generators)
// ==========================================
struct SFCResult {
    Eigen::Vector3d primaryPosition;
    Eigen::Vector3d secondaryPosition;

    // OSQP Constraint Matrices (6 planes for OBB, or 6+6 for map-intersected)
    Eigen::MatrixXd A_mat;   // (num_planes x 3)
    Eigen::VectorXd b_vec;   // (num_planes)

    // 6 bounds: [+x_fwd, -x_back, +y_left, -y_right, +z_up, -z_down]
    Eigen::Matrix<double, 6, 1> bounds;

    // Local frame unit vectors
    Eigen::Vector3d ux, uy, uz;

    int numDimensions = 3;

    double getDistancePrimaryToSecondary() const {
        return (secondaryPosition - primaryPosition).norm();
    }

    /**
     * Returns the 8 corners of the OBB as a 3x8 matrix.
     * Vertex ordering matches the Python `getAllVertices_3D()` exactly.
     */
    Eigen::Matrix<double, 3, 8> getAllVertices_3D() const {
        double xmax = bounds(0), xmin = -bounds(1);
        double ymax = bounds(2), ymin = -bounds(3);
        double zmax = bounds(4), zmin = -bounds(5);

        std::array<double, 8> x_vals = {xmin, xmax, xmax, xmin, xmin, xmax, xmax, xmin};
        std::array<double, 8> y_vals = {ymin, ymin, ymax, ymax, ymin, ymin, ymax, ymax};
        std::array<double, 8> z_vals = {zmin, zmin, zmin, zmin, zmax, zmax, zmax, zmax};

        Eigen::Vector3d pA = primaryPosition;
        Eigen::Matrix<double, 3, 8> vertices;
        for (int i = 0; i < 8; ++i) {
            vertices.col(i) = pA + x_vals[i] * ux + y_vals[i] * uy + z_vals[i] * uz;
        }
        return vertices;
    }
};

// ==========================================
// Edge/Normal Definitions (Port of static_sfc.py constants)
// ==========================================
static const int edges_verticesIndices_3D[12][2] = {
    {0,1}, {0,3}, {0,4}, {1,2}, {1,5}, {2,3}, {2,6}, {3,7}, {4,5}, {4,7}, {5,6}, {6,7}
};
static const int edgeVectorPairsIndices_list[6][2] = {
    {1,0}, {0,2}, {3,4}, {5,6}, {2,1}, {8,9}
};

// ==========================================
// StaticSFCManager (Port of static_sfc.py)
// ==========================================
class StaticSFCManager {
public:
    StaticSFCManager(double sfc_height, double sfc_width);

    /**
     * Generates a pre-sized SFC between pA and pB with the given extensions.
     * Returns the complete SFCResult with A/b matrices computed via cross-product normals.
     */
    SFCResult generate_sfc(const Eigen::Vector3d& pA, const Eigen::Vector3d& pB,
                           double ext_start, double ext_end) const;

private:
    double H_, W_;

    /**
     * Generates the 6 A/b constraint matrices from the 8 vertices of the OBB.
     * Uses cross-product normals exactly as the Python implementation.
     */
    static void generateAbMatrices(const Eigen::Matrix<double, 3, 8>& vertices,
                                   Eigen::MatrixXd& A_out, Eigen::VectorXd& b_out);
};

} // namespace sfc
} // namespace trajectory_planner
