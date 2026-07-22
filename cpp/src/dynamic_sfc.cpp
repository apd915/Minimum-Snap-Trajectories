#include "dynamic_sfc.hpp"
#include <cmath>
#include <algorithm>

namespace trajectory_planner {
namespace sfc {

// ==========================================
// SpatialScanner Implementation
// ==========================================
SpatialScanner::SpatialScanner(const std::vector<Eigen::Vector3d>& obstacle_points)
    : points_(obstacle_points) {}

std::vector<Eigen::Vector3d> SpatialScanner::get_broad_phase_obstacles(
    const Eigen::Vector3d& pA, const Eigen::Vector3d& pB,
    double W, double ext_start, double ext_end) const {

    if (points_.empty()) return {};

    Eigen::Vector3d v = pB - pA;
    double dist = v.norm();
    Eigen::Vector3d u = (dist > 0) ? v / dist : Eigen::Vector3d(1.0, 0.0, 0.0);

    // Define the mathematical cylinder
    Eigen::Vector3d p_start = pA - u * ext_start;
    double total_len = dist + ext_start + ext_end;

    // Broad-phase: bounding sphere
    Eigen::Vector3d midpoint = p_start + u * (total_len / 2.0);
    double search_radius = std::sqrt((total_len / 2.0) * (total_len / 2.0) + W * W);

    std::vector<Eigen::Vector3d> result;
    result.reserve(points_.size() / 4); // Reasonable pre-alloc

    for (const auto& pt : points_) {
        // Broad-phase sphere check
        if ((pt - midpoint).norm() > search_radius) continue;

        // Narrow-phase: project onto cylinder axis
        Eigen::Vector3d vec_to_pt = pt - p_start;
        double t = vec_to_pt.dot(u);

        // Length filter
        if (t < 0.0 || t > total_len) continue;

        // Perpendicular distance filter
        Eigen::Vector3d proj_point = p_start + t * u;
        double perp_dist = (pt - proj_point).norm();
        if (perp_dist <= W) {
            result.push_back(pt);
        }
    }

    return result;
}

// ==========================================
// AsymmetricBoxBuilder Implementation
// ==========================================
AsymmetricBoxBuilder::AsymmetricBoxBuilder(double drone_physical_radius, double max_cp_drift, double voxel_resolution)
    : drone_radius_(drone_physical_radius), max_drift_(max_cp_drift), voxel_resolution_(voxel_resolution) {}

std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d, Eigen::Matrix<double, 6, 1>>
AsymmetricBoxBuilder::build_bounds(
    const Eigen::Vector3d& pA, const Eigen::Vector3d& pB,
    const std::vector<Eigen::Vector3d>& obs_points,
    double ext_start, double ext_end) const {

    double dist = (pB - pA).norm();
    Eigen::Vector3d ux = (dist > 0) ? (pB - pA) / dist : Eigen::Vector3d(1.0, 0.0, 0.0);

    Eigen::Vector3d global_up(0.0, 0.0, 1.0);
    if (std::abs(ux.dot(global_up)) > 0.99) global_up = Eigen::Vector3d(1.0, 0.0, 0.0);

    Eigen::Vector3d uy = global_up.cross(ux);
    uy.normalize();
    Eigen::Vector3d uz = ux.cross(uy);

    Eigen::Matrix<double, 6, 1> bounds;
    bounds << dist + ext_end, ext_start,
              max_drift_, max_drift_,
              max_drift_, max_drift_;

    // Exact OBB-AABB Projection (SAT)
    double half_res = voxel_resolution_ / 2.0;
    double r_x = drone_radius_ + half_res * (std::abs(ux.x()) + std::abs(ux.y()) + std::abs(ux.z()));
    double r_y = drone_radius_ + half_res * (std::abs(uy.x()) + std::abs(uy.y()) + std::abs(uy.z()));
    double r_z = drone_radius_ + half_res * (std::abs(uz.x()) + std::abs(uz.y()) + std::abs(uz.z()));

    // Project and sort by true segment distance
    struct ProjectedPoint {
        double seg_dist, proj_x, proj_y, proj_z;
    };
    std::vector<ProjectedPoint> projected;
    projected.reserve(obs_points.size());

    for (const auto& obs : obs_points) {
        Eigen::Vector3d v_obs = obs - pA;
        double px = v_obs.dot(ux);
        double py = v_obs.dot(uy);
        double pz = v_obs.dot(uz);

        if (px >= -(bounds(1) + r_x) && px <= (bounds(0) + r_x)) {
            double seg_dist;
            if (px < 0) {
                seg_dist = std::sqrt(px * px + py * py + pz * pz);
            } else if (px > dist) {
                seg_dist = std::sqrt((px - dist) * (px - dist) + py * py + pz * pz);
            } else {
                seg_dist = std::sqrt(py * py + pz * pz);
            }
            projected.push_back({seg_dist, px, py, pz});
        }
    }

    std::sort(projected.begin(), projected.end(),
              [](const ProjectedPoint& a, const ProjectedPoint& b) { return a.seg_dist < b.seg_dist; });

    // Iteratively shrink the 6 planes
    for (const auto& pt : projected) {
        double eps = 1e-4;
        double proj_x = pt.proj_x, proj_y = pt.proj_y, proj_z = pt.proj_z;

        // 6-Plane active slice check
        bool in_x = -(bounds(1) + r_x) + eps < proj_x && proj_x < (bounds(0) + r_x) - eps;
        bool in_y = -(bounds(3) + r_y) + eps < proj_y && proj_y < (bounds(2) + r_y) - eps;
        bool in_z = -(bounds(5) + r_z) + eps < proj_z && proj_z < (bounds(4) + r_z) - eps;

        if (!(in_x && in_y && in_z)) continue;

        if (proj_x >= 0.0 && proj_x <= dist) {
            // Inside main tunnel
            if (std::abs(proj_y) >= std::abs(proj_z)) {
                if (proj_y >= 0) bounds(2) = std::min(bounds(2), std::max(0.0, proj_y - r_y));
                else             bounds(3) = std::min(bounds(3), std::max(0.0, std::abs(proj_y) - r_y));
            } else {
                if (proj_z >= 0) bounds(4) = std::min(bounds(4), std::max(0.0, proj_z - r_z));
                else             bounds(5) = std::min(bounds(5), std::max(0.0, std::abs(proj_z) - r_z));
            }
        } else {
            // End-caps: 6-Plane equality
            double dist_x = (proj_x < 0) ? std::abs(proj_x) : std::abs(proj_x - dist);

            if (dist_x >= std::abs(proj_y) && dist_x >= std::abs(proj_z)) {
                if (proj_x > dist)  bounds(0) = std::min(bounds(0), std::max(dist + 0.0, proj_x - r_x));
                else if (proj_x < 0) bounds(1) = std::min(bounds(1), std::max(0.0, std::abs(proj_x) - r_x));
            } else if (std::abs(proj_y) >= std::abs(proj_z)) {
                if (proj_y >= 0) bounds(2) = std::min(bounds(2), std::max(0.0, proj_y - r_y));
                else             bounds(3) = std::min(bounds(3), std::max(0.0, std::abs(proj_y) - r_y));
            } else {
                if (proj_z >= 0) bounds(4) = std::min(bounds(4), std::max(0.0, proj_z - r_z));
                else             bounds(5) = std::min(bounds(5), std::max(0.0, std::abs(proj_z) - r_z));
            }
        }
    }

    return {ux, uy, uz, bounds};
}

// ==========================================
// OBBAdapters Implementation
// ==========================================
void OBBAdapters::get_osqp_matrices(
    const Eigen::Vector3d& pA,
    const Eigen::Vector3d& ux, const Eigen::Vector3d& uy, const Eigen::Vector3d& uz,
    const Eigen::Matrix<double, 6, 1>& bounds,
    Eigen::MatrixXd& A_out, Eigen::VectorXd& b_out) {

    // A = [ux; -ux; uy; -uy; uz; -uz]
    A_out.resize(6, 3);
    A_out.row(0) =  ux.transpose();
    A_out.row(1) = -ux.transpose();
    A_out.row(2) =  uy.transpose();
    A_out.row(3) = -uy.transpose();
    A_out.row(4) =  uz.transpose();
    A_out.row(5) = -uz.transpose();

    // b_base = A @ pA
    Eigen::VectorXd b_base = A_out * pA;

    // b = b_base + bounds
    b_out = b_base + bounds;
}

// ==========================================
// AsymmetricSFCManager Implementation
// ==========================================
AsymmetricSFCManager::AsymmetricSFCManager(
    const std::vector<Eigen::Vector3d>& raw_uninflated_obstacle_points,
    double drone_physical_radius,
    double max_cp_drift,
    double voxel_resolution)
    : scanner_(raw_uninflated_obstacle_points),
      builder_(drone_physical_radius, max_cp_drift, voxel_resolution) {}

SFCResult AsymmetricSFCManager::generate_sfc(
    const Eigen::Vector3d& pA, const Eigen::Vector3d& pB,
    double W, double ext_start, double ext_end) const {

    // Layer 1: Fetch raw points
    auto obs_points = scanner_.get_broad_phase_obstacles(pA, pB, W, ext_start, ext_end);

    // Layer 2: Calculate local frame and shrink bounds
    auto [ux, uy, uz, bounds] = builder_.build_bounds(pA, pB, obs_points, ext_start, ext_end);

    // Layer 3: Translate to OSQP matrices
    Eigen::MatrixXd A_mat;
    Eigen::VectorXd b_vec;
    OBBAdapters::get_osqp_matrices(pA, ux, uy, uz, bounds, A_mat, b_vec);

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

} // namespace sfc
} // namespace trajectory_planner
