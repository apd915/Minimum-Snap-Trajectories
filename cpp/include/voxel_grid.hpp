#pragma once

#include <Eigen/Dense>
#include <vector>
#include <unordered_set>
#include <cmath>

namespace trajectory_planner {
namespace mapping {

// ==========================================
// Custom Hash for Eigen::Vector3i
// ==========================================
struct Vector3iHash {
    std::size_t operator()(const Eigen::Vector3i& v) const {
        std::size_t h1 = std::hash<int>()(v.x());
        std::size_t h2 = std::hash<int>()(v.y());
        std::size_t h3 = std::hash<int>()(v.z());
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

// ==========================================
// Obstacle Bounding Box (Input Format)
// ==========================================
struct ObstacleBox {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3d min;
    Eigen::Vector3d max;
};

// ==========================================
// SparseVoxelGrid (Port of voxel_grid.py)
// ==========================================
class SparseVoxelGrid {
public:
    explicit SparseVoxelGrid(double resolution = 1.0);

    /**
     * Populates both inflated and raw voxel sets from a list of axis-aligned
     * obstacle bounding boxes. Each box is inflated by `inflation_radius` meters
     * to create the configuration space.
     */
    void populate_from_obstacles(const std::vector<ObstacleBox>& obstacles, double inflation_radius);

    /**
     * Returns true if the given voxel index is in the inflated occupied set.
     */
    bool is_occupied_inflated(const Eigen::Vector3i& idx) const;

    /**
     * Returns true if the given voxel index is in the raw (non-inflated) occupied set.
     */
    bool is_occupied_raw(const Eigen::Vector3i& idx) const;

    // Direct access to the underlying sets (for A* planner compatibility)
    const std::unordered_set<Eigen::Vector3i, Vector3iHash>& get_occupied_inflated() const { return occupied_voxels_inflated_; }
    const std::unordered_set<Eigen::Vector3i, Vector3iHash>& get_occupied_raw() const { return occupied_voxels_raw_; }

    // Continuous inflated bounds for the LoS Smoother (AABB raycaster)
    const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& get_continuous_inflated_bounds() const {
        return continuous_inflated_bounds_;
    }

    double get_resolution() const { return voxel_resolution_; }
    std::size_t num_inflated() const { return occupied_voxels_inflated_.size(); }
    std::size_t num_raw() const { return occupied_voxels_raw_.size(); }

private:
    double voxel_resolution_;
    std::unordered_set<Eigen::Vector3i, Vector3iHash> occupied_voxels_inflated_;
    std::unordered_set<Eigen::Vector3i, Vector3iHash> occupied_voxels_raw_;
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> continuous_inflated_bounds_;
};

} // namespace mapping
} // namespace trajectory_planner
