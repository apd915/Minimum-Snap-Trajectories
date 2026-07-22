#include "voxel_grid.hpp"
#include <cmath>

namespace trajectory_planner {
namespace mapping {

SparseVoxelGrid::SparseVoxelGrid(double resolution)
    : voxel_resolution_(resolution) {}

void SparseVoxelGrid::populate_from_obstacles(
    const std::vector<ObstacleBox>& obstacles, double inflation_radius) {
    
    for (const auto& obstacle : obstacles) {
        // Inflated bounds
        Eigen::Vector3d inflated_min = obstacle.min.array() - inflation_radius;
        Eigen::Vector3d inflated_max = obstacle.max.array() + inflation_radius;

        // Save the continuous inflated bounds for the LoS smoother
        continuous_inflated_bounds_.emplace_back(inflated_min, inflated_max);

        // Convert to voxel indices (floor division, matching Python's // operator)
        Eigen::Vector3i inf_min_idx(
            static_cast<int>(std::floor(inflated_min.x() / voxel_resolution_)),
            static_cast<int>(std::floor(inflated_min.y() / voxel_resolution_)),
            static_cast<int>(std::floor(inflated_min.z() / voxel_resolution_))
        );
        Eigen::Vector3i inf_max_idx(
            static_cast<int>(std::floor(inflated_max.x() / voxel_resolution_)),
            static_cast<int>(std::floor(inflated_max.y() / voxel_resolution_)),
            static_cast<int>(std::floor(inflated_max.z() / voxel_resolution_))
        );

        // Raw (non-inflated) bounds
        Eigen::Vector3i raw_min_idx(
            static_cast<int>(std::floor(obstacle.min.x() / voxel_resolution_)),
            static_cast<int>(std::floor(obstacle.min.y() / voxel_resolution_)),
            static_cast<int>(std::floor(obstacle.min.z() / voxel_resolution_))
        );
        Eigen::Vector3i raw_max_idx(
            static_cast<int>(std::floor(obstacle.max.x() / voxel_resolution_)),
            static_cast<int>(std::floor(obstacle.max.y() / voxel_resolution_)),
            static_cast<int>(std::floor(obstacle.max.z() / voxel_resolution_))
        );

        // Fill inflated voxels
        for (int x = inf_min_idx.x(); x <= inf_max_idx.x(); ++x) {
            for (int y = inf_min_idx.y(); y <= inf_max_idx.y(); ++y) {
                for (int z = inf_min_idx.z(); z <= inf_max_idx.z(); ++z) {
                    occupied_voxels_inflated_.insert(Eigen::Vector3i(x, y, z));
                }
            }
        }

        // Fill raw voxels
        for (int x = raw_min_idx.x(); x <= raw_max_idx.x(); ++x) {
            for (int y = raw_min_idx.y(); y <= raw_max_idx.y(); ++y) {
                for (int z = raw_min_idx.z(); z <= raw_max_idx.z(); ++z) {
                    occupied_voxels_raw_.insert(Eigen::Vector3i(x, y, z));
                }
            }
        }
    }
}

bool SparseVoxelGrid::is_occupied_inflated(const Eigen::Vector3i& idx) const {
    return occupied_voxels_inflated_.count(idx) > 0;
}

bool SparseVoxelGrid::is_occupied_raw(const Eigen::Vector3i& idx) const {
    return occupied_voxels_raw_.count(idx) > 0;
}

} // namespace mapping
} // namespace trajectory_planner
