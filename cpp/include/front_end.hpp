#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <memory>
#include "voxel_grid.hpp"
#include "ring_buffer.hpp"
#include "astar_sfc.hpp"
#include "dynamic_sfc.hpp"
#include "static_sfc.hpp"

namespace trajectory_planner {

// ==========================================
// Constraint Pool (Port of front_end.py allocation_data)
// ==========================================
struct ConstraintPool {
    int pts;                       // Number of control points in this pool
    std::vector<int> sfc_indices;  // Which SFC(s) constrain this pool
    std::string type;              // "exclusive" or "bridge"
};

// ==========================================
// FrontEnd Configuration
// ==========================================
struct FrontEndConfig {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double voxel_resolution = 0.5;
    double drone_physical_radius = 0.5;
    double sfc_height = 5.0;
    double sfc_width = 5.0;
    double sfc_start_ext = 5.0;
    double sfc_end_ext = 5.0;
    std::string spline_type = "clamped";
    std::string aircraft_type = "multi-rotor";
    int degree = 4;
    Eigen::Vector3d map_bounds = Eigen::Vector3d(100.0, 100.0, 15.0);
};

// ==========================================
// FrontEnd Result
// ==========================================
struct FrontEndResult {
    std::vector<sfc::SFCResult> corridors;
    std::vector<ConstraintPool> allocation_pools;    // multi-rotor
    std::vector<int> allocation_list;                // fixed-wing (num_pts_list)
    sfc::StandaloneWaypointsSFC waypoints_smooth;
    sfc::StandaloneWaypointsSFC waypoints_not_smooth;
};

// ==========================================
// FrontEndSFC (Port of front_end.py)
// ==========================================
class FrontEndSFC {
public:
    explicit FrontEndSFC(const FrontEndConfig& config,
                         const std::vector<mapping::ObstacleBox>& obstacles);

    /**
     * Main entry point: runs A* → smoothing → SFC generation → control point allocation.
     */
    FrontEndResult get_corridors_astar(const Eigen::Vector3d& start_pos,
                                       const Eigen::Vector3d& end_pos);

    /**
     * Compiles the unified constraint pools into A_sfc / b_sfc matrices for OSQP.
     */
    static void compile_system_constraints(
        const std::vector<sfc::SFCResult>& corridors,
        const std::vector<ConstraintPool>& pools,
        Eigen::MatrixXd& A_sfc_out, Eigen::VectorXd& b_sfc_out, int& total_points_out);

    const FrontEndConfig& config() const { return config_; }

private:
    /**
     * Multi-rotor: Sequential pairwise allocator with bridge/exclusive logic.
     */
    std::vector<ConstraintPool> allocate_dynamic_control_points(
        const std::vector<sfc::SFCResult>& corridors) const;

    /**
     * Fixed-wing: Apex-centric allocator with momentum shedding.
     */
    std::vector<int> allocate_dynamic_control_points_fixed_wing(
        const std::vector<sfc::SFCResult>& corridors) const;

    FrontEndConfig config_;

    // Map data
    mapping::SparseVoxelGrid discrete_grid_;
    std::unique_ptr<mapping::RingBufferGrid> ring_buffer_;

    // SFC generators
    sfc::AsymmetricSFCManager sfc_manager_;
    sfc::StaticSFCManager static_sfc_manager_;

    // A* planner (uses astar_sfc.hpp's existing VoxelGridData + planner)
    std::unique_ptr<path_finding::AStarSFCPlanner> astar_planner_;

    // Obstacle data in continuous meters (for the KD-tree / scanner)
    std::vector<Eigen::Vector3d> inflated_obstacle_meters_;
};

} // namespace trajectory_planner
