#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <string>
#include <memory>
#include "front_end.hpp"
#include "min_snap_clamped.hpp"
#include "min_snap_natural.hpp"
#include "optimize.hpp"
#include "minvo_builder.hpp"

namespace trajectory_planner {

// ==========================================
// Planning Result
// ==========================================
struct PlanningResult {
    Eigen::VectorXd control_points;     // Flat [x0,y0,z0, x1,y1,z1, ...]
    Eigen::VectorXd knots;
    std::vector<sfc::SFCResult> corridors;
    int degree;
    int total_control_points;
    bool success;
    bool reached_goal;

    // Timing metrics (milliseconds)
    double sfc_time_ms;
    double opt_time_ms;      // summed over ALL solve attempts
    double overhead_ms;
    double total_time_ms;
    double matrix_time_ms = 0.0;  // constraint-matrix assembly, summed over attempts
    int solve_attempts = 0;       // >1 means the trajectory was re-sized and re-solved
};

// ==========================================
// TrajectoryPlanner (Port of trajectory_planner.py)
// ==========================================
class TrajectoryPlanner {
public:
    TrajectoryPlanner(const FrontEndConfig& config,
                      std::shared_ptr<mapping::SparseVoxelGrid> grid,
                      double v_max = 3.0, double a_max = 2.0);

    /**
     * Main entry point: Front-End → Matrix Build → OSQP → Control Points.
     */
    PlanningResult plan_mission(const Eigen::Vector3d& start_pos,
                                const Eigen::Vector3d& end_pos,
                                const Eigen::Vector3d& start_vel = Eigen::Vector3d::Zero(),
                                const Eigen::Vector3d& start_acc = Eigen::Vector3d::Zero());

private:
    FrontEndConfig config_;
    std::unique_ptr<FrontEndSFC> front_end_;
    double v_max_;
    double a_max_;
    // Wall-clock ceiling on the OSQP retry cycle, summed over attempts. See the bail in
    // plan_mission(). 0 disables. Taken from FrontEndConfig::max_solve_time_ms.
    double max_solve_time_ms_ = 60.0;

    /**
     * Multi-rotor overlap constraint builder with map boundary injection.
     * Port of Python's _build_overlap_constraints().
     */
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> build_overlap_constraints(
        const std::vector<sfc::SFCResult>& corridors,
        const std::vector<ConstraintPool>& pools,
        int total_num_points) const;

    /**
     * Fixed-wing overlap constraint builder with map boundary injection.
     * Port of Python's _build_overlap_constraints_fixed_wing().
     */
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> build_overlap_constraints_fixed_wing(
        const std::vector<sfc::SFCResult>& corridors,
        const std::vector<int>& num_pts_list,
        int total_num_points) const;
};

} // namespace trajectory_planner
