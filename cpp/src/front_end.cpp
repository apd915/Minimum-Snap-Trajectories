#include "front_end.hpp"
#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>

namespace trajectory_planner {

// ==========================================
// Constructor
// ==========================================
FrontEndSFC::FrontEndSFC(const FrontEndConfig& config,
                         const std::vector<mapping::ObstacleBox>& obstacles)
    : config_(config),
      discrete_grid_(config.voxel_resolution),
      static_sfc_manager_(config.sfc_height, config.sfc_width),
      sfc_manager_({}, 0.0, 0.0, config.voxel_resolution) // placeholder, rebuilt below
{
    // Calculate inflation radius based on aircraft type
    double grid_inflation_radius;
    if (config_.aircraft_type == "fixed-wing") {
        grid_inflation_radius = (config_.sfc_width / 2.0) + config_.drone_physical_radius;
    } else {
        int inflation_voxels = static_cast<int>(std::ceil(config_.drone_physical_radius / config_.voxel_resolution));
        grid_inflation_radius = inflation_voxels * config_.voxel_resolution;
    }

    // Populate the discrete grid
    discrete_grid_.populate_from_obstacles(obstacles, grid_inflation_radius);

    // Convert inflated voxel indices back to continuous meters for the spatial scanner
    inflated_obstacle_meters_.reserve(discrete_grid_.num_inflated());
    for (const auto& idx : discrete_grid_.get_occupied_inflated()) {
        Eigen::Vector3d pt = idx.cast<double>() * config_.voxel_resolution
                             + Eigen::Vector3d::Constant(config_.voxel_resolution / 2.0);
        inflated_obstacle_meters_.push_back(pt);
    }

    // Build the asymmetric SFC manager with the inflated obstacle points
    double max_cp_drift = (config_.sfc_width / 2.0) - config_.drone_physical_radius;
    sfc_manager_ = sfc::AsymmetricSFCManager(
        inflated_obstacle_meters_, 0.0, max_cp_drift, config_.voxel_resolution);

    // Build the ring buffer
    int size_x = static_cast<int>(std::ceil(config_.map_bounds.x() / config_.voxel_resolution)) + 1;
    int size_y = static_cast<int>(std::ceil(config_.map_bounds.y() / config_.voxel_resolution)) + 1;
    int size_z = static_cast<int>(std::ceil(config_.map_bounds.z() / config_.voxel_resolution)) + 1;

    ring_buffer_ = std::make_unique<mapping::RingBufferGrid>(size_x, size_y, size_z);
    for (const auto& idx : discrete_grid_.get_occupied_inflated()) {
        ring_buffer_->set_occupied(idx.x(), idx.y(), idx.z());
    }

    // Build the A* planner using the existing VoxelGridData interface
    // We must bridge from our SparseVoxelGrid to astar_sfc.hpp's VoxelGridData
    // Note: mapping::Vector3iHash and path_finding::Vector3iHash are identical structs
    // in different namespaces, so we copy element-by-element.
    path_finding::VoxelGridData astar_grid;
    astar_grid.voxel_resolution = config_.voxel_resolution;
    for (const auto& idx : discrete_grid_.get_occupied_inflated()) {
        astar_grid.occupied_voxels_inflated.insert(idx);
    }
    for (const auto& bound : discrete_grid_.get_continuous_inflated_bounds()) {
        astar_grid.continuous_inflated_bounds.push_back({bound.first, bound.second});
    }

    Eigen::Vector3i bounds_min(0, 0, 0);
    Eigen::Vector3i bounds_max(
        static_cast<int>(config_.map_bounds.x() / config_.voxel_resolution),
        static_cast<int>(config_.map_bounds.y() / config_.voxel_resolution),
        static_cast<int>(config_.map_bounds.z() / config_.voxel_resolution)
    );

    astar_planner_ = std::make_unique<path_finding::AStarSFCPlanner>(astar_grid, bounds_min, bounds_max);
}

// ==========================================
// get_corridors_astar
// ==========================================
FrontEndResult FrontEndSFC::get_corridors_astar(
    const Eigen::Vector3d& start_pos, const Eigen::Vector3d& end_pos) {

    double res = config_.voxel_resolution;

    // 1. Discretize
    Eigen::Vector3i start_idx(
        static_cast<int>(std::floor(start_pos.x() / res)),
        static_cast<int>(std::floor(start_pos.y() / res)),
        static_cast<int>(std::floor(start_pos.z() / res))
    );
    Eigen::Vector3i goal_idx(
        static_cast<int>(std::floor(end_pos.x() / res)),
        static_cast<int>(std::floor(end_pos.y() / res)),
        static_cast<int>(std::floor(end_pos.z() / res))
    );

    // 2. Search + Smooth
    auto t_start = std::chrono::high_resolution_clock::now();
    astar_planner_->search(start_idx, goal_idx);
    auto astar_path = astar_planner_->sfc_smoother();
    auto t_end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "[Front-End] A* Search & Smoothing Completed in " << elapsed_ms << " ms" << std::endl;

    // 3. Convert to continuous path
    std::vector<Eigen::Vector3d> continuous_path;
    continuous_path.reserve(astar_path.size());
    for (const auto& idx : astar_path) {
        Eigen::Vector3d pos = idx.cast<double>() * res + Eigen::Vector3d::Constant(res / 2.0);
        continuous_path.push_back(pos);
    }

    // Anchor exact start/goal
    if (continuous_path.size() >= 2) {
        continuous_path.front() = start_pos;
        continuous_path.back() = end_pos;
    }

    // 4. Build SFCs
    sfc::StandaloneWaypointsSFC waypoints_smooth(3);
    waypoints_smooth.add(continuous_path[0], -1, 0.0, false);

    std::vector<sfc::SFCResult> corridors;

    for (size_t i = 1; i < continuous_path.size(); ++i) {
        const auto& prev = continuous_path[i - 1];
        const auto& curr = continuous_path[i];
        bool is_goal = (i == continuous_path.size() - 1);

        waypoints_smooth.add(curr, static_cast<int>(i - 1), 0.0, is_goal);

        double ext_s = config_.sfc_start_ext;
        double ext_e = config_.sfc_end_ext;
        if (config_.spline_type == "natural") {
            if (i == 1) ext_s = 30.0;
            if (is_goal) ext_e = 30.0;
        }

        sfc::SFCResult sfc_result;
        if (config_.aircraft_type == "fixed-wing") {
            sfc_result = static_sfc_manager_.generate_sfc(prev, curr, ext_s, ext_e);
        } else {
            sfc_result = sfc_manager_.generate_sfc(prev, curr, config_.sfc_width, ext_s, ext_e);
        }

        waypoints_smooth.addSFC(sfc_result);
        corridors.push_back(sfc_result);
    }

    std::cout << "[Front-End] Extracted " << corridors.size() << " Safe Flight Corridors via A*." << std::endl;

    // 5. Allocate control points
    FrontEndResult result;
    result.corridors = corridors;
    result.waypoints_smooth = waypoints_smooth;

    if (config_.aircraft_type == "fixed-wing") {
        result.allocation_list = allocate_dynamic_control_points_fixed_wing(corridors);
    } else {
        result.allocation_pools = allocate_dynamic_control_points(corridors);
    }

    return result;
}

// ==========================================
// allocate_dynamic_control_points (Multi-Rotor)
// ==========================================
std::vector<ConstraintPool> FrontEndSFC::allocate_dynamic_control_points(
    const std::vector<sfc::SFCResult>& corridors) const {

    int num_corridors = static_cast<int>(corridors.size());
    int degree = config_.degree;
    double v_max = 3.0, a_max = 2.0, pts_per_sec = 1.0;

    std::vector<ConstraintPool> pools;

    for (int i = 0; i < num_corridors; ++i) {
        double L_base = corridors[i].getDistancePrimaryToSecondary();

        // Determine overlap from neighbors
        double actual_ext_prev = (i > 0)
            ? corridors[i-1].bounds(0) - corridors[i-1].getDistancePrimaryToSecondary()
            : 0.0;
        double actual_ext_next = (i < num_corridors - 1)
            ? corridors[i+1].bounds(1)
            : 0.0;

        int exclusive_pts;
        if (L_base <= (actual_ext_prev + actual_ext_next)) {
            exclusive_pts = 0; // Subsumed by bridges
        } else {
            double t_target = std::max(L_base / v_max, 2.0 * std::sqrt(L_base / a_max));
            exclusive_pts = static_cast<int>(std::ceil(t_target * pts_per_sec));
        }

        if (num_corridors == 1) {
            exclusive_pts = std::max(exclusive_pts, degree * 2);
        }

        if (exclusive_pts > 0) {
            pools.push_back({exclusive_pts, {i}, "exclusive"});
        }

        // Bridge to next SFC
        if (i < num_corridors - 1) {
            pools.push_back({degree, {i, i + 1}, "bridge"});
        }
    }

    return pools;
}

// ==========================================
// allocate_dynamic_control_points_fixed_wing
// ==========================================
std::vector<int> FrontEndSFC::allocate_dynamic_control_points_fixed_wing(
    const std::vector<sfc::SFCResult>& corridors) const {

    int num_corridors = static_cast<int>(corridors.size());
    int degree = config_.degree;
    double v_max = 3.0, a_max = 2.0, pts_per_sec = 1.5;

    std::vector<int> num_pts_list(num_corridors, 0);

    // Pass 1: Straightaway Baseline
    for (int i = 0; i < num_corridors; ++i) {
        double L = corridors[i].getDistancePrimaryToSecondary();
        double t_cruise = L / v_max;
        double t_accel = 2.0 * std::sqrt(L / a_max);
        double t_target = std::max(t_cruise, t_accel);

        int N_kinematic = static_cast<int>(std::ceil(t_target * pts_per_sec));
        num_pts_list[i] = std::max(N_kinematic, 2 * degree);
    }

    // Pass 2: Apex Injector
    for (int i = 0; i < num_corridors - 1; ++i) {
        Eigen::Vector3d p0 = corridors[i].primaryPosition;
        Eigen::Vector3d p1 = corridors[i].secondaryPosition;
        Eigen::Vector3d p2 = corridors[i+1].secondaryPosition;

        Eigen::Vector3d v_in = p1 - p0;
        Eigen::Vector3d v_out = p2 - p1;

        double norm_in = v_in.norm();
        double norm_out = v_out.norm();

        if (norm_in > 0.001 && norm_out > 0.001) {
            double cos_theta = std::clamp(v_in.dot(v_out) / (norm_in * norm_out), -1.0, 1.0);
            double momentum_shed = 1.0 - cos_theta;

            if (momentum_shed > 0.1) {
                int apex_pool = static_cast<int>(std::ceil(momentum_shed * (degree * 3)));
                int half_pool = apex_pool / 2;
                num_pts_list[i] += half_pool;
                num_pts_list[i + 1] += half_pool;
            }
        }
    }

    return num_pts_list;
}

// ==========================================
// compile_system_constraints
// ==========================================
void FrontEndSFC::compile_system_constraints(
    const std::vector<sfc::SFCResult>& corridors,
    const std::vector<ConstraintPool>& pools,
    Eigen::MatrixXd& A_sfc_out, Eigen::VectorXd& b_sfc_out, int& total_points_out) {

    int num_dimensions = 3;
    total_points_out = 0;
    for (const auto& pool : pools) {
        total_points_out += pool.pts;
    }

    std::vector<Eigen::MatrixXd> A_rows;
    std::vector<Eigen::VectorXd> b_rows;
    int global_cp_index = 0;

    for (const auto& pool : pools) {
        // Combine A/b from all SFCs in this pool
        std::vector<Eigen::MatrixXd> A_parts;
        std::vector<Eigen::VectorXd> b_parts;
        int total_ineq = 0;

        for (int sfc_idx : pool.sfc_indices) {
            A_parts.push_back(corridors[sfc_idx].A_mat);
            b_parts.push_back(corridors[sfc_idx].b_vec);
            total_ineq += static_cast<int>(corridors[sfc_idx].A_mat.rows());
        }

        Eigen::MatrixXd A_pool(total_ineq, num_dimensions);
        Eigen::VectorXd b_pool(total_ineq);
        int row = 0;
        for (size_t k = 0; k < A_parts.size(); ++k) {
            int r = static_cast<int>(A_parts[k].rows());
            A_pool.block(row, 0, r, num_dimensions) = A_parts[k];
            b_pool.segment(row, r) = b_parts[k];
            row += r;
        }

        // Lock control points inside the overlapping volume
        for (int j = 0; j < pool.pts; ++j) {
            Eigen::MatrixXd A_padded = Eigen::MatrixXd::Zero(total_ineq, total_points_out * num_dimensions);
            int col_start = global_cp_index * num_dimensions;
            A_padded.block(0, col_start, total_ineq, num_dimensions) = A_pool;

            A_rows.push_back(A_padded);
            b_rows.push_back(b_pool);
            global_cp_index++;
        }
    }

    // Stack into final matrices
    int total_rows = 0;
    for (const auto& a : A_rows) total_rows += static_cast<int>(a.rows());

    A_sfc_out.resize(total_rows, total_points_out * num_dimensions);
    b_sfc_out.resize(total_rows);

    int r = 0;
    for (size_t i = 0; i < A_rows.size(); ++i) {
        int nr = static_cast<int>(A_rows[i].rows());
        A_sfc_out.block(r, 0, nr, A_rows[i].cols()) = A_rows[i];
        b_sfc_out.segment(r, nr) = b_rows[i];
        r += nr;
    }
}

} // namespace trajectory_planner
