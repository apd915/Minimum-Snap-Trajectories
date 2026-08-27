#include "front_end.hpp"
#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdlib>

namespace trajectory_planner {

// ==========================================
// Constructor
// ==========================================
FrontEndSFC::FrontEndSFC(const FrontEndConfig& config,
                         std::shared_ptr<mapping::SparseVoxelGrid> grid)
    : config_(config),
      discrete_grid_(std::move(grid)),
      sfc_manager_(discrete_grid_, 0.0, 0.0, config.voxel_resolution),
      static_sfc_manager_(config.sfc_height, config.sfc_width)
{
    // Reconfigure the asymmetric SFC manager with the correct parameters
    double max_cp_drift = (config_.sfc_width / 2.0) - config_.drone_physical_radius;
    sfc_manager_ = sfc::AsymmetricSFCManager(
        discrete_grid_, 0.0, max_cp_drift, config_.voxel_resolution);

    // Build the ring buffer
    int size_x = static_cast<int>(std::ceil(config_.map_bounds.x() / config_.voxel_resolution)) + 1;
    int size_y = static_cast<int>(std::ceil(config_.map_bounds.y() / config_.voxel_resolution)) + 1;
    int size_z = static_cast<int>(std::ceil(config_.map_bounds.z() / config_.voxel_resolution)) + 1;

    ring_buffer_ = std::make_unique<mapping::RingBufferGrid>(size_x, size_y, size_z);
    for (const auto& idx : discrete_grid_->get_occupied_inflated()) {
        ring_buffer_->set_occupied(idx.x(), idx.y(), idx.z());
    }

    // Build the A* planner using the existing VoxelGridData interface.
    // astar_grid_ is a member (not a local) because AStarSFCPlanner stores it by
    // const reference, which must stay valid for the lifetime of astar_planner_.
    // Borrow the grid's sets rather than copying them. The planner is reconstructed every
    // replan, so copying tens of thousands of cells here cost milliseconds per cycle and got
    // worse the longer the vehicle flew. discrete_grid_ is a shared_ptr held by this object,
    // so the referenced data outlives astar_planner_.
    astar_grid_.voxel_resolution = config_.voxel_resolution;
    astar_grid_.occupied_voxels_inflated = &discrete_grid_->get_occupied_inflated();
    astar_grid_.free_voxels = &discrete_grid_->get_free();
    astar_grid_.continuous_inflated_bounds = &discrete_grid_->get_continuous_inflated_bounds();

    Eigen::Vector3i bounds_min(
        static_cast<int>(-config_.map_bounds.x() / (2.0 * config_.voxel_resolution)),
        static_cast<int>(-config_.map_bounds.y() / (2.0 * config_.voxel_resolution)),
        static_cast<int>(-config_.map_bounds.z() / (2.0 * config_.voxel_resolution))
    );
    Eigen::Vector3i bounds_max(
        static_cast<int>(config_.map_bounds.x() / (2.0 * config_.voxel_resolution)),
        static_cast<int>(config_.map_bounds.y() / (2.0 * config_.voxel_resolution)),
        static_cast<int>(config_.map_bounds.z() / (2.0 * config_.voxel_resolution))
    );

    astar_planner_ = std::make_unique<path_finding::AStarSFCPlanner>(astar_grid_, bounds_min, bounds_max);
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
    // Bound the search to a horizon. Expansion count grows steeply with goal distance --
    // unknown space is traversable now, so a far goal makes A* fan out across a huge volume,
    // and the receding-horizon truncation then throws most of that work away. Projecting the
    // goal onto the horizon keeps the search local and cheap; the next replan carries on
    // from further along.
    Eigen::Vector3d search_goal = end_pos;
    const double goal_dist = (end_pos - start_pos).norm();
    if (config_.planning_horizon > 0.0 && goal_dist > config_.planning_horizon) {
        search_goal = start_pos + (end_pos - start_pos) * (config_.planning_horizon / goal_dist);
    }

    Eigen::Vector3i goal_idx(
        static_cast<int>(std::floor(search_goal.x() / res)),
        static_cast<int>(std::floor(search_goal.y() / res)),
        static_cast<int>(std::floor(search_goal.z() / res))
    );

    // Slide the target off any obstacle it landed on.
    //
    // The horizon projection is a blind extrapolation along the line to the goal, so it lands
    // inside a wall whenever one happens to sit at that distance. An unreachable target is the
    // worst case for A*: it can never terminate early, so it expands until it hits the time
    // limit and then returns a partial path -- turning what should be the cheap case into the
    // most expensive one. Nudging to the nearest non-obstacle cell costs a handful of lookups.
    if (discrete_grid_ && discrete_grid_->is_occupied_inflated(goal_idx)) {
        const int kMaxNudgeCells = 12;
        bool found = false;
        for (int r = 1; r <= kMaxNudgeCells && !found; ++r) {
            for (int dx = -r; dx <= r && !found; ++dx) {
                for (int dy = -r; dy <= r && !found; ++dy) {
                    for (int dz = -r; dz <= r; ++dz) {
                        // Shell at radius r only; the interior was covered by earlier r.
                        if (std::abs(dx) != r && std::abs(dy) != r && std::abs(dz) != r) continue;
                        const Eigen::Vector3i cand = goal_idx + Eigen::Vector3i(dx, dy, dz);
                        if (discrete_grid_->is_occupied_inflated(cand)) continue;
                        goal_idx = cand;
                        found = true;
                        break;
                    }
                }
            }
        }
    }

    // 2. Search + Smooth
    auto t_start = std::chrono::high_resolution_clock::now();
    
    // Clear any previous virtual obstacles
    if (discrete_grid_) {
        discrete_grid_->clear_virtual_obstacles();
    }

    // Convert sensor range from meters to voxel units
    double sensor_range_voxels = (config_.sensor_range > 0.0)
        ? config_.sensor_range / config_.voxel_resolution
        : -1.0;

    // Convert min_altitude (positive-up meters) to NED voxel z-index floor.
    // In NED, z is negative-up: min_altitude 0.5m → z_ned_max = -0.5 → z_index = floor(-0.5 / res)
    int max_z_index = static_cast<int>(std::floor(-config_.min_altitude / config_.voxel_resolution));

    // max_time_ms is a loose backstop only -- config_.astar_max_nodes is the real, and
    // deterministic, limit. See the search budget comment in astar_sfc.cpp.
    auto raw_path = astar_planner_->search(start_idx, goal_idx, config_.astar_max_time_ms,
                           sensor_range_voxels,
                           max_z_index, config_.fov_vertical_min, config_.fov_vertical_max,
                           config_.unknown_traversal_cost, config_.astar_max_nodes);
    auto t_astar = std::chrono::high_resolution_clock::now();
    const std::size_t raw_cells = raw_path.size();
    auto astar_path = astar_planner_->sfc_smoother();

    auto t_end = std::chrono::high_resolution_clock::now();
    double search_ms = std::chrono::duration<double, std::milli>(t_astar - t_start).count();
    double smooth_ms = std::chrono::duration<double, std::milli>(t_end - t_astar).count();
    double elapsed_ms = search_ms + smooth_ms;
    // Split, because the two stages have completely different levers: search time is bounded by
    // astar_max_nodes / planning_horizon, while smoothing is driven by how many cells the search
    // returned (each shortcut test walks the ray, so a long meandering path costs quadratically
    // in voxel probes). Lumping them hid which one was responsible for a 95 ms front-end.
    std::cout << "[Front-End] A* search " << search_ms << " ms (" << raw_cells
              << " cells) + smoothing " << smooth_ms << " ms = " << elapsed_ms << " ms" << std::endl;

    // 3. Convert to continuous path
    std::vector<Eigen::Vector3d> continuous_path;
    continuous_path.reserve(astar_path.size());
    for (const auto& idx : astar_path) {
        Eigen::Vector3d pos = idx.cast<double>() * res + Eigen::Vector3d::Constant(res / 2.0);
        continuous_path.push_back(pos);
    }

    // Reaching the projected horizon is not reaching the mission goal -- only claim the
    // latter, or the manager will stop replanning short of the actual waypoint.
    const bool horizon_was_clamped = (search_goal - end_pos).squaredNorm() > 1e-9;
    bool reached_goal = (!astar_path.empty() && astar_path.back() == goal_idx && !horizon_was_clamped);

    // Anchor exact start/goal
    if (continuous_path.size() >= 2) {
        continuous_path.front() = start_pos;
        if (reached_goal) {
            continuous_path.back() = end_pos;
        }
    }

    // --- Receding-horizon truncation ---------------------------------------------
    // One SFC is generated per path segment, so N waypoints produce N-1 corridors.
    // Capping the corridor count caps the constraint-matrix growth (and therefore the
    // OSQP solve time) on long paths. Truncating means we no longer reach the goal
    // this cycle -- the next replan continues from further along.
    if (config_.max_sfc_count > 0 &&
        static_cast<int>(continuous_path.size()) - 1 > config_.max_sfc_count) {
        continuous_path.resize(config_.max_sfc_count + 1);
        reached_goal = false;
    }

    // 4. Build SFCs.
    // Corridors are kept out of unmeasured space by the map's FREE/UNKNOWN frontier, which
    // SparseVoxelGrid::get_obstacles_in_radius() now returns alongside real obstacles. The
    // previous scheme injected A*'s shadowed voxels here instead: those were whatever cells
    // the search happened to test and fail a line-of-sight check on, so the barrier was
    // incomplete, unstable between replans, and scattered through free space -- it collapsed
    // corridors to zero width. Nothing to inject any more.
    
    // Tell the map where corridors are about to be built so it only extracts the frontier
    // there. Radius covers the path plus the corridor extensions and lateral drift budget.
    if (discrete_grid_ && continuous_path.size() >= 2) {
        Eigen::Vector3d lo = continuous_path.front(), hi = continuous_path.front();
        for (const auto& p : continuous_path) { lo = lo.cwiseMin(p); hi = hi.cwiseMax(p); }
        const Eigen::Vector3d center = 0.5 * (lo + hi);
        const double margin = std::max(config_.sfc_start_ext, config_.sfc_end_ext)
                            + config_.sfc_width;
        discrete_grid_->set_frontier_focus(center, 0.5 * (hi - lo).norm() + margin);
    }

    sfc::StandaloneWaypointsSFC waypoints_smooth(3);
    waypoints_smooth.add(continuous_path[0], -1, 0.0, false);

    std::vector<sfc::SFCResult> corridors;

    auto build_corridor = [&](size_t i, bool is_goal) {
        double ext_s = config_.sfc_start_ext;
        double ext_e = config_.sfc_end_ext;
        if (config_.spline_type == "natural") {
            if (i == 1) ext_s = 30.0;
            if (is_goal) ext_e = 30.0;
        }
        if (config_.aircraft_type == "fixed-wing") {
            return static_sfc_manager_.generate_sfc(continuous_path[i - 1], continuous_path[i],
                                                    ext_s, ext_e);
        }
        return sfc_manager_.generate_sfc(continuous_path[i - 1], continuous_path[i],
                                         config_.sfc_width, ext_s, ext_e);
    };

    auto t_sfc0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 1; i < continuous_path.size(); ++i) {
        corridors.push_back(build_corridor(i, i == continuous_path.size() - 1));
    }
    double sfc_build_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_sfc0).count();
    std::cout << "[Front-End] SFC construction " << sfc_build_ms << " ms for "
              << corridors.size() << " corridor(s)" << std::endl;

    for (size_t i = 1; i < continuous_path.size(); ++i) {
        waypoints_smooth.add(continuous_path[i], static_cast<int>(i - 1), 0.0,
                             i == continuous_path.size() - 1);
        waypoints_smooth.addSFC(corridors[i - 1]);
    }

    std::cout << "[Front-End] Extracted " << corridors.size() << " Safe Flight Corridors via A*." << std::endl;

    // 5. Allocate control points
    FrontEndResult result;
    result.corridors = corridors;
    result.waypoints_smooth = waypoints_smooth;
    result.reached_goal = reached_goal;
    result.actual_end_pos = continuous_path.empty() ? end_pos : continuous_path.back();

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
    double v_max = std::max(config_.v_max, 1e-3);
    double a_max = std::max(config_.a_max, 1e-3);
    double pts_per_sec = 1.0;

    std::vector<ConstraintPool> pools;

    // --- Global time budget --------------------------------------------------------------
    // The acceleration-limited time 2*sqrt(L/a) is the duration of an accelerate-from-rest-
    // to-rest manoeuvre. Evaluating it PER LEG and summing bills the vehicle for stopping and
    // restarting at every intermediate waypoint, which it never does -- it flies straight
    // through. That overcharges by sqrt(N): a 15 m route split into 8 legs was allocated
    // 15.5 s instead of 5.5 s. The vehicle then crawled through a trajectory sized for three
    // times the real flight time, and since duration also sets the control-point count, the
    // QP grew until it stopped converging.
    //
    // So: size the budget ONCE from the whole path, then hand each leg its share by length.
    double total_path_length = 0.0;
    for (int i = 0; i < num_corridors; ++i) {
        total_path_length += corridors[i].getDistancePrimaryToSecondary();
    }
    const double total_time_budget =
        std::max(config_.time_budget_factor, 1e-3) *
        std::max(total_path_length / v_max, 2.0 * std::sqrt(total_path_length / a_max));

    for (int i = 0; i < num_corridors; ++i) {
        double L_base = corridors[i].getDistancePrimaryToSecondary();

        // Determine overlap from neighbors
        double actual_ext_prev = (i > 0)
            ? corridors[i-1].bounds(0) - corridors[i-1].getDistancePrimaryToSecondary()
            : 0.0;
        double actual_ext_next = (i < num_corridors - 1)
            ? corridors[i+1].bounds(1)
            : 0.0;

        // This leg's share of the global budget, proportional to how much of the path it is.
        double t_target = (total_path_length > 1e-9)
            ? total_time_budget * (L_base / total_path_length)
            : total_time_budget;

        int exclusive_pts;
        if (L_base <= (actual_ext_prev + actual_ext_next)) {
            exclusive_pts = 0; // Subsumed by bridges
        } else {
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

    if (pools.empty()) {
        return pools;
    }

    // --- Time-budget correction ---------------------------------------------------
    // The knot vector is unit-spaced, so the flight time a spline actually delivers is
    // (total control points - degree) seconds, NOT (total control points). Sizing the
    // pools directly from t_target therefore under-allocates the trajectory duration by
    // `degree` seconds, which makes the MINVO velocity/acceleration constraints
    // infeasible before the solver ever runs (worst for a single corridor, where the
    // shortfall is the full `degree`). Top the pools up until the delivered duration
    // covers what the kinematics need.
    int total_pts = 0;
    for (const auto& pool : pools) total_pts += pool.pts;

    // Total flight time must come from the TOTAL path length, not from summing the per-leg
    // times. The acceleration-limited term 2*sqrt(L/a) is the duration of an
    // accelerate-from-rest-to-rest manoeuvre, so summing it over N legs bills the vehicle for
    // stopping and restarting at every intermediate waypoint -- it does not, it flies
    // straight through. Summed, the same path costs sqrt(N) times too much: a 15 m route
    // split into 8 legs was allocated 15.5 s instead of 5.5 s. That inflated duration is why
    // the vehicle crawled, and (because duration sets the control-point count) it is also
    // what blew the QP up to hundreds of milliseconds.
    const double total_time_required = total_time_budget;

    int required_segments = static_cast<int>(std::ceil(total_time_required * pts_per_sec));
    int deficit = required_segments - (total_pts - degree);

    // Distribute the shortfall round-robin. Growing a "bridge" pool is safe: its points
    // are constrained by the intersection of both corridors, so every degree+1 window of
    // control points still lies wholly inside at least one corridor (convex hull property).
    for (int k = 0; k < deficit; ++k) {
        pools[k % pools.size()].pts += 1;
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
            // Relax constraints for the first 3 and last 3 control points 
            // since they are strictly bound by the equality constraints (Boundary conditions)
            if (global_cp_index < 3 || global_cp_index >= total_points_out - 3) {
                global_cp_index++;
                continue;
            }

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
