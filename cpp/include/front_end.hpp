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
    double sensor_range = -1.0;   // meters, -1 = disabled (full map visibility)
    double fov_vertical_min = -7.0; // degrees, bottom edge of FOV
    double fov_vertical_max = 52.0; // degrees, top edge of FOV
    double min_altitude = 0.3;    // meters above ground (positive-up, converted to NED internally)
    int max_sfc_count = -1;       // Maximum number of SFCs to generate in a single replan (Receding Horizon Truncation). -1 = disabled.


    // Kinematic limits used by the control-point/time allocator. TrajectoryPlanner
    // stamps these from its own v_max/a_max so the allocator and the MINVO
    // kinodynamic constraints are always sized against the same limits.
    double v_max = 3.0;
    double a_max = 2.0;
    // Extra cost multiplier A* pays per step through UNKNOWN (never-observed) space.
    // 0 = unknown is as cheap as free (unsafe); negative = hard-block it, restoring the old
    // behaviour where the search simply could not route past a corner.
    double unknown_traversal_cost = 2.0;
    // How far ahead A* is allowed to search, in metres. A goal beyond this is projected
    // onto the horizon and searched to instead. Bounding the SEARCH (rather than searching
    // to a distant goal and then discarding most of the result via max_sfc_count) is what
    // keeps expansion cost flat while still handing the optimiser a full-length corridor
    // chain. <= 0 disables, restoring search-to-goal.
    double planning_horizon = 10.0;
    // Deterministic cap on A* node expansions. This, not a wall-clock timer, is what bounds
    // the search -- a time-based budget makes the planner produce different trajectories on
    // identical inputs depending on CPU load. ~6000 nodes is roughly 10 ms on the reference
    // machine. <= 0 disables (search to completion).
    // 6000 was buying nothing measurable: over 204 goals, 3000 / 6000 / 12000 all solved 196
    // and reached the goal 152 times; only 1500 lost any (150). Meanwhile an exhausted search
    // is the single largest front-end cost -- a budget-exhausting expansion measured 11-15 ms
    // on a small map and ~95 ms in flight, where the accumulated map makes each expansion far
    // more cache-hostile. Halving the budget halves that worst case for free.
    int astar_max_nodes = 3000;
    // Wall-clock backstop on the A* search, ms. This was hardcoded to 100.0 -- the ENTIRE replan
    // period -- so it could never protect the 10 Hz loop it was supposed to guard: a single
    // search was permitted to consume the whole budget before it fired. Node count remains the
    // primary, deterministic limit (see astar_sfc.cpp); this only bounds the pathological case
    // where per-node cost has grown with map density. Keep it well inside the replan period.
    double astar_max_time_ms = 25.0;
    // Multiplier on the initial flight-time budget.
    //
    // The budget max(L/v_max, 2*sqrt(L/a_max)) is the time for an ideal straight-line
    // accelerate-to-cruise-decelerate profile. A real trajectory also has to curve, and the
    // terminal condition pins it to a full stop -- so the ideal figure is systematically a
    // little short, the first solve comes back infeasible, and the retry loop lengthens it.
    // Those retries are pure cost: they add solve time AND permanently inflate the duration,
    // because control-point count IS flight time. Budgeting slightly generously up front
    // trades a marginally slower ideal for far fewer retries.
    // 1.20 chosen from a 12-scenario sweep: it doubles the first-attempt success rate
    // (30% -> 60%) and cuts mean solve time ~32% (8.5 -> 5.8 ms) without losing any
    // scenario that 1.00 could solve. Higher values (1.30+) push first-try to 78% and are
    // faster still, but started failing a climb case outright -- a larger initial problem
    // interacts with the retry loop's stall detection. Retuning against your own
    // environment is worthwhile; the sweep harness is tools/ + the bench in the notes.
    double time_budget_factor = 1.20;
    // Wall-clock ceiling, in ms, on the whole OSQP retry cycle for one replan (summed over
    // attempts). A doomed solve burns a full max_iter per attempt before OSQP can certify
    // infeasibility, so an unlucky replan could run 4 attempts and ~270 ms against a 100 ms
    // period. Missing a replan is cheap -- the committed trajectory keeps flying and the next
    // cycle retries against a fresher map -- but overrunning the period starves the 50 Hz
    // dispatch loop. 0 disables the bound.
    double max_solve_time_ms = 60.0;
    // Worst constraint violation, in the constraints' own units (m, m/s, m/s^2), at which a
    // NON-CONVERGED OSQP iterate is still accepted as flyable. OSQP needs both primal and dual
    // residuals inside tolerance, so it can stop at max_iter holding a point whose constraints
    // are satisfied to ~1e-4 purely because optimality has not settled -- discarding that costs
    // a whole replan and can leave the vehicle hovering. Never applied to a PRIMAL_INFEASIBLE
    // verdict, whose iterate is a diverging ray rather than a nearly feasible point.
    // 1e-2 is negligible against the margins actually in play: corridors already carry the
    // drone-radius inflation (~0.49 m) and v_max is 3 m/s. 0 disables.
    double qp_accept_violation = 1e-2;
    // The MINVO kinodynamic constraints are applied as a PER-AXIS box: |v_x|,|v_y|,|v_z| <= v_max.
    // That does NOT bound the speed -- it permits ||v|| up to v_max*sqrt(3) (5.20 m/s at
    // v_max = 3), and likewise ||a|| up to a_max*sqrt(3). Anything claiming feasibility with
    // respect to v_max/a_max as SPEED is therefore overclaiming.
    //
    // Set true to scale the box by 1/sqrt(3), i.e. inscribe it in the sphere, which makes
    // ||v|| <= v_max and ||a|| <= a_max true guarantees. It is conservative: the box is the
    // largest one that fits inside the sphere, so axis-aligned motion loses the most. The time
    // allocator is left alone -- it already reasons in norms along a 1-D speed profile, so only
    // the QP box is inconsistent with it.
    bool enforce_norm_limits = false;
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
    bool reached_goal;
    Eigen::Vector3d actual_end_pos;
};

// ==========================================
// FrontEndSFC (Port of front_end.py)
// ==========================================
class FrontEndSFC {
public:
    explicit FrontEndSFC(const FrontEndConfig& config,
                         std::shared_ptr<mapping::SparseVoxelGrid> grid);

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
    std::shared_ptr<mapping::SparseVoxelGrid> discrete_grid_;
    std::unique_ptr<mapping::RingBufferGrid> ring_buffer_;

    // SFC generators
    sfc::AsymmetricSFCManager sfc_manager_;
    sfc::StaticSFCManager static_sfc_manager_;

    // A* planner (uses astar_sfc.hpp's existing VoxelGridData + planner)
    // astar_grid_ must outlive astar_planner_, which stores a reference to it.
    path_finding::VoxelGridData astar_grid_;
    std::unique_ptr<path_finding::AStarSFCPlanner> astar_planner_;
};

} // namespace trajectory_planner
