#include "trajectory_planner.hpp"
#include <iostream>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace trajectory_planner {

// ==========================================
// Helper: Sparse Kronecker Product  kron(A, I_n)
// Specialized for the common case where B = Identity(n).
// Result has dims (A.rows()*n, A.cols()*n).
// ==========================================
static Eigen::SparseMatrix<double> kron_with_identity(
    const Eigen::SparseMatrix<double>& A, int n)
{
    int rows_out = static_cast<int>(A.rows()) * n;
    int cols_out = static_cast<int>(A.cols()) * n;

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(A.nonZeros() * n);

    for (int k = 0; k < A.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            int row_a = static_cast<int>(it.row());
            int col_a = static_cast<int>(it.col());
            double val = it.value();
            for (int d = 0; d < n; ++d) {
                triplets.emplace_back(row_a * n + d, col_a * n + d, val);
            }
        }
    }

    Eigen::SparseMatrix<double> result(rows_out, cols_out);
    result.setFromTriplets(triplets.begin(), triplets.end());
    return result;
}

// ==========================================
// Helper: Dense → Sparse kron with identity
// ==========================================
static Eigen::SparseMatrix<double> kron_dense_with_identity(
    const Eigen::MatrixXd& A, int n)
{
    Eigen::SparseMatrix<double> A_sparse = A.sparseView();
    return kron_with_identity(A_sparse, n);
}

// ==========================================
// Constructor
// ==========================================
TrajectoryPlanner::TrajectoryPlanner(
    const FrontEndConfig& config,
    const std::vector<mapping::ObstacleBox>& obstacles,
    double v_max, double a_max)
    : config_(config), v_max_(v_max), a_max_(a_max) {
    front_end_ = std::make_unique<FrontEndSFC>(config, obstacles);
}

// ==========================================
// plan_mission
// ==========================================
PlanningResult TrajectoryPlanner::plan_mission(
    const Eigen::Vector3d& start_pos, const Eigen::Vector3d& end_pos,
    const Eigen::Vector3d& start_vel, const Eigen::Vector3d& start_acc) {

    std::cout << "--- Starting Trajectory Planning Mission ---" << std::endl;
    auto mission_start = std::chrono::high_resolution_clock::now();

    // =========================================================
    // PHASE 1: Front-End (A* + SFC Generation)
    // =========================================================
    std::cout << "[Phase 1] Generating Safe Flight Corridors..." << std::endl;
    auto sfc_start = std::chrono::high_resolution_clock::now();
    auto fe_result = front_end_->get_corridors_astar(start_pos, end_pos);
    auto sfc_end = std::chrono::high_resolution_clock::now();
    double sfc_ms = std::chrono::duration<double, std::milli>(sfc_end - sfc_start).count();

    if (fe_result.corridors.empty()) {
        std::cout << "[Error] No valid path found." << std::endl;
        return {Eigen::VectorXd(), 0, false, sfc_ms, 0, 0, sfc_ms};
    }

    // =========================================================
    // Build SE Boundary Matrix (3 x 6)
    // S = [start_pos, start_vel, start_acc]  (3 x 3)
    // =========================================================
    Eigen::MatrixXd S(3, 3);
    S.col(0) = start_pos;
    S.col(1) = start_vel;
    S.col(2) = start_acc;

    Eigen::MatrixXd SE(3, 6);
    if (config_.spline_type == "natural") {
        // E_standard = [end_pos, zeros, zeros]
        Eigen::MatrixXd E(3, 3);
        E.col(0) = end_pos;
        E.col(1) = Eigen::Vector3d::Zero();
        E.col(2) = Eigen::Vector3d::Zero();
        SE << S, E;
    } else {
        // Clamped: E_reversed = [zeros, zeros, end_pos]
        Eigen::MatrixXd E(3, 3);
        E.col(0) = Eigen::Vector3d::Zero();
        E.col(1) = Eigen::Vector3d::Zero();
        E.col(2) = end_pos;
        SE << S, E;
    }

    // =========================================================
    // Time-Stretching Retry Loop
    // =========================================================
    int max_stretches = 5;
    int stretch_count = 0;
    Eigen::VectorXd optimal_cp;
    double opt_ms = 0.0;
    bool solve_success = false;
    int total_control_points = 0;

    // Make mutable copies for stretching
    auto corridors = fe_result.corridors;
    auto allocation_pools = fe_result.allocation_pools;
    auto allocation_list = fe_result.allocation_list;

    while (stretch_count <= max_stretches) {
        std::cout << "\n--- Optimization Attempt " << (stretch_count + 1) << " ---" << std::endl;

        // Time-stretching fallback: add control points on retry
        if (stretch_count > 0) {
            if (config_.aircraft_type == "fixed-wing") {
                for (auto& pts : allocation_list) pts += 1;
            } else {
                for (auto& pool : allocation_pools) pool.pts += 1;
            }
        }

        // =========================================================
        // PHASE 2: Build Constraint Matrices
        // =========================================================
        Eigen::MatrixXd A_sfc;
        Eigen::VectorXd b_sfc;

        if (config_.aircraft_type == "fixed-wing") {
            total_control_points = 0;
            for (int pts : allocation_list) total_control_points += pts;
            std::tie(A_sfc, b_sfc) = build_overlap_constraints_fixed_wing(
                corridors, allocation_list, total_control_points);
        } else {
            // Use front_end compile_system_constraints to get raw A_sfc/b_sfc
            // then apply map boundary injection via build_overlap_constraints
            std::tie(A_sfc, b_sfc) = build_overlap_constraints(
                corridors, allocation_pools, 0 /* computed internally */);
            // Recompute total from pools
            total_control_points = 0;
            for (const auto& pool : allocation_pools) total_control_points += pool.pts;
        }

        std::cout << "[Phase 2] Translating " << corridors.size()
                  << " SFCs for " << total_control_points << " points..." << std::endl;

        // =========================================================
        // PHASE 3: Build Optimizer Matrices + Call OSQP
        // =========================================================
        auto opt_start = std::chrono::high_resolution_clock::now();

        int num_segments = total_control_points - config_.degree;

        try {
            Eigen::MatrixXd W, B_combined, D_vel, D_accel;
            Eigen::MatrixXd SE_qp;

            if (config_.spline_type == "natural") {
                MinSnapEvalNatural optimizer(num_segments, config_.degree);
                D_vel = optimizer.getFastCascadedDMatrix(num_segments, config_.degree, 1).transpose();
                D_accel = optimizer.getFastCascadedDMatrix(num_segments, config_.degree, 2).transpose();
                W = optimizer.getW();
                B_combined = optimizer.getBCombined();
                SE_qp = SE;
            } else {
                MinSnapEvalClamped optimizer(num_segments, config_.degree);
                D_vel = optimizer.get_fast_cascaded_D_matrix(num_segments, config_.degree, 1);
                D_accel = optimizer.get_fast_cascaded_D_matrix(num_segments, config_.degree, 2);
                Eigen::MatrixXd B_d3 = optimizer.get_B_d3_matrix(config_.degree);
                W = optimizer.get_W();
                B_combined = optimizer.get_B_combined();
                SE_qp = SE * B_d3;
            }

            int N = static_cast<int>(W.rows()); // == total_control_points
            Eigen::MatrixXd A_eq = B_combined.transpose(); // N x 6

            // ==========================================
            // 1. OBJECTIVE MATRICES (P and q)
            // ==========================================
            // P_3D = 2 * kron(W, I_3) — expand 1D penalty to 3D
            Eigen::SparseMatrix<double> P_osqp_full = 2.0 * kron_dense_with_identity(W, 3);
            Eigen::SparseMatrix<double> P_osqp = P_osqp_full.triangularView<Eigen::Upper>();
            Eigen::VectorXd q_osqp = Eigen::VectorXd::Zero(3 * N);

            // ==========================================
            // 2. EQUALITY CONSTRAINTS (Boundary Conditions)
            // ==========================================
            // A_eq_3D = kron(A_eq, I_3)
            Eigen::SparseMatrix<double> A_eq_3D = kron_dense_with_identity(A_eq, 3);

            // Flatten SE: SE is (3, 6), SE.T is (6, 3), flatten row-major = interleave
            // Python: SE.T.flatten() means column-major flatten of SE.T
            // SE.T shape (6, 3): row 0 = [SE(0,0), SE(1,0), SE(2,0)] = [sx, sy, sz]
            // flatten() = [SE(0,0), SE(1,0), SE(2,0), SE(0,1), SE(1,1), SE(2,1), ...]
            Eigen::MatrixXd SE_qp_T = SE_qp.transpose(); // (6, 3)
            Eigen::VectorXd l_eq(18), u_eq(18);
            for (int r = 0; r < 6; ++r) {
                for (int c = 0; c < 3; ++c) {
                    l_eq(r * 3 + c) = SE_qp_T(r, c);
                    u_eq(r * 3 + c) = SE_qp_T(r, c);
                }
            }

            // ==========================================
            // 3. GEOMETRIC CONSTRAINTS (Safe Flight Corridors)
            // ==========================================
            Eigen::SparseMatrix<double> A_sfc_sparse = A_sfc.sparseView();
            Eigen::VectorXd l_sfc = Eigen::VectorXd::Constant(b_sfc.size(), -std::numeric_limits<double>::infinity());
            Eigen::VectorXd u_sfc = b_sfc;

            // ==========================================
            // 4. KINODYNAMIC CONSTRAINTS (MINVO)
            // ==========================================
            std::cout << "[Phase 3] Building MINVO kinodynamic constraints..." << std::endl;

            Eigen::SparseMatrix<double> A_vel_1D = minvo::build_minvo_kinodynamic_matrix(
                D_vel, config_.degree - 1, config_.spline_type);
            Eigen::SparseMatrix<double> A_accel_1D = minvo::build_minvo_kinodynamic_matrix(
                D_accel, config_.degree - 2, config_.spline_type);

            Eigen::SparseMatrix<double> A_vel_3D = kron_with_identity(A_vel_1D, 3);
            Eigen::SparseMatrix<double> A_accel_3D = kron_with_identity(A_accel_1D, 3);

            Eigen::VectorXd l_vel = Eigen::VectorXd::Constant(A_vel_3D.rows(), -v_max_);
            Eigen::VectorXd u_vel = Eigen::VectorXd::Constant(A_vel_3D.rows(), v_max_);
            Eigen::VectorXd l_accel = Eigen::VectorXd::Constant(A_accel_3D.rows(), -a_max_);
            Eigen::VectorXd u_accel = Eigen::VectorXd::Constant(A_accel_3D.rows(), a_max_);

            // ==========================================
            // 5. ASSEMBLE THE MASTER MATRICES
            // ==========================================
            // A_osqp = vstack([A_eq_3D, A_sfc_sparse, A_vel_3D, A_accel_3D])
            int total_rows = static_cast<int>(A_eq_3D.rows() + A_sfc_sparse.rows() + A_vel_3D.rows() + A_accel_3D.rows());
            int total_cols = 3 * N;

            std::vector<Eigen::Triplet<double>> a_triplets;
            a_triplets.reserve(A_eq_3D.nonZeros() + A_sfc_sparse.nonZeros() + A_vel_3D.nonZeros() + A_accel_3D.nonZeros());

            int row_offset = 0;

            // Block 1: A_eq
            for (int k = 0; k < A_eq_3D.outerSize(); ++k)
                for (Eigen::SparseMatrix<double>::InnerIterator it(A_eq_3D, k); it; ++it)
                    a_triplets.emplace_back(static_cast<int>(it.row()) + row_offset, static_cast<int>(it.col()), it.value());
            row_offset += static_cast<int>(A_eq_3D.rows());

            // Block 2: A_sfc
            for (int k = 0; k < A_sfc_sparse.outerSize(); ++k)
                for (Eigen::SparseMatrix<double>::InnerIterator it(A_sfc_sparse, k); it; ++it)
                    a_triplets.emplace_back(static_cast<int>(it.row()) + row_offset, static_cast<int>(it.col()), it.value());
            row_offset += static_cast<int>(A_sfc_sparse.rows());

            // Block 3: A_vel
            for (int k = 0; k < A_vel_3D.outerSize(); ++k)
                for (Eigen::SparseMatrix<double>::InnerIterator it(A_vel_3D, k); it; ++it)
                    a_triplets.emplace_back(static_cast<int>(it.row()) + row_offset, static_cast<int>(it.col()), it.value());
            row_offset += static_cast<int>(A_vel_3D.rows());

            // Block 4: A_accel
            for (int k = 0; k < A_accel_3D.outerSize(); ++k)
                for (Eigen::SparseMatrix<double>::InnerIterator it(A_accel_3D, k); it; ++it)
                    a_triplets.emplace_back(static_cast<int>(it.row()) + row_offset, static_cast<int>(it.col()), it.value());

            Eigen::SparseMatrix<double> A_osqp(total_rows, total_cols);
            A_osqp.setFromTriplets(a_triplets.begin(), a_triplets.end());

            // l_osqp = hstack([l_eq, l_sfc, l_vel, l_accel])
            Eigen::VectorXd l_osqp(total_rows);
            l_osqp << l_eq, l_sfc, l_vel, l_accel;

            Eigen::VectorXd u_osqp(total_rows);
            u_osqp << u_eq, u_sfc, u_vel, u_accel;

            // ==========================================
            // 6. EXECUTE OSQP
            // ==========================================
            std::cout << "[Phase 3] Running OSQP Solver (" << total_rows << " constraints, "
                      << total_cols << " variables)..." << std::endl;

            optimal_cp = run_qp_solver(P_osqp, q_osqp, A_osqp, l_osqp, u_osqp);

            auto opt_end = std::chrono::high_resolution_clock::now();
            opt_ms = std::chrono::duration<double, std::milli>(opt_end - opt_start).count();
            solve_success = true;

            std::cout << "[Phase 3] Optimization Successful in " << opt_ms << " ms!" << std::endl;
            break; // Success — exit the retry loop

        } catch (const std::exception& e) {
            auto opt_end = std::chrono::high_resolution_clock::now();
            opt_ms = std::chrono::duration<double, std::milli>(opt_end - opt_start).count();

            std::cout << "[Phase 4] Solver failed (Kinematically impossible): " << e.what() << std::endl;

            if (stretch_count < max_stretches) {
                std::cout << "[Phase 4] Stretching time allocation (+1 point per pool)..." << std::endl;
            } else {
                std::cout << "[Error] Max stretching attempts reached." << std::endl;
            }
            stretch_count++;
        }
    }

    auto mission_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(mission_end - mission_start).count();
    double overhead_ms = total_ms - sfc_ms - opt_ms;

    std::cout << "\n=================================================="
              << "\n          TRAJECTORY PLANNER BENCHMARKS"
              << "\n=================================================="
              << "\nSFC Generation (Front-End):   " << sfc_ms << " ms"
              << "\nPath Generation (Back-End):   " << opt_ms << " ms"
              << "\nMatrix & System Overhead:     " << overhead_ms << " ms"
              << "\n--------------------------------------------------"
              << "\nTOTAL PLANNING TIME:          " << total_ms << " ms"
              << "\n==================================================" << std::endl;

    return {optimal_cp, total_control_points, solve_success, sfc_ms, opt_ms, overhead_ms, total_ms};
}

// ==========================================
// build_overlap_constraints (Multi-Rotor)
// Port of Python's _build_overlap_constraints()
// ==========================================
std::pair<Eigen::MatrixXd, Eigen::VectorXd> TrajectoryPlanner::build_overlap_constraints(
    const std::vector<sfc::SFCResult>& corridors,
    const std::vector<ConstraintPool>& pools,
    int /* total_num_points */) const
{
    int num_dimensions = 3;

    // Compute total control points from pools
    int total_pts = 0;
    for (const auto& pool : pools) total_pts += pool.pts;

    // Map boundary hyperplanes: [+x, -x, +y, -y, +z, -z]
    double north_end = config_.map_bounds.x();
    double east_end = config_.map_bounds.y();
    double alt_end = config_.map_bounds.z();

    Eigen::MatrixXd A_map(6, 3);
    A_map << 1.0,  0.0,  0.0,
            -1.0,  0.0,  0.0,
             0.0,  1.0,  0.0,
             0.0, -1.0,  0.0,
             0.0,  0.0,  1.0,
             0.0,  0.0, -1.0;
    Eigen::VectorXd b_map(6);
    b_map << north_end, 0.0, east_end, 0.0, alt_end, 0.0;

    std::vector<Eigen::MatrixXd> A_rows;
    std::vector<Eigen::VectorXd> b_rows;
    int global_cp_index = 0;

    for (const auto& pool : pools) {
        // Combine A/b from all SFCs in this pool + map boundaries
        std::vector<Eigen::MatrixXd> A_parts;
        std::vector<Eigen::VectorXd> b_parts;

        for (int sfc_idx : pool.sfc_indices) {
            A_parts.push_back(corridors[sfc_idx].A_mat);
            b_parts.push_back(corridors[sfc_idx].b_vec);
        }

        // Append map boundaries
        A_parts.push_back(A_map);
        b_parts.push_back(b_map);

        // Combine into single A_pool/b_pool
        int total_ineq = 0;
        for (const auto& a : A_parts) total_ineq += static_cast<int>(a.rows());

        Eigen::MatrixXd A_pool(total_ineq, num_dimensions);
        Eigen::VectorXd b_pool(total_ineq);
        int row = 0;
        for (size_t k = 0; k < A_parts.size(); ++k) {
            int r = static_cast<int>(A_parts[k].rows());
            A_pool.block(row, 0, r, num_dimensions) = A_parts[k];
            b_pool.segment(row, r) = b_parts[k];
            row += r;
        }

        // Virtual runway logic for natural splines
        if (config_.spline_type == "natural") {
            // Check if this pool touches the first or last SFC
            bool is_first = false, is_last = false;
            for (int idx : pool.sfc_indices) {
                if (idx == 0) is_first = true;
                if (idx == static_cast<int>(corridors.size()) - 1) is_last = true;
            }

            if (is_first || is_last) {
                double runway_length = 30.0;
                for (int k = 0; k < static_cast<int>(b_pool.size()); ++k) {
                    Eigen::Vector3d normal = A_pool.row(k).transpose();
                    double val = b_pool(k);

                    // Relax map boundaries
                    if (normal.isApprox(Eigen::Vector3d(1, 0, 0), 1e-2) && std::abs(val - north_end) < 1e-2)
                        b_pool(k) += runway_length;
                    else if (normal.isApprox(Eigen::Vector3d(-1, 0, 0), 1e-2) && std::abs(val) < 1e-2)
                        b_pool(k) += runway_length;
                    else if (normal.isApprox(Eigen::Vector3d(0, 1, 0), 1e-2) && std::abs(val - east_end) < 1e-2)
                        b_pool(k) += runway_length;
                    else if (normal.isApprox(Eigen::Vector3d(0, -1, 0), 1e-2) && std::abs(val) < 1e-2)
                        b_pool(k) += runway_length;
                    else if (normal.isApprox(Eigen::Vector3d(0, 0, 1), 1e-2) && std::abs(val - alt_end) < 1e-2)
                        b_pool(k) += runway_length;
                    else if (normal.isApprox(Eigen::Vector3d(0, 0, -1), 1e-2) && std::abs(val) < 1e-2)
                        b_pool(k) += runway_length;
                }
            }
        }

        // Lock control points inside the overlapping volume
        for (int j = 0; j < pool.pts; ++j) {
            Eigen::MatrixXd A_padded = Eigen::MatrixXd::Zero(total_ineq, total_pts * num_dimensions);
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

    Eigen::MatrixXd A_sfc_out(total_rows, total_pts * num_dimensions);
    Eigen::VectorXd b_sfc_out(total_rows);

    int r = 0;
    for (size_t i = 0; i < A_rows.size(); ++i) {
        int nr = static_cast<int>(A_rows[i].rows());
        A_sfc_out.block(r, 0, nr, A_rows[i].cols()) = A_rows[i];
        b_sfc_out.segment(r, nr) = b_rows[i];
        r += nr;
    }

    return {A_sfc_out, b_sfc_out};
}

// ==========================================
// build_overlap_constraints_fixed_wing
// Port of Python's _build_overlap_constraints_fixed_wing()
// ==========================================
std::pair<Eigen::MatrixXd, Eigen::VectorXd> TrajectoryPlanner::build_overlap_constraints_fixed_wing(
    const std::vector<sfc::SFCResult>& corridors,
    const std::vector<int>& num_pts_list,
    int total_num_points) const
{
    int num_dimensions = 3;
    int degree = config_.degree;

    double north_end = config_.map_bounds.x();
    double east_end = config_.map_bounds.y();
    double alt_end = config_.map_bounds.z();

    // Map boundary hyperplanes
    Eigen::MatrixXd A_map(6, 3);
    A_map << 1.0,  0.0,  0.0,
            -1.0,  0.0,  0.0,
             0.0,  1.0,  0.0,
             0.0, -1.0,  0.0,
             0.0,  0.0,  1.0,
             0.0,  0.0, -1.0;
    Eigen::VectorXd b_map(6);
    b_map << north_end, 0.0, east_end, 0.0, alt_end, 0.0;

    std::vector<Eigen::MatrixXd> A_ineq_list;
    std::vector<Eigen::VectorXd> b_ineq_list;
    int start_idx = 0;

    for (size_t i = 0; i < corridors.size(); ++i) {
        // Intersect SFC with map bounding box
        int sfc_rows = static_cast<int>(corridors[i].A_mat.rows());
        int total_ineq = sfc_rows + 6;

        Eigen::MatrixXd A_mat(total_ineq, 3);
        A_mat.topRows(sfc_rows) = corridors[i].A_mat;
        A_mat.bottomRows(6) = A_map;

        Eigen::VectorXd b_vec(total_ineq);
        b_vec.head(sfc_rows) = corridors[i].b_vec;
        b_vec.tail(6) = b_map;

        // Virtual runway logic for natural splines
        if (config_.spline_type == "natural") {
            if (i == 0 || i == corridors.size() - 1) {
                double runway_length = 30.0;
                for (int k = 0; k < total_ineq; ++k) {
                    Eigen::Vector3d normal = A_mat.row(k).transpose();
                    double val = b_vec(k);

                    if (normal.isApprox(Eigen::Vector3d(1, 0, 0), 1e-2) && std::abs(val - north_end) < 1e-2)
                        b_vec(k) += runway_length;
                    else if (normal.isApprox(Eigen::Vector3d(-1, 0, 0), 1e-2) && std::abs(val) < 1e-2)
                        b_vec(k) += runway_length;
                    else if (normal.isApprox(Eigen::Vector3d(0, 1, 0), 1e-2) && std::abs(val - east_end) < 1e-2)
                        b_vec(k) += runway_length;
                    else if (normal.isApprox(Eigen::Vector3d(0, -1, 0), 1e-2) && std::abs(val) < 1e-2)
                        b_vec(k) += runway_length;
                    else if (normal.isApprox(Eigen::Vector3d(0, 0, 1), 1e-2) && std::abs(val - alt_end) < 1e-2)
                        b_vec(k) += runway_length;
                    else if (normal.isApprox(Eigen::Vector3d(0, 0, -1), 1e-2) && std::abs(val) < 1e-2)
                        b_vec(k) += runway_length;
                }
            }
        }

        int num_pts_in_box = num_pts_list[i];

        for (int j = 0; j < num_pts_in_box; ++j) {
            int global_cp_index = start_idx + j;
            Eigen::MatrixXd A_padded = Eigen::MatrixXd::Zero(total_ineq, total_num_points * num_dimensions);
            int col_start = global_cp_index * num_dimensions;
            A_padded.block(0, col_start, total_ineq, num_dimensions) = A_mat;

            A_ineq_list.push_back(A_padded);
            b_ineq_list.push_back(b_vec);
        }

        start_idx += (num_pts_in_box - degree);
    }

    // Stack
    int total_rows = 0;
    for (const auto& a : A_ineq_list) total_rows += static_cast<int>(a.rows());

    Eigen::MatrixXd A_sfc_total(total_rows, total_num_points * num_dimensions);
    Eigen::VectorXd b_sfc_total(total_rows);

    int r = 0;
    for (size_t i = 0; i < A_ineq_list.size(); ++i) {
        int nr = static_cast<int>(A_ineq_list[i].rows());
        A_sfc_total.block(r, 0, nr, A_ineq_list[i].cols()) = A_ineq_list[i];
        b_sfc_total.segment(r, nr) = b_ineq_list[i];
        r += nr;
    }

    return {A_sfc_total, b_sfc_total};
}

} // namespace trajectory_planner
