#include "optimize.hpp"
#include <iostream>
#include <stdexcept>

namespace trajectory_planner {

Eigen::VectorXd run_qp_solver(
    const Eigen::SparseMatrix<double>& P, 
    const Eigen::VectorXd& q, 
    const Eigen::SparseMatrix<double>& A, 
    const Eigen::VectorXd& l, 
    const Eigen::VectorXd& u) 
{
    // ==========================================
    // 1. DATA CONVERSION (Eigen -> OSQP CSC format)
    // ==========================================
    // OSQP strictly requires Compressed Sparse Column (CSC) format.
    // Eigen::SparseMatrix is naturally CSC, so we just pass the internal pointers.
    
    csc P_csc = { 
        .nzmax = (c_int)P.nonZeros(),
        .m = (c_int)P.rows(),
        .n = (c_int)P.cols(),
        .p = (c_int*)P.outerIndexPtr(),
        .i = (c_int*)P.innerIndexPtr(),
        .x = (c_float*)P.valuePtr()
    };

    csc A_csc = { 
        .nzmax = (c_int)A.nonZeros(),
        .m = (c_int)A.rows(),
        .n = (c_int)A.cols(),
        .p = (c_int*)A.outerIndexPtr(),
        .i = (c_int*)A.innerIndexPtr(),
        .x = (c_float*)A.valuePtr()
    };

    // ==========================================
    // 2. WORKSPACE AND SETTINGS SETUP
    // ==========================================
    osqp_settings* settings = (osqp_settings*)malloc(sizeof(osqp_settings));
    if (!settings) {
        throw std::runtime_error("Failed to allocate OSQP settings.");
    }
    
    osqp_set_default_settings(settings);
    
    // Tuning parameters (matching your Python script)
    settings->alpha = 1.6;        // Standard relaxation parameter
    settings->eps_abs = 1e-3;     // Absolute tolerance
    settings->eps_rel = 1e-3;     // Relative tolerance
    settings->verbose = 0;        // Set to 1 to print OSQP solve times to console
    settings->warm_start = 1;     // Speeds up sequential trajectory solving

    osqp_workspace* work = nullptr;
    
    // Initialize the workspace with our problem data
    c_int setup_status = osqp_setup(&work, &P_csc, q.data(), &A_csc, l.data(), u.data(), P.rows(), P.cols(), settings);
    
    if (setup_status != 0) {
        free(settings);
        throw std::runtime_error("OSQP Setup Failed! Error Code: " + std::to_string(setup_status));
    }

    // ==========================================
    // 3. EXECUTE OPTIMIZATION
    // ==========================================
    c_int solve_status = osqp_solve(work);
    
    // Status 1 = OSQP_SOLVED, Status 2 = OSQP_SOLVED_INACCURATE
    if (solve_status != 0 && work->info->status_val != 1 && work->info->status_val != 2) {
        // If it fails (e.g., Kinematically Infeasible), cleanup and throw so the caller can handle it (like stretching time)
        osqp_cleanup(work);
        free(settings);
        throw std::runtime_error("OSQP Solve Failed! Solver Status: " + std::string(work->info->status));
    }

    // ==========================================
    // 4. EXTRACT RESULTS AND CLEANUP
    // ==========================================
    // Map the raw C array back into an Eigen::VectorXd
    Eigen::VectorXd optimized_control_points = Eigen::Map<Eigen::VectorXd>(work->solution->x, P.cols());

    // Free the C-allocated memory to prevent leaks
    osqp_cleanup(work);
    free(settings);

    return optimized_control_points;
}

} // namespace trajectory_planner