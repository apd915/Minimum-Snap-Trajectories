#include "optimize.hpp"
#include <osqp/osqp.h>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace trajectory_planner {

Eigen::VectorXd run_qp_solver(
    const Eigen::SparseMatrix<double>& P, 
    const Eigen::VectorXd& q, 
    const Eigen::SparseMatrix<double>& A, 
    const Eigen::VectorXd& l, 
    const Eigen::VectorXd& u) 
{
    OSQPInt n = P.cols();
    OSQPInt m = A.rows();

    // ==========================================
    // 1. SAFE DATA CONVERSION (Eigen -> OSQP v1.0)
    // ==========================================
    // OSQP v1.0 uses OSQPInt (usually 64-bit). Eigen uses 32-bit ints by default. 
    // We copy the sparse index arrays to guarantee memory safety.
    
    std::vector<OSQPInt> P_i(P.innerIndexPtr(), P.innerIndexPtr() + P.nonZeros());
    std::vector<OSQPInt> P_p(P.outerIndexPtr(), P.outerIndexPtr() + P.cols() + 1);

    std::vector<OSQPInt> A_i(A.innerIndexPtr(), A.innerIndexPtr() + A.nonZeros());
    std::vector<OSQPInt> A_p(A.outerIndexPtr(), A.outerIndexPtr() + A.cols() + 1);

    OSQPCscMatrix* P_csc = OSQPCscMatrix_new(
        n, n, 
        (OSQPInt)P.nonZeros(), 
        (OSQPFloat*)P.valuePtr(), 
        P_i.data(), 
        P_p.data()
    );

    OSQPCscMatrix* A_csc = OSQPCscMatrix_new(
        m, n, 
        (OSQPInt)A.nonZeros(), 
        (OSQPFloat*)A.valuePtr(), 
        A_i.data(), 
        A_p.data()
    );

    // ==========================================
    // 2. SETTINGS SETUP
    // ==========================================
    OSQPSettings* settings = OSQPSettings_new();
    if (!settings) {
        OSQPCscMatrix_free(P_csc);
        OSQPCscMatrix_free(A_csc);
        throw std::runtime_error("Failed to allocate OSQP settings.");
    }
    
    settings->alpha = 1.6;        
    settings->eps_abs = 1e-3;     
    settings->eps_rel = 1e-3;   
    settings->polishing = 1;  
    settings->verbose = 0;        
    settings->warm_starting = 1;  // Note: renamed from 'warm_start' in v1.0

    // ==========================================
    // 3. WORKSPACE SETUP
    // ==========================================
    OSQPSolver* solver = nullptr; // Note: renamed from 'osqp_workspace'
    
    // v1.0 setup signature requires passing m and n directly
    OSQPInt setup_status = osqp_setup(&solver, P_csc, (OSQPFloat*)q.data(), A_csc, (OSQPFloat*)l.data(), (OSQPFloat*)u.data(), m, n, settings);
    
    if (setup_status != 0) {
        OSQPSettings_free(settings);
        OSQPCscMatrix_free(P_csc);
        OSQPCscMatrix_free(A_csc);
        throw std::runtime_error("OSQP Setup Failed! Error Code: " + std::to_string(setup_status));
    }

    // ==========================================
    // 4. EXECUTE OPTIMIZATION
    // ==========================================
    OSQPInt solve_status = osqp_solve(solver);
    
    // Status 1 = OSQP_SOLVED, Status 2 = OSQP_SOLVED_INACCURATE
    if (solve_status != 0 && solver->info->status_val != 1 && solver->info->status_val != 2) {
        OSQPInt status_code = solver->info->status_val;
        osqp_cleanup(solver);
        OSQPSettings_free(settings);
        OSQPCscMatrix_free(P_csc);
        OSQPCscMatrix_free(A_csc);
        throw std::runtime_error("OSQP Solve Failed! Status Code: " + std::to_string(status_code));
    }

    // ==========================================
    // 5. EXTRACT RESULTS AND CLEANUP
    // ==========================================
    Eigen::VectorXd optimized_control_points = Eigen::Map<Eigen::VectorXd>(solver->solution->x, n);

    // Free the C-allocated structs to prevent memory leaks
    osqp_cleanup(solver);
    OSQPSettings_free(settings);
    OSQPCscMatrix_free(P_csc);
    OSQPCscMatrix_free(A_csc);

    return optimized_control_points;
}

} // namespace trajectory_planner