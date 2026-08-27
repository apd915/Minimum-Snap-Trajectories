#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace trajectory_planner {

/**
 * Sentinel used for "no bound" in the OSQP l/u vectors.
 *
 * OSQP treats any bound with magnitude >= OSQP_INFTY (1e30) as infinite. A real
 * IEEE infinity must NOT be used: it propagates NaN through OSQP's termination
 * tolerance computation (eps = eps_abs + eps_rel * max(||Ax||, ||z||)), so the
 * convergence test never passes and every solve runs to max_iter and returns an
 * unconverged iterate.
 */
constexpr double kQpInfinity = 1e30;

/**
 * What OSQP knew at the moment it gave up.
 *
 * On PRIMAL INFEASIBLE, OSQP returns a certificate y with A'y = 0 and y'[l,u] < 0 -- a proof
 * that no x satisfies the constraints. Its nonzero entries are exactly the rows that
 * participate in the contradiction, so attributing them back to the stacked blocks
 * (boundary conditions / corridors / velocity / acceleration) says WHICH requirement is
 * impossible rather than merely that something is.
 *
 * On MAX ITER there is no certificate, so `x` carries the last iterate and per-row violation
 * can be measured directly from it.
 */
struct QpFailureInfo {
    long long status_val = 0;
    double prim_res = 0.0;
    Eigen::VectorXd prim_inf_cert;  ///< empty unless primal infeasible
    Eigen::VectorXd x;              ///< last iterate (always populated)
};

/**
 * @brief Executes the OSQP convex optimization solver for B-Spline trajectory generation.
 * * @param P The objective matrix (Hessian) minimizing the snap derivative.
 * @param q The objective vector (usually zeros for unconstrained boundary derivatives).
 * @param A The massive, stacked inequality matrix (A_total = A_eq + A_sfc + A_vel + A_accel).
 * @param l The stacked lower bounds vector.
 * @param u The stacked upper bounds vector.
 * @return Eigen::VectorXd A flattened 1D array of the optimized Control Points.
 */
Eigen::VectorXd run_qp_solver(
    const Eigen::SparseMatrix<double>& P,
    const Eigen::VectorXd& q,
    const Eigen::SparseMatrix<double>& A,
    const Eigen::VectorXd& l,
    const Eigen::VectorXd& u,
    double* prim_res_out = nullptr,  ///< primal residual, set on success AND before throwing
    QpFailureInfo* failure_out = nullptr,  ///< populated before throwing; see QpFailureInfo
    double time_limit_s = 0.0,            ///< per-solve wall-clock cap, seconds; <=0 disables
    double accept_violation = 0.0         ///< accept a non-converged iterate whose worst
                                          ///< constraint violation is <= this; <=0 disables
);

} // namespace trajectory_planner