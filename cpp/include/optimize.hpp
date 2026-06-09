#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace trajectory_planner {

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
    const Eigen::VectorXd& u
);

} // namespace trajectory_planner