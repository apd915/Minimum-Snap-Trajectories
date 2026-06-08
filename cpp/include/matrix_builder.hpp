#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

namespace trajectory_planner {

/**
 * @brief Creates a sparse matrix replicating a sliding window for MINVO bounds.
 * @param D_matrix The derivative mapping matrix (e.g., D_vel or D_accel).
 * @param F_matrix The static MINVO stencil for the current degree.
 * @param current_degree The polynomial degree of the B-Spline.
 * @return A Compressed Sparse Column (CSC) matrix representing the kinematic bounds.
 */
Eigen::SparseMatrix<double> build_minvo_sparse_matrix(
    const Eigen::SparseMatrix<double>& D_matrix, 
    const Eigen::MatrixXd& F_matrix, 
    int current_degree
);

/**
 * @brief Vertically stacks a standard C++ vector of Sparse Matrices.
 * This is the high-performance C++ equivalent of Python's scipy.sparse.vstack().
 * @param matrices A std::vector containing the Sparse Matrices to stack.
 * @return A single, vertically concatenated Eigen::SparseMatrix.
 */
Eigen::SparseMatrix<double> vstack_sparse_matrices(
    const std::vector<Eigen::SparseMatrix<double>>& matrices
);

} // namespace trajectory_planner