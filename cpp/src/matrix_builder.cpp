#include "matrix_builder.hpp"

namespace trajectory_planner {

Eigen::SparseMatrix<double> build_minvo_sparse_matrix(
    const Eigen::SparseMatrix<double>& D_matrix, 
    const Eigen::MatrixXd& F_matrix, 
    int current_degree) 
{
    int num_deriv_points = D_matrix.rows();
    int num_windows = num_deriv_points - current_degree;
    int num_minvo_points_per_window = F_matrix.rows();
    
    // Initialize the Triplet list for rapid memory allocation
    std::vector<Eigen::Triplet<double>> tripletList;
    
    // Pre-allocate exact memory size to prevent OS bottlenecks
    tripletList.reserve(num_windows * num_minvo_points_per_window * (current_degree + 1));
    
    int row_idx = 0;
    
    // Replicate the "sliding window" by shifting the column index
    for (int s = 0; s < num_windows; ++s) {
        for (int i = 0; i < num_minvo_points_per_window; ++i) {
            for (int j = 0; j <= current_degree; ++j) {
                
                double val = F_matrix(i, j);
                if (val != 0.0) {
                    // (row, col, value)
                    tripletList.push_back(Eigen::Triplet<double>(row_idx, s + j, val));
                }
            }
            row_idx++;
        }
    }
    
    // Assemble the Sliding Window matrix
    Eigen::SparseMatrix<double> M_window(row_idx, num_deriv_points);
    M_window.setFromTriplets(tripletList.begin(), tripletList.end());
    
    // Multiply by the Derivative mapping matrix and return
    return M_window * D_matrix;
}


Eigen::SparseMatrix<double> vstack_sparse_matrices(
    const std::vector<Eigen::SparseMatrix<double>>& matrices) 
{
    if (matrices.empty()) {
        return Eigen::SparseMatrix<double>(0, 0);
    }

    int total_rows = 0;
    int total_cols = matrices[0].cols();
    int total_non_zeros = 0;

    // First pass: Calculate exact memory requirements to prevent reallocation
    for (const auto& mat : matrices) {
        total_rows += mat.rows();
        total_non_zeros += mat.nonZeros();
    }

    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(total_non_zeros);

    int current_row_offset = 0;

    // Second pass: Extract data using Eigen's ultra-fast internal iterators
    for (const auto& mat : matrices) {
        for (int k = 0; k < mat.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it) {
                // Shift the row index downward by the current offset
                tripletList.push_back(Eigen::Triplet<double>(
                    it.row() + current_row_offset, 
                    it.col(), 
                    it.value()
                ));
            }
        }
        current_row_offset += mat.rows();
    }

    // Assemble and compress the final stacked matrix
    Eigen::SparseMatrix<double> stacked_matrix(total_rows, total_cols);
    stacked_matrix.setFromTriplets(tripletList.begin(), tripletList.end());
    
    return stacked_matrix;
}

} // namespace trajectory_planner