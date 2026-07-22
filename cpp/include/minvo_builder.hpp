#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <string>
#include <cmath>
#include "minvo_bounds_clamped.hpp"
#include "minvo_bounds_natural.hpp"

namespace trajectory_planner {
namespace minvo {

/**
 * @brief Port of Python's build_minvo_kinodynamic_matrix() from optimize.py.
 *
 * Compiles the massive sparse matrix that applies MINVO kinematics simultaneously 
 * across the trajectory. Routes logic based on whether the spline is "natural" 
 * (sliding window) or "clamped" (boundary distorted).
 *
 * @param D_matrix     The dense derivative matrix (from get_fast_cascaded_D_matrix)
 * @param current_degree  The degree of the derivative-space basis (degree - derivative_order)
 * @param spline_type  "clamped" or "natural"
 * @return Sparse matrix A_minvo = M_mapping * D_matrix
 */
inline Eigen::SparseMatrix<double> build_minvo_kinodynamic_matrix(
    const Eigen::MatrixXd& D_matrix,
    int current_degree,
    const std::string& spline_type = "clamped")
{
    int num_deriv_points = static_cast<int>(D_matrix.rows());
    int total_spans = num_deriv_points - current_degree;

    // COO triplets for the mapping matrix
    std::vector<Eigen::Triplet<double>> triplets;
    int row_idx = 0;

    if (spline_type == "clamped") {
        // ---------------------------------------------------------
        // PATH A: The Clamped Stencil Logic (MADER)
        // ---------------------------------------------------------
        auto stencils = get_clamped_stencils();
        const auto& degree_stencils = stencils.at(current_degree);

        for (int span_idx = 0; span_idx < total_spans; ++span_idx) {
            // 1. Grab the correct stencil based on physical position
            Eigen::MatrixXd F_matrix;
            if (span_idx < current_degree) {
                F_matrix = degree_stencils.at("start_" + std::to_string(span_idx));
            } else if (span_idx >= total_spans - current_degree) {
                int end_idx = total_spans - 1 - span_idx;
                F_matrix = degree_stencils.at("end_" + std::to_string(end_idx));
            } else {
                F_matrix = degree_stencils.at("interior");
            }

            // 2. Apply it to the sparse mapping
            int num_minvo_pts = static_cast<int>(F_matrix.rows());
            for (int i = 0; i < num_minvo_pts; ++i) {
                for (int j = 0; j <= current_degree; ++j) {
                    double val = F_matrix(i, j);
                    if (std::abs(val) > 1e-9) {
                        triplets.emplace_back(row_idx, span_idx + j, val);
                    }
                }
                row_idx++;
            }
        }
    } else {
        // ---------------------------------------------------------
        // PATH B: The Natural Stencil Logic (Standard Sliding Window)
        // ---------------------------------------------------------
        auto stencils = get_natural_stencils();
        const Eigen::MatrixXd& F_matrix = stencils.at(current_degree);
        int num_minvo_pts = static_cast<int>(F_matrix.rows());

        for (int span_idx = 0; span_idx < total_spans; ++span_idx) {
            for (int i = 0; i < num_minvo_pts; ++i) {
                for (int j = 0; j <= current_degree; ++j) {
                    double val = F_matrix(i, j);
                    if (std::abs(val) > 1e-9) {
                        triplets.emplace_back(row_idx, span_idx + j, val);
                    }
                }
                row_idx++;
            }
        }
    }

    // Assemble the Mapping matrix (COO -> CSC)
    Eigen::SparseMatrix<double> M_mapping(row_idx, num_deriv_points);
    M_mapping.setFromTriplets(triplets.begin(), triplets.end());

    // Multiply it by the Calculus Derivative matrix to get the final Kinodynamic Bounds
    Eigen::SparseMatrix<double> D_sparse = D_matrix.sparseView();
    Eigen::SparseMatrix<double> A_minvo = (M_mapping * D_sparse).pruned();

    return A_minvo;
}

} // namespace minvo
} // namespace trajectory_planner
