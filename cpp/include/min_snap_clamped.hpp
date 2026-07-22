#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <vector>
#include <utility> 

class MinSnapEvalClamped {
private:
    int8_t degree_; 
    int16_t M_; 
    int16_t num_control_points_;

    Eigen::VectorXd knots_;
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd W_;
    Eigen::MatrixXd B_combined_;

    // Core Math Core
    void calculate_Q();
    Eigen::VectorXd create_clamped_knot_points(int16_t num_ctrl_pts, int8_t degree, double start_time);
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> get_U_matrices(int16_t num_control_points);
    Eigen::MatrixXd get_basis_integral_matrix(int16_t M, int8_t degree, int8_t derivative_order);
    
    // Self-Contained Symbolic Fallback Generators
    Eigen::MatrixXd get_single_D_step(int8_t k, int16_t M_dummy = 15);
    std::pair<std::vector<double>, Eigen::MatrixXd> generate_fallback_S_stencil(int8_t d, int8_t j);

public:
    MinSnapEvalClamped(int16_t num_segments, int8_t degree = 4);
    void update_segments(int16_t new_num_segments);

    Eigen::MatrixXd get_Q_matrix() const;
    Eigen::MatrixXd get_B_combined() const;
    Eigen::VectorXd get_knots() const;

    // Returns the stored W_ matrix (pure minimum-snap penalty computed at construction)
    Eigen::MatrixXd get_W() const { return W_; }

    // Penalty matrix generator (exposed for trajectory planner)
    Eigen::MatrixXd get_W_matrix(double rho_vel = 0.0, double rho_accel = 0.0, double rho_snap = 1.0);

    // Derivative matrix builder (needed by trajectory_planner for MINVO kinodynamic constraints)
    Eigen::MatrixXd get_fast_cascaded_D_matrix(int16_t M, int8_t degree, int8_t derivative_order);

    // Clamped boundary transformation matrix
    Eigen::MatrixXd get_B_d3_matrix(int8_t d);
};