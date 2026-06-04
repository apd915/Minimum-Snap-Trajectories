#include "min_snap_clamped.hpp"
#include <stdexcept>
#include <cmath>

using namespace std;
using namespace Eigen;

// ==========================================
// PUBLIC API
// ==========================================

MinSnapEvalClamped::MinSnapEvalClamped(int16_t num_segments, int8_t degree) {
    if (degree < 4) {
        throw invalid_argument("Minimum Snap requires a polynomial of at least degree 4.");
    }
    if (num_segments < 3) {
        throw invalid_argument("To satisfy 6 physical constraints, you need at least 3 flight segments.");
    }
    
    degree_ = degree;
    update_segments(num_segments);
}

void MinSnapEvalClamped::update_segments(int16_t new_num_segments) {
    M_ = new_num_segments;
    num_control_points_ = M_ + degree_;
    double start_time = 0.0;
    
    knots_ = create_clamped_knot_points(num_control_points_, degree_, start_time);
    calculate_Q();
}

// ==========================================
// CORE OPTIMIZATION MATH
// ==========================================

void MinSnapEvalClamped::calculate_Q() {
    MatrixXd B_d3 = get_B_d3_matrix(degree_);
    auto [U1, U2] = get_U_matrices(num_control_points_);

    B_combined_ = U1; // Expose boundary matrix for OSQP equality extraction
    W_ = get_W_matrix(0.0, 0.0, 1.0); // Pure minimum snap tracking

    // Solve the minimum energy system via fast LDLT linear solve
    MatrixXd A_bar = U2.transpose() * W_ * U2;
    MatrixXd B_bar = U2.transpose() * W_;
    
    MatrixXd X_T = A_bar.ldlt().solve(B_bar);
    MatrixXd X = X_T.transpose();

    // Final analytical execution loop
    MatrixXd I = MatrixXd::Identity(num_control_points_, num_control_points_);
    Q_ = B_d3 * U1.transpose() * (I - X * U2.transpose());
}

VectorXd MinSnapEvalClamped::create_clamped_knot_points(int16_t num_ctrl_pts, int8_t degree, double start_time) {
    int16_t number_of_knot_points = num_ctrl_pts + degree + 1;
    int16_t number_of_unique_knot_points = number_of_knot_points - 2 * degree;
    
    VectorXd knots = VectorXd::Zero(number_of_knot_points);
    
    // Populate middle uniform unique intervals
    for (int16_t i = 0; i < number_of_unique_knot_points; ++i) {
        knots(degree + i) = static_cast<double>(i) + start_time;
    }
    
    // Hard clamp the trailing edge values to match the final unique timestamp
    double max_time = knots(degree + number_of_unique_knot_points - 1);
    for (int16_t i = degree + number_of_unique_knot_points; i < number_of_knot_points; ++i) {
        knots(i) = max_time;
    }
    
    // Ensure leading edge clamps match the start time
    for (int16_t i = 0; i < degree; ++i) {
        knots(i) = start_time;
    }
    
    return knots;
}

MatrixXd MinSnapEvalClamped::get_B_d3_matrix(int8_t d) {
    double d_inv = 1.0 / d;
    double accel_scalar = 2.0 / (d * (d - 1));
    
    MatrixXd B_d3 = MatrixXd::Zero(6, 6);
    B_d3 << 1,  1,       1,             0,             0,        0,
            0,  d_inv,   3 * d_inv,     0,             0,        0,
            0,  0,       accel_scalar,  0,             0,        0,
            0,  0,       0,             accel_scalar,  0,        0,
            0,  0,       0,            -3 * d_inv,    -d_inv,    0,
            0,  0,       0,             1,             1,        1; 
            
    return B_d3;
}

pair<MatrixXd, MatrixXd> MinSnapEvalClamped::get_U_matrices(int16_t num_control_points) {
    MatrixXd I = MatrixXd::Identity(num_control_points, num_control_points);
    
    MatrixXd U1(num_control_points, 6);
    U1.leftCols(3) = I.leftCols(3);
    U1.rightCols(3) = I.rightCols(3);
    
    MatrixXd U2 = I.block(0, 3, num_control_points, num_control_points - 6);
    return {U1, U2};
}

MatrixXd MinSnapEvalClamped::get_W_matrix(double rho_vel, double rho_accel, double rho_snap) {
    MatrixXd W_total = MatrixXd::Zero(num_control_points_, num_control_points_);

    // 1. Snap Penalty (d=4, j=4) -> Uses d-j=0 stencils (Identity)
    if (rho_snap > 0.0 && degree_ >= 4) {
        MatrixXd D_snap = get_fast_cascaded_D_matrix(M_, degree_, 4);
        MatrixXd W_snap;
        if (degree_ - 4 == 0) {
            W_snap = D_snap.transpose() * D_snap;
        } else {
            W_snap = D_snap.transpose() * get_basis_integral_matrix(M_, degree_, 4) * D_snap;
        }
        W_total += rho_snap * W_snap;
    }
    
    // 2. Acceleration Penalty (d=4, j=2) -> Uses d-j=2 stencils
    if (rho_accel > 0.0 && degree_ >= 2) {
        MatrixXd D_accel = get_fast_cascaded_D_matrix(M_, degree_, 2);
        MatrixXd W_accel = D_accel.transpose() * get_basis_integral_matrix(M_, degree_, 2) * D_accel;
        W_total += rho_accel * W_accel;
    }
    
    // 3. Velocity Penalty (d=4, j=1) -> Uses d-j=3 stencils
    if (rho_vel > 0.0 && degree_ >= 1) {
        MatrixXd D_vel = get_fast_cascaded_D_matrix(M_, degree_, 1);
        MatrixXd W_vel = D_vel.transpose() * get_basis_integral_matrix(M_, degree_, 1) * D_vel;
        W_total += rho_vel * W_vel;
    }

    return W_total;
}

// ==========================================
// SYMBOLIC STENCIL STEP GENERATORS
// ==========================================

MatrixXd MinSnapEvalClamped::get_basis_integral_matrix(int16_t M, int8_t degree, int8_t derivative_order) {
    int8_t d_minus_j = degree - derivative_order;
    int16_t N = M + degree - derivative_order; 
    
    MatrixXd W_int = MatrixXd::Zero(N, N);
    
    std::vector<double> interior;
    MatrixXd block;

    // 1. SELECT THE STENCILS
    if (d_minus_j == 0) {
        interior = {1.0};
        block = MatrixXd::Ones(1, 1);
    } 
    else if (d_minus_j == 1) {
        interior = {4.0/6.0, 1.0/6.0};
        block = MatrixXd(2, 2);
        block << 1.0/3.0, 1.0/6.0,
                 1.0/6.0, 4.0/6.0; 
    } 
    else if (d_minus_j == 2) {
        // Parabolas (e.g., d=4, j=2 -> Acceleration)
        interior = {11.0/20.0, 13.0/60.0, 1.0/120.0};
        block = MatrixXd(2, 2);
        block << 1.0/5.0,  7.0/60.0,
                 7.0/60.0, 1.0/3.0; 
    } 
    else if (d_minus_j == 3) {
        // Cubics (e.g., d=4, j=1 -> Velocity)
        interior = {0.479365079333, 0.2363095241, 0.0238095238095, 0.000198412694895};
        
        // Expanded to 5x5 to capture the unique bleed-over values in the 5th column/row
        block = MatrixXd(5, 5);
        block << 
            0.142857142857,   0.0875,           0.0184523809524, 0.00119047619046, 0.0,
            0.0875,           0.221428571428,   0.15625,         0.0345238095238,  0.00029761904752,
            0.0184523809524,  0.15625,          0.326785714332,  0.224603174668,   0.0237103174609,
            0.00119047619046, 0.0345238095238,  0.224603174668,  0.479365079333,   0.2363095241,
            0.0,              0.00029761904752, 0.0237103174609, 0.2363095241,     0.479365079333;
    } 
    else {
        throw std::invalid_argument("Integral stencil for (d-j) not found.");
    }

    // 2. POPULATE THE SHIFT-INVARIANT INTERIOR BAND
    for (int16_t i = 0; i < N; ++i) {
        W_int(i, i) = interior[0];
    }
    for (size_t offset = 1; offset < interior.size(); ++offset) {
        for (int16_t i = 0; i < N - offset; ++i) {
            W_int(i, i + offset) = interior[offset];     // Upper band
            W_int(i + offset, i) = interior[offset];     // Lower band
        }
    }

    // 3. OVERWRITE THE BOUNDARY CORNERS
    int16_t block_size = block.rows();
    
    // Top-Left Overwrite
    W_int.topLeftCorner(block_size, block_size) = block;
    
    // Bottom-Right Overwrite (Mirrors the top-left block perfectly for the end of the spline)
    for (int16_t r = 0; r < block_size; ++r) {
        for (int16_t c = 0; c < block_size; ++c) {
            W_int(N - 1 - r, N - 1 - c) = block(r, c);
        }
    }

    return W_int;
}

MatrixXd MinSnapEvalClamped::get_single_D_step(int8_t k, int16_t M_dummy) {
    int16_t size = M_dummy + k - 1;
    VectorXd diag = VectorXd::Zero(size);
    
    for (int8_t i = 1; i < k; ++i) {
        diag(i - 1) = static_cast<double>(k) / i;
    }
    int16_t num_ones = size - 2 * (k - 1);
    for (int16_t i = 0; i < num_ones; ++i) {
        diag(k - 1 + i) = 1.0;
    }
    for (int8_t i = k - 1; i >= 1; --i) {
        diag(size - i) = static_cast<double>(k) / i;
    }
    
    MatrixXd D = MatrixXd::Zero(size, size + 1);
    for (int16_t r = 0; r < size; ++r) {
        D(r, r) = -diag(r);
        D(r, r + 1) = diag(r);
    }
    return D;
}

pair<vector<double>, MatrixXd> MinSnapEvalClamped::generate_fallback_S_stencil(int8_t d, int8_t j) {
    int16_t M_dummy = 15;
    MatrixXd D_cascaded;
    bool initialized = false;
    
    for (int8_t k = d - j + 1; k <= d; ++k) {
        MatrixXd D_current = get_single_D_step(k, M_dummy);
        if (!initialized) {
            D_cascaded = D_current;
            initialized = true;
        } else {
            D_cascaded = D_cascaded * D_current;
        }
    }
    
    int16_t boundary_rows = d;
    int16_t boundary_cols = d + j;
    MatrixXd boundary_block = D_cascaded.block(0, 0, boundary_rows, boundary_cols);
    
    vector<double> interior(j + 1);
    for (int8_t i = 0; i <= j; ++i) {
        interior[i] = D_cascaded(boundary_rows + 1, boundary_rows + 1 + i);
    }
    return {interior, boundary_block};
}

MatrixXd MinSnapEvalClamped::get_fast_cascaded_D_matrix(int16_t M, int8_t degree, int8_t derivative_order) {
    auto [interior_band, boundary_block] = generate_fallback_S_stencil(degree, derivative_order);
    
    int16_t rows = M + degree - derivative_order;
    int16_t cols = M + degree;
    MatrixXd S_cascaded = MatrixXd::Zero(rows, cols);
    
    for (int16_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < interior_band.size(); ++j) {
            if (i + j < cols) {
                S_cascaded(i, i + j) = interior_band[j];
            }
        }
    }
    
    int16_t block_rows = boundary_block.rows();
    int16_t block_cols = boundary_block.cols();
    S_cascaded.block(0, 0, block_rows, block_cols) = boundary_block;
    
    double sign = (derivative_order % 2 == 0) ? 1.0 : -1.0;
    for (int16_t r = 0; r < block_rows; ++r) {
        for (int16_t c = 0; c < block_cols; ++c) {
            S_cascaded(rows - 1 - r, cols - 1 - c) = boundary_block(r, c) * sign;
        }
    }
    return S_cascaded;
}

// ==========================================
// API GETTERS
// ==========================================

MatrixXd MinSnapEvalClamped::get_Q_matrix() const { return Q_; }
MatrixXd MinSnapEvalClamped::get_B_combined() const { return B_combined_; }
VectorXd MinSnapEvalClamped::get_knots() const { return knots_; }