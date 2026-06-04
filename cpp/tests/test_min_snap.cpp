#include <gtest/gtest.h>
#include "min_snap_natural.hpp" 
#include <Eigen/Dense>
#include <chrono>   // High-precision timing
#include <iostream> // Printing to the terminal

using namespace std;
using namespace Eigen;
using namespace std::chrono;

TEST(MinSnapNaturalTest, UnconstrainedTrajectory) {
    int8_t degree = 4;
    int16_t base_segments = 15;

    cout << "\n[ C++ ENGINE ] Booting up and pre-computing Q Matrix..." << endl;
    
    // 1. Start the High-Precision Timer
    auto start_time = high_resolution_clock::now();

    // 2. Initialize Evaluator and trigger the heavy math
    MinSnapEvalNatural evaluator(base_segments, degree);
    MatrixXd Q = evaluator.getQMatrix();

    // 3. Define Start and End States (Exactly matching Python)
    Vector3d p0(0, 0, 0);
    Vector3d v0(-10, -10, 10);
    Vector3d a0(0, 0, 0);

    Vector3d pf(10, 10, 10);
    Vector3d vf(-10, -10, -10);
    Vector3d af(0, 0, 0);

    // Build the SE constraint matrix (3 rows, 6 columns)
    // We use .col() to safely replicate Python's np.hstack
    MatrixXd SE(3, 6);
    SE.col(0) = p0;
    SE.col(1) = v0;
    SE.col(2) = a0;
    SE.col(3) = pf;
    SE.col(4) = vf;
    SE.col(5) = af;

    // 4. Calculate the optimal unconstrained 3D flight path
    // Python equivalent: C_p_min_snap = SE @ Q
    MatrixXd C_p_min_snap = SE * Q;

    // 5. Stop the Timer
    auto end_time = high_resolution_clock::now();
    duration<double> total_time = end_time - start_time;

    // ==========================================
    // PRINT THE RESULTS
    // ==========================================
    cout << "\n============================================" << endl;
    cout << "  C++ Execution Time: " << total_time.count() * 1000.0 << " ms" << endl;
    cout << "============================================\n" << endl;

    cout << "--- 3D Minimum Snap Control Points (C_p_min_snap) ---\n" << endl;
    
    // We transpose it when printing so it reads vertically (N rows, 3 columns [X, Y, Z]) 
    // exactly how numpy prints it by default in the terminal.
    cout << C_p_min_snap.transpose() << "\n" << endl;

    // Google Test basic assertions to ensure matrix sizes are correct
    EXPECT_EQ(C_p_min_snap.rows(), 3);
    EXPECT_EQ(C_p_min_snap.cols(), base_segments + degree);
}