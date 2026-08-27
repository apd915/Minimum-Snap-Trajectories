#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include "static_sfc.hpp"
#include "dynamic_sfc.hpp"

namespace trajectory_planner {
namespace sfc {
namespace tests {

// ==========================================
// TEST 1: StaticSFCManager — Axis-Aligned Case
// ==========================================
TEST(StaticSFCTest, AxisAlignedSFC) {
    StaticSFCManager manager(2.0, 4.0); // H=2, W=4

    Eigen::Vector3d pA(0.0, 0.0, 5.0);
    Eigen::Vector3d pB(10.0, 0.0, 5.0);

    SFCResult sfc = manager.generate_sfc(pA, pB, 1.0, 2.0);

    // Verify bounds
    // bounds = [L + ext_end, ext_start, W/2, W/2, H/2, H/2]
    EXPECT_NEAR(sfc.bounds(0), 10.0 + 2.0, 1e-6); // +x: dist + ext_end
    EXPECT_NEAR(sfc.bounds(1), 1.0, 1e-6);         // -x: ext_start
    EXPECT_NEAR(sfc.bounds(2), 2.0, 1e-6);         // +y: W/2
    EXPECT_NEAR(sfc.bounds(3), 2.0, 1e-6);         // -y: W/2
    EXPECT_NEAR(sfc.bounds(4), 1.0, 1e-6);         // +z: H/2
    EXPECT_NEAR(sfc.bounds(5), 1.0, 1e-6);         // -z: H/2

    // A_mat should be 6x3
    EXPECT_EQ(sfc.A_mat.rows(), 6);
    EXPECT_EQ(sfc.A_mat.cols(), 3);

    // b_vec should be 6x1
    EXPECT_EQ(sfc.b_vec.size(), 6);

    // Primary/secondary positions preserved
    EXPECT_TRUE(sfc.primaryPosition.isApprox(pA));
    EXPECT_TRUE(sfc.secondaryPosition.isApprox(pB));

    std::cout << "\n--- Static SFC Test (Axis-Aligned) ---" << std::endl;
    std::cout << "A_mat:\n" << sfc.A_mat << std::endl;
    std::cout << "b_vec: " << sfc.b_vec.transpose() << std::endl;
    std::cout << "Bounds: " << sfc.bounds.transpose() << std::endl;
}

// ==========================================
// TEST 2: StaticSFCManager — Diagonal Case
// ==========================================
TEST(StaticSFCTest, DiagonalSFC) {
    StaticSFCManager manager(2.0, 4.0);

    Eigen::Vector3d pA(0.0, 0.0, 0.0);
    Eigen::Vector3d pB(10.0, 10.0, 0.0); // 45-degree diagonal

    SFCResult sfc = manager.generate_sfc(pA, pB, 1.0, 1.0);

    double expected_dist = std::sqrt(200.0); // ~14.14m
    EXPECT_NEAR(sfc.getDistancePrimaryToSecondary(), expected_dist, 1e-6);
    EXPECT_NEAR(sfc.bounds(0), expected_dist + 1.0, 1e-6);

    // Verify the local frame is orthonormal
    EXPECT_NEAR(sfc.ux.dot(sfc.uy), 0.0, 1e-6);
    EXPECT_NEAR(sfc.ux.dot(sfc.uz), 0.0, 1e-6);
    EXPECT_NEAR(sfc.uy.dot(sfc.uz), 0.0, 1e-6);

    std::cout << "\n--- Static SFC Test (Diagonal) ---" << std::endl;
    std::cout << "ux: " << sfc.ux.transpose() << std::endl;
    std::cout << "uy: " << sfc.uy.transpose() << std::endl;
    std::cout << "uz: " << sfc.uz.transpose() << std::endl;
}

// ==========================================
// TEST 3: AsymmetricBoxBuilder — Open Corridor
// ==========================================
TEST(DynamicSFCTest, OpenCorridorNoBounds) {
    AsymmetricBoxBuilder builder(0.5, 2.5, 0.5); // drone_r=0.5, max_drift=2.5, voxel_res=0.5

    Eigen::Vector3d pA(0.0, 0.0, 5.0);
    Eigen::Vector3d pB(10.0, 0.0, 5.0);

    // No obstacles → bounds should remain at max_drift
    std::vector<Eigen::Vector3d> no_obstacles;
    auto [ux, uy, uz, bounds] = builder.build_bounds(pA, pB, no_obstacles, 1.0, 1.0);

    EXPECT_NEAR(bounds(2), 2.5, 1e-6); // +y unchanged
    EXPECT_NEAR(bounds(3), 2.5, 1e-6); // -y unchanged
    EXPECT_NEAR(bounds(4), 2.5, 1e-6); // +z unchanged
    EXPECT_NEAR(bounds(5), 2.5, 1e-6); // -z unchanged

    // Frame should be orthonormal
    EXPECT_NEAR(ux.dot(uy), 0.0, 1e-6);
    EXPECT_NEAR(ux.dot(uz), 0.0, 1e-6);

    std::cout << "\n--- Dynamic SFC Test (Open Corridor) ---" << std::endl;
    std::cout << "Bounds: " << bounds.transpose() << std::endl;
}

// ==========================================
// TEST 4: AsymmetricBoxBuilder — Obstacle Shrinking
// ==========================================
TEST(DynamicSFCTest, ObstacleShrinksBounds) {
    AsymmetricBoxBuilder builder(0.0, 5.0, 0.5); // No drone radius for clean math

    Eigen::Vector3d pA(0.0, 0.0, 5.0);
    Eigen::Vector3d pB(10.0, 0.0, 5.0);

    // Place an obstacle at (5, 2, 5) — right in the middle of the corridor, 2m to the left
    std::vector<Eigen::Vector3d> obstacles = {Eigen::Vector3d(5.0, 2.0, 5.0)};
    auto [ux, uy, uz, bounds] = builder.build_bounds(pA, pB, obstacles, 1.0, 1.0);

    // The +y bound should be shrunk below 5.0 (the max_drift)
    EXPECT_LT(bounds(2), 5.0);

    // +y is shrunk to (2.0 - r_y) = 1.75 by the obstacle.
    EXPECT_NEAR(bounds(2), 1.75, 1e-6);

    // -y does NOT stay at max_drift. build_bounds enforces an ASYMMETRIC drift budget: the
    // two opposing faces share a total of 2*max_cp_drift, so squeezing one side lets the
    // other take up the slack (10.0 - 1.75 = 8.25). Only when both sides are unconstrained
    // do they settle at max_cp_drift each. This test previously asserted 5.0 on all three,
    // which predates that redistribution.
    EXPECT_NEAR(bounds(3), 8.25, 1e-6);

    // z is untouched on both faces, so the budget splits evenly.
    EXPECT_NEAR(bounds(4), 5.0, 1e-6);
    EXPECT_NEAR(bounds(5), 5.0, 1e-6);

    std::cout << "\n--- Dynamic SFC Test (Obstacle Shrinking) ---" << std::endl;
    std::cout << "Bounds after shrink: " << bounds.transpose() << std::endl;
}

// ==========================================
// TEST 5: OBBAdapters — OSQP Matrix Generation
// ==========================================
TEST(DynamicSFCTest, OSQPMatrixGeneration) {
    Eigen::Vector3d pA(0.0, 0.0, 0.0);
    Eigen::Vector3d ux(1.0, 0.0, 0.0);
    Eigen::Vector3d uy(0.0, 1.0, 0.0);
    Eigen::Vector3d uz(0.0, 0.0, 1.0);

    Eigen::Matrix<double, 6, 1> bounds;
    bounds << 10.0, 2.0, 3.0, 3.0, 1.0, 1.0;

    Eigen::MatrixXd A_mat;
    Eigen::VectorXd b_vec;
    OBBAdapters::get_osqp_matrices(pA, ux, uy, uz, bounds, A_mat, b_vec);

    // A should be 6x3
    EXPECT_EQ(A_mat.rows(), 6);
    EXPECT_EQ(A_mat.cols(), 3);

    // For axis-aligned case with pA at origin:
    // b = A @ [0,0,0] + bounds = bounds
    EXPECT_NEAR(b_vec(0), 10.0, 1e-6); // +x
    EXPECT_NEAR(b_vec(1),  2.0, 1e-6); // -x
    EXPECT_NEAR(b_vec(2),  3.0, 1e-6); // +y
    EXPECT_NEAR(b_vec(3),  3.0, 1e-6); // -y
    EXPECT_NEAR(b_vec(4),  1.0, 1e-6); // +z
    EXPECT_NEAR(b_vec(5),  1.0, 1e-6); // -z

    std::cout << "\n--- OBB Adapter Test ---" << std::endl;
    std::cout << "A:\n" << A_mat << std::endl;
    std::cout << "b: " << b_vec.transpose() << std::endl;
}

// ==========================================
// TEST 6: SpatialScanner — Cylinder Filter
// ==========================================
TEST(DynamicSFCTest, SpatialScannerCylinderFilter) {
    // Create a cloud of points: some inside the corridor, some outside
    std::vector<Eigen::Vector3d> cloud = {
        {5.0, 0.5, 5.0},   // Inside cylinder (centered on the line, 0.5m lateral)
        {5.0, 10.0, 5.0},  // Far outside (10m away laterally)
        {-5.0, 0.0, 5.0},  // Behind the start (should be filtered by length)
        {5.0, 0.0, 5.0},   // Exactly on the line
    };

    auto grid = std::make_shared<mapping::SparseVoxelGrid>(0.5, 0.5);
    std::vector<mapping::ObstacleBox> obs_boxes;
    for(const auto& p : cloud) {
        obs_boxes.push_back({p - Eigen::Vector3d::Constant(0.25), p + Eigen::Vector3d::Constant(0.25)});
    }
    grid->update_from_obstacles(obs_boxes);

    SpatialScanner scanner(grid);

    Eigen::Vector3d pA(0.0, 0.0, 5.0);
    Eigen::Vector3d pB(10.0, 0.0, 5.0);

    auto result = scanner.get_broad_phase_obstacles(pA, pB, 2.0, 1.0, 1.0);

    // The scanner returns INFLATED VOXEL CENTRES from the grid, not the original cloud
    // points -- each point becomes a voxel-sized box grown by the inflation radius, which
    // spans many cells. So the count is not 2; asserting on it was a leftover from when
    // SpatialScanner took a raw point list. Assert the property that actually matters
    // instead: everything returned lies inside the query cylinder, and the two points that
    // should be excluded contributed nothing.
    ASSERT_FALSE(result.empty());

    const Eigen::Vector3d axis = (pB - pA).normalized();
    const double W = 2.0, ext_start = 1.0, ext_end = 1.0;
    const Eigen::Vector3d p_start = pA - axis * ext_start;
    const double total_len = (pB - pA).norm() + ext_start + ext_end;

    for (const auto& p : result) {
        const double t = (p - p_start).dot(axis);
        EXPECT_GE(t, 0.0);
        EXPECT_LE(t, total_len);
        EXPECT_LE((p - (p_start + t * axis)).norm(), W + 1e-9);
    }

    // Nothing from the far-lateral point (5,10,5) or the behind-the-start point (-5,0,5).
    for (const auto& p : result) {
        EXPECT_GT((p - Eigen::Vector3d(5.0, 10.0, 5.0)).norm(), 1.0);
        EXPECT_GT((p - Eigen::Vector3d(-5.0, 0.0, 5.0)).norm(), 1.0);
    }

    std::cout << "\n--- Spatial Scanner Test ---" << std::endl;
    std::cout << "Points in cylinder: " << result.size() << std::endl;
    for (const auto& p : result) {
        std::cout << "  " << p.transpose() << std::endl;
    }
}

// ==========================================
// TEST 7: Full AsymmetricSFCManager Pipeline
// ==========================================
TEST(DynamicSFCTest, FullPipeline) {
    std::vector<Eigen::Vector3d> obstacles = {
        {5.0, 1.5, 5.0},
        {5.0, -1.5, 5.0},
    };

    auto grid = std::make_shared<mapping::SparseVoxelGrid>(0.5, 0.5);
    std::vector<mapping::ObstacleBox> obs_boxes;
    for(const auto& p : obstacles) {
        obs_boxes.push_back({p - Eigen::Vector3d::Constant(0.25), p + Eigen::Vector3d::Constant(0.25)});
    }
    grid->update_from_obstacles(obs_boxes);

    AsymmetricSFCManager manager(grid, 0.0, 5.0, 0.5);

    Eigen::Vector3d pA(0.0, 0.0, 5.0);
    Eigen::Vector3d pB(10.0, 0.0, 5.0);

    SFCResult sfc = manager.generate_sfc(pA, pB, 3.0, 1.0, 1.0);

    EXPECT_EQ(sfc.A_mat.rows(), 6);
    EXPECT_EQ(sfc.A_mat.cols(), 3);
    EXPECT_EQ(sfc.b_vec.size(), 6);

    // Both +y and -y should be shrunk from max_drift due to the symmetric obstacles
    EXPECT_LT(sfc.bounds(2), 5.0);
    EXPECT_LT(sfc.bounds(3), 5.0);

    std::cout << "\n--- Full Pipeline Test ---" << std::endl;
    std::cout << "Bounds: " << sfc.bounds.transpose() << std::endl;
    std::cout << "A_mat:\n" << sfc.A_mat << std::endl;
    std::cout << "b_vec: " << sfc.b_vec.transpose() << std::endl;
}

} // namespace tests
} // namespace sfc
} // namespace trajectory_planner
