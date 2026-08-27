#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <iostream>
#include <chrono>
#include "front_end.hpp"

namespace trajectory_planner {
namespace tests {

// ==========================================
// Helper: Build a simple obstacle field for testing
// ==========================================

// ---------------------------------------------------------------------------------------
// COORDINATE CONVENTION
//
// These fixtures were originally written for a 0..100 m positive-orthant world with
// POSITIVE-UP z. The planner works in NED: the A* geofence is +/- map_bounds/2 about the
// ORIGIN, and z is NEGATIVE-up (altitude 5 m is z = -5). Under the old coordinates the start
// sat several metres underground -- rejected by the min_altitude floor -- and the goal fell
// outside the geofence entirely, so A* correctly returned a single-point path and every
// assertion on corridor count failed. The scene below is the same layout expressed in NED:
// x/y centred on the origin, altitudes negative and inside the +/- map_bounds.z()/2 band.
// ---------------------------------------------------------------------------------------
static std::vector<mapping::ObstacleBox> build_test_obstacles() {
    std::vector<mapping::ObstacleBox> obstacles;

    // A wall at X=0, spanning Y=[-20,20], from ground up to 7 m altitude
    obstacles.push_back({{0.0, -20.0, -7.0}, {2.0, 20.0, 0.0}});

    // A pillar at (-20, 0), up to 5 m altitude
    obstacles.push_back({{-22.0, -2.0, -5.0}, {-18.0, 2.0, 0.0}});

    // A pillar at (20, 0), up to 5 m altitude
    obstacles.push_back({{18.0, -2.0, -5.0}, {22.0, 2.0, 0.0}});

    return obstacles;
}

// ==========================================
// TEST 1: FrontEnd — Corridor Generation
// ==========================================
TEST(FrontEndTest, GeneratesCorridors) {
    FrontEndConfig config;
    config.voxel_resolution = 0.5;
    config.drone_physical_radius = 0.5;
    config.sfc_height = 5.0;
    config.sfc_width = 5.0;
    config.sfc_start_ext = 5.0;
    config.sfc_end_ext = 5.0;
    config.spline_type = "clamped";
    config.aircraft_type = "multi-rotor";
    config.degree = 4;
    config.map_bounds = Eigen::Vector3d(100.0, 100.0, 15.0);

    auto obstacles = build_test_obstacles();
    auto grid = std::make_shared<mapping::SparseVoxelGrid>(config.voxel_resolution, config.drone_physical_radius);
    grid->update_from_obstacles(obstacles);
    FrontEndSFC front_end(config, grid);

    Eigen::Vector3d start(-30.0, -30.0, -4.0);
    Eigen::Vector3d goal(30.0, 30.0, -5.0);

    auto t1 = std::chrono::high_resolution_clock::now();
    auto result = front_end.get_corridors_astar(start, goal);
    auto t2 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    ASSERT_GT(result.corridors.size(), 0) << "No corridors generated!";
    ASSERT_GT(result.allocation_pools.size(), 0) << "No allocation pools generated!";

    // Verify all corridors have valid A/b matrices
    for (const auto& sfc : result.corridors) {
        EXPECT_EQ(sfc.A_mat.rows(), 6);
        EXPECT_EQ(sfc.A_mat.cols(), 3);
        EXPECT_EQ(sfc.b_vec.size(), 6);
    }

    // Verify constraint compilation works
    Eigen::MatrixXd A_sfc;
    Eigen::VectorXd b_sfc;
    int total_pts;
    FrontEndSFC::compile_system_constraints(
        result.corridors, result.allocation_pools, A_sfc, b_sfc, total_pts);

    EXPECT_GT(total_pts, 0);
    EXPECT_GT(A_sfc.rows(), 0);
    EXPECT_EQ(A_sfc.cols(), total_pts * 3);

    std::cout << "\n============================================" << std::endl;
    std::cout << "  C++ FRONT-END TEST RESULTS" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Time:               " << ms << " ms" << std::endl;
    std::cout << "Corridors:          " << result.corridors.size() << std::endl;
    std::cout << "Allocation Pools:   " << result.allocation_pools.size() << std::endl;
    std::cout << "Total Ctrl Points:  " << total_pts << std::endl;
    std::cout << "A_sfc size:         " << A_sfc.rows() << " x " << A_sfc.cols() << std::endl;
    std::cout << "============================================\n" << std::endl;
}

// ==========================================
// TEST 2: FrontEnd — Empty Map (No Obstacles)
// ==========================================
TEST(FrontEndTest, EmptyMap) {
    FrontEndConfig config;
    config.voxel_resolution = 1.0;
    config.drone_physical_radius = 0.5;
    config.sfc_height = 5.0;
    config.sfc_width = 5.0;
    config.degree = 4;
    config.map_bounds = Eigen::Vector3d(50.0, 50.0, 15.0);

    std::vector<mapping::ObstacleBox> no_obstacles;
    auto grid = std::make_shared<mapping::SparseVoxelGrid>(config.voxel_resolution, config.drone_physical_radius);
    grid->update_from_obstacles(no_obstacles);
    FrontEndSFC front_end(config, grid);

    Eigen::Vector3d start(-20.0, -20.0, -4.0);
    Eigen::Vector3d goal(20.0, 20.0, -5.0);

    auto result = front_end.get_corridors_astar(start, goal);

    // With no obstacles, A* should find a nearly straight path
    // The smoother should collapse it to very few waypoints
    ASSERT_GT(result.corridors.size(), 0);

    std::cout << "\n--- Empty Map Test ---" << std::endl;
    std::cout << "Corridors: " << result.corridors.size() << std::endl;
}

} // namespace tests
} // namespace trajectory_planner
