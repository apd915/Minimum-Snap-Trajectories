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
static std::vector<mapping::ObstacleBox> build_test_obstacles() {
    std::vector<mapping::ObstacleBox> obstacles;

    // A wall at X=50, spanning Y=[30,70], Z=[0,15]
    obstacles.push_back({{50.0, 30.0, 0.0}, {52.0, 70.0, 15.0}});

    // A pillar at (30, 50), Z=[0,10]
    obstacles.push_back({{28.0, 48.0, 0.0}, {32.0, 52.0, 10.0}});

    // A pillar at (70, 50), Z=[0,10]
    obstacles.push_back({{68.0, 48.0, 0.0}, {72.0, 52.0, 10.0}});

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
    FrontEndSFC front_end(config, obstacles);

    Eigen::Vector3d start(2.0, 2.0, 5.0);
    Eigen::Vector3d goal(98.0, 98.0, 10.0);

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
    FrontEndSFC front_end(config, no_obstacles);

    Eigen::Vector3d start(2.0, 2.0, 5.0);
    Eigen::Vector3d goal(48.0, 48.0, 10.0);

    auto result = front_end.get_corridors_astar(start, goal);

    // With no obstacles, A* should find a nearly straight path
    // The smoother should collapse it to very few waypoints
    ASSERT_GT(result.corridors.size(), 0);

    std::cout << "\n--- Empty Map Test ---" << std::endl;
    std::cout << "Corridors: " << result.corridors.size() << std::endl;
}

} // namespace tests
} // namespace trajectory_planner
