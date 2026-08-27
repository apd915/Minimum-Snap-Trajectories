#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <chrono>
#include <random>
#include <iostream>
#include "voxel_grid.hpp"
#include "ring_buffer.hpp"

namespace trajectory_planner {
namespace mapping {
namespace tests {

// ==========================================
// TEST 1: SparseVoxelGrid Populate & Query
// ==========================================
TEST(VoxelGridTest, PopulateFromObstacles) {
    SparseVoxelGrid grid(1.0, 1.0);

    // Create a single 3x3x3 obstacle from (2,2,2) to (4,4,4)
    std::vector<ObstacleBox> obstacles;
    obstacles.push_back({{2.0, 2.0, 2.0}, {4.0, 4.0, 4.0}});

    // Inflate by 1.0 meter
    grid.update_from_obstacles(obstacles);

    // The raw obstacle should span voxels (2,2,2) to (4,4,4)
    EXPECT_TRUE(grid.is_occupied_raw(Eigen::Vector3i(2, 2, 2)));
    EXPECT_TRUE(grid.is_occupied_raw(Eigen::Vector3i(3, 3, 3)));
    EXPECT_TRUE(grid.is_occupied_raw(Eigen::Vector3i(4, 4, 4)));
    EXPECT_FALSE(grid.is_occupied_raw(Eigen::Vector3i(5, 5, 5)));

    // The inflated obstacle should span voxels (1,1,1) to (5,5,5)
    EXPECT_TRUE(grid.is_occupied_inflated(Eigen::Vector3i(1, 1, 1)));
    EXPECT_TRUE(grid.is_occupied_inflated(Eigen::Vector3i(5, 5, 5)));
    EXPECT_FALSE(grid.is_occupied_inflated(Eigen::Vector3i(6, 6, 6)));

    // Verify continuous bounds were saved
    EXPECT_EQ(grid.get_continuous_inflated_bounds().size(), 1);

    std::cout << "\n--- VoxelGrid Population Test ---" << std::endl;
    std::cout << "Raw voxels:     " << grid.num_raw() << std::endl;
    std::cout << "Inflated voxels: " << grid.num_inflated() << std::endl;
}

// ==========================================
// TEST 2: RingBuffer Populate & Query Parity
// ==========================================
TEST(VoxelGridTest, RingBufferParityWithHashSet) {
    SparseVoxelGrid grid(1.0, 1.0);

    // Create a wall obstacle at X=5, spanning Y=[0,10], Z=[0,5]
    std::vector<ObstacleBox> obstacles;
    obstacles.push_back({{5.0, 0.0, 0.0}, {5.0, 10.0, 5.0}});
    grid.update_from_obstacles(obstacles);

    // Build a ring buffer covering the same space
    RingBufferGrid ring(20, 20, 10);

    // Populate ring buffer from the hash set
    for (const auto& idx : grid.get_occupied_inflated()) {
        ring.set_occupied(idx.x(), idx.y(), idx.z());
    }

    // Verify EVERY inflated voxel in the hash set is also in the ring buffer
    int mismatches = 0;
    for (const auto& idx : grid.get_occupied_inflated()) {
        if (!ring.is_occupied(idx.x(), idx.y(), idx.z())) {
            mismatches++;
        }
    }
    EXPECT_EQ(mismatches, 0) << "Ring buffer is missing " << mismatches << " voxels!";

    // Verify a known free voxel is free in both
    EXPECT_FALSE(grid.is_occupied_inflated(Eigen::Vector3i(0, 0, 0)));
    EXPECT_FALSE(ring.is_occupied(0, 0, 0));

    std::cout << "\n--- Ring Buffer Parity Test ---" << std::endl;
    std::cout << "Hash set voxels:  " << grid.num_inflated() << std::endl;
    std::cout << "Ring buffer size: " << ring.total_voxels() << " voxels ("
              << ring.memory_bytes() / 1024 << " KB)" << std::endl;
    std::cout << "Mismatches: " << mismatches << std::endl;
}

// ==========================================
// TEST 3: Benchmark — Hash Set vs Ring Buffer
// ==========================================
TEST(VoxelGridTest, BenchmarkHashSetVsRingBuffer) {
    // Build a dense obstacle field: 50 random obstacles in a 100x100x30 space
    SparseVoxelGrid grid(0.5, 1.0);

    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<double> pos_dist(5.0, 95.0);
    std::uniform_real_distribution<double> size_dist(1.0, 5.0);

    std::vector<ObstacleBox> obstacles;
    for (int i = 0; i < 50; ++i) {
        double cx = pos_dist(rng), cy = pos_dist(rng), cz = pos_dist(rng) * 0.3;
        double sx = size_dist(rng), sy = size_dist(rng), sz = size_dist(rng);
        obstacles.push_back({
            {cx - sx/2, cy - sy/2, cz - sz/2},
            {cx + sx/2, cy + sy/2, cz + sz/2}
        });
    }

    grid.update_from_obstacles(obstacles);

    // Build ring buffer
    RingBufferGrid ring(200, 200, 60);
    for (const auto& idx : grid.get_occupied_inflated()) {
        ring.set_occupied(idx.x(), idx.y(), idx.z());
    }

    // Generate 100k random query points
    std::uniform_int_distribution<int> query_dist(0, 199);
    std::vector<Eigen::Vector3i> queries;
    queries.reserve(100000);
    for (int i = 0; i < 100000; ++i) {
        queries.emplace_back(query_dist(rng), query_dist(rng), query_dist(rng) % 60);
    }

    // Benchmark Hash Set
    volatile bool sink = false; // Prevent compiler from optimizing away the lookups
    auto t1 = std::chrono::high_resolution_clock::now();
    for (const auto& q : queries) {
        sink = grid.is_occupied_inflated(q);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto hash_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    // Benchmark Ring Buffer
    auto start = std::chrono::high_resolution_clock::now();
    for (const auto& q : queries) {
        sink = ring.is_occupied(q.x(), q.y(), q.z());
    }
    auto end = std::chrono::high_resolution_clock::now();
    (void)sink;
    auto ring_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "\n============================================" << std::endl;
    std::cout << "  C++ BENCHMARK: 100k Random Lookups" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Inflated voxels in map: " << grid.num_inflated() << std::endl;
    std::cout << "Ring buffer memory:     " << ring.memory_bytes() / 1024 << " KB" << std::endl;
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << "Hash Set (unordered_set):  " << hash_us << " µs" << std::endl;
    std::cout << "Ring Buffer (flat array):  " << ring_us << " µs" << std::endl;
    if (ring_us < hash_us) {
        std::cout << "Ring Buffer is " << static_cast<double>(hash_us) / ring_us
                  << "x FASTER!" << std::endl;
    } else {
        std::cout << "Hash Set is " << static_cast<double>(ring_us) / hash_us
                  << "x FASTER!" << std::endl;
    }
    std::cout << "============================================\n" << std::endl;

    // The test itself just needs to not crash — the benchmark is informational
    SUCCEED();
}

} // namespace tests
} // namespace mapping
} // namespace trajectory_planner
