#pragma once

#include <Eigen/Dense>
#include <vector>
#include <unordered_set>
#include <queue>
#include <memory>

namespace trajectory_planner {
namespace path_finding {

// ==========================================
// 1. NODE & DATA STRUCTURES (Replaces node.py)
// ==========================================
struct Node {
    double f_cost; // Placed first, mimicking your dataclass order
    double g_cost;
    double h_cost;
    Eigen::Vector3i position;
    Node* parent;

    Node(double f, double g, double h, Eigen::Vector3i pos, Node* p = nullptr) 
        : f_cost(f), g_cost(g), h_cost(h), position(pos), parent(p) {}
};

// Custom Hash for Eigen::Vector3i to use O(1) unordered_set lookups
struct Vector3iHash {
    std::size_t operator()(const Eigen::Vector3i& v) const {
        std::size_t h1 = std::hash<int>()(v.x());
        std::size_t h2 = std::hash<int>()(v.y());
        std::size_t h3 = std::hash<int>()(v.z());
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

// Priority Queue Min-Heap Comparator
struct CompareNode {
    bool operator()(const Node* a, const Node* b) const {
        return a->f_cost > b->f_cost; 
    }
};

struct BoundingBox {
    Eigen::Vector3d b_min;
    Eigen::Vector3d b_max;
};

// Lightweight Interface to hold your Voxel Grid data
struct VoxelGridData {
    double voxel_resolution;
    std::unordered_set<Eigen::Vector3i, Vector3iHash> occupied_voxels_inflated;
    std::vector<BoundingBox> continuous_inflated_bounds; 
};

// ==========================================
// 2. PLANNER INTERFACE (Replaces astar_sfc.py)
// ==========================================
class AStarSFCPlanner {
public:
    // Pass the pre-computed grid and the discrete bounding limits
    AStarSFCPlanner(const VoxelGridData& grid, const Eigen::Vector3i& bounds_min, const Eigen::Vector3i& bounds_max);

    std::vector<Eigen::Vector3i> search(const Eigen::Vector3i& start_pos, const Eigen::Vector3i& goal_pos);
    std::vector<Eigen::Vector3i> sfc_smoother();
    
    bool is_line_of_sight_clear(const Eigen::Vector3i& idx_a, const Eigen::Vector3i& idx_b);
    double get_safe_extension_length(const Eigen::Vector3i& idx_a, const Eigen::Vector3i& idx_b, double requested_extension);

    std::vector<Eigen::Vector3i> get_path() const { return path_; }

private:
    std::vector<Eigen::Vector3i> reconstruct_path(Node* end_node);

    VoxelGridData voxel_grid_;
    Eigen::Vector3i min_bounds_;
    Eigen::Vector3i max_bounds_;

    // Store precalculated 26 directions: (dx, dy, dz, move_cost)
    struct Direction {
        int dx, dy, dz;
        double cost;
    };
    std::vector<Direction> directions_;
    
    std::vector<Eigen::Vector3i> path_;
};

} // namespace path_finding
} // namespace trajectory_planner