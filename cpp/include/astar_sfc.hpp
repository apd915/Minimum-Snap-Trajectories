#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include "voxel_grid.hpp"
#include <vector>
#include <unordered_set>
#include <queue>
#include <memory>
#include <string>
#include <limits>

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
using Vector3iHash = mapping::Vector3iHash;

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
using VoxelSet = std::unordered_set<Eigen::Vector3i, Vector3iHash>;

// Lightweight VIEW over a SparseVoxelGrid.
//
// These are borrowed pointers, not owned copies. The planner is rebuilt on every replan, so
// copying the occupancy and free sets in here cost several milliseconds per cycle and grew
// without bound as the map accumulated. The referenced grid must outlive the planner --
// FrontEndSFC holds a shared_ptr to it, which guarantees that.
struct VoxelGridData {
    double voxel_resolution = 0.5;
    const VoxelSet* occupied_voxels_inflated = nullptr;
    const VoxelSet* free_voxels = nullptr;
    const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>* continuous_inflated_bounds = nullptr;

    bool has_occupancy() const { return occupied_voxels_inflated && !occupied_voxels_inflated->empty(); }
    bool has_free() const { return free_voxels && !free_voxels->empty(); }
};

// ==========================================
// 2. PLANNER INTERFACE (Replaces astar_sfc.py)
// ==========================================
class AStarSFCPlanner {
public:
    // Pass the pre-computed grid and the discrete bounding limits
    AStarSFCPlanner(const VoxelGridData& grid, const Eigen::Vector3i& bounds_min, const Eigen::Vector3i& bounds_max);

    std::vector<Eigen::Vector3i> search(const Eigen::Vector3i& start_pos, const Eigen::Vector3i& goal_pos,
                                         double max_time_ms = 10.0,
                                         double sensor_range_voxels = -1.0,
                                         int max_z_index = std::numeric_limits<int>::max(),
                                         double fov_vertical_min_deg = -90.0,
                                         double fov_vertical_max_deg = 90.0,
                                         double unknown_traversal_cost = 2.0,
                                         int max_nodes = 6000,
                                         double clearance_penalty = 1.0);
    std::vector<Eigen::Vector3i> sfc_smoother();
    
    /**
     * @param margin_cells extra clearance demanded either side of the ray, in voxels. 0 is the
     *        safety test (does the line itself hit anything); >0 additionally refuses lines that
     *        merely graze inflation. See sfc_smoother() for why that distinction matters.
     */
    bool is_line_of_sight_clear(const Eigen::Vector3i& idx_a, const Eigen::Vector3i& idx_b,
                                int margin_cells = 0);

    /** Amanatides-Woo walk over the inflated occupancy set. True if the ray hits anything. */
    bool discrete_ray_hits_obstacle(const Eigen::Vector3i& idx_a, const Eigen::Vector3i& idx_b,
                                    int margin_cells = 0) const;
    double get_safe_extension_length(const Eigen::Vector3i& idx_a, const Eigen::Vector3i& idx_b, double requested_extension);

    std::vector<Eigen::Vector3i> get_path() const { return path_; }
    std::vector<Eigen::Vector3i> get_shadowed_voxels() const { return shadowed_voxels_; }

private:
    std::vector<Eigen::Vector3i> reconstruct_path(Node* end_node);

    const VoxelGridData& voxel_grid_;
    Eigen::Vector3i min_bounds_;
    Eigen::Vector3i max_bounds_;

    double fov_min_rad_ = -M_PI;
    double fov_max_rad_ = M_PI;
    bool check_fov_ = false;
    double clearance_cost_ = 1.0;
    double unknown_cost_ = 2.0;
    Eigen::Vector3i search_start_pos_;

    // Store precalculated 26 directions: (dx, dy, dz, move_cost)
    struct Direction {
        int dx, dy, dz;
        double cost;
    };
    std::vector<Direction> directions_;
    
    std::vector<Eigen::Vector3i> path_;
    std::vector<Eigen::Vector3i> shadowed_voxels_;
};

} // namespace path_finding
} // namespace trajectory_planner