#include "astar_sfc.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>

namespace trajectory_planner {
namespace path_finding {

AStarSFCPlanner::AStarSFCPlanner(const VoxelGridData& grid, const Eigen::Vector3i& bounds_min, const Eigen::Vector3i& bounds_max) 
    : voxel_grid_(grid), min_bounds_(bounds_min), max_bounds_(bounds_max) 
{
    // Precalculate the 26 directions to avoid math in the inner loop!
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                double cost = std::sqrt(dx*dx + dy*dy + dz*dz);
                directions_.push_back({dx, dy, dz, cost});
            }
        }
    }
}

std::vector<Eigen::Vector3i> AStarSFCPlanner::reconstruct_path(Node* end_node) {
    path_.clear();
    Node* current = end_node;
    while (current != nullptr) {
        path_.push_back(current->position);
        current = current->parent;
    }
    std::reverse(path_.begin(), path_.end());
    return path_;
}

std::vector<Eigen::Vector3i> AStarSFCPlanner::search(const Eigen::Vector3i& start_pos, const Eigen::Vector3i& goal_pos) {
    std::priority_queue<Node*, std::vector<Node*>, CompareNode> open_list;
    std::unordered_set<Eigen::Vector3i, Vector3iHash> closed_set;
    
    // --- MEMORY ARENA (Replaces Python Garbage Collector) ---
    // This safely owns all nodes. When 'search' returns, it instantly deletes all of them.
    std::vector<std::unique_ptr<Node>> node_memory;
    node_memory.reserve(50000); // Pre-allocate to prevent mid-search OS bottleneck
    
    node_memory.push_back(std::make_unique<Node>(0.0, 0.0, 0.0, start_pos, nullptr));
    open_list.push(node_memory.back().get());
    
    int nodes_expanded = 0;
    
    while (!open_list.empty()) {
        nodes_expanded++;
        Node* current_node = open_list.top();
        open_list.pop();
        
        // --- LAZY DELETION CHECK ---
        if (closed_set.find(current_node->position) != closed_set.end()) {
            continue;
        }
        
        if (current_node->position == goal_pos) {
            std::cout << "[ A* ] Path found! Explored " << nodes_expanded << " nodes." << std::endl;
            return reconstruct_path(current_node);
        }
        
        closed_set.insert(current_node->position);
        
        for (const auto& dir : directions_) {
            Eigen::Vector3i neighbor_pos(
                current_node->position.x() + dir.dx,
                current_node->position.y() + dir.dy,
                current_node->position.z() + dir.dz
            );
            
            // 1. Geofence Check
            if (neighbor_pos.x() < min_bounds_.x() || neighbor_pos.x() > max_bounds_.x() ||
                neighbor_pos.y() < min_bounds_.y() || neighbor_pos.y() > max_bounds_.y() ||
                neighbor_pos.z() < min_bounds_.z() || neighbor_pos.z() > max_bounds_.z()) {
                continue;
            }
            
            // 2. Collision Check (O(1) hash lookup)
            if (voxel_grid_.occupied_voxels_inflated.find(neighbor_pos) != voxel_grid_.occupied_voxels_inflated.end()) {
                continue;
            }
            
            // 3. Skip fully evaluated nodes
            if (closed_set.find(neighbor_pos) != closed_set.end()) {
                continue;
            }
            
            double new_g = current_node->g_cost + dir.cost;
            // WEIGHTED A* (2.0 multiplier matches your Python Magic Number)
            double new_h = (neighbor_pos - goal_pos).cast<double>().norm() * 2.0; 
            double new_f = new_g + new_h;
            
            // Create neighbor and push pointer to Priority Queue
            node_memory.push_back(std::make_unique<Node>(new_f, new_g, new_h, neighbor_pos, current_node));
            open_list.push(node_memory.back().get());
        }
    }
    
    std::cout << "[ A* ] Open list emptied, no path found." << std::endl;
    return std::vector<Eigen::Vector3i>();
}

std::vector<Eigen::Vector3i> AStarSFCPlanner::sfc_smoother() {
    if (path_.size() <= 2) return path_;
    
    std::vector<Eigen::Vector3i> smoothed_path;
    smoothed_path.push_back(path_[0]);
    int anchor_idx = 0;
    
    // The Look-Ahead Loop
    for (size_t i = 2; i < path_.size(); ++i) {
        if (!is_line_of_sight_clear(path_[anchor_idx], path_[i])) {
            smoothed_path.push_back(path_[i - 1]);
            anchor_idx = i - 1;
        }
    }
    smoothed_path.push_back(path_.back());
    
    path_ = smoothed_path;
    return path_;
}

bool AStarSFCPlanner::is_line_of_sight_clear(const Eigen::Vector3i& idx_a, const Eigen::Vector3i& idx_b) {
    double res = voxel_grid_.voxel_resolution;
    
    // --- 1. CONTINUOUS FRONT-END ---
    if (!voxel_grid_.continuous_inflated_bounds.empty()) {
        Eigen::Vector3d p0 = idx_a.cast<double>() * res;
        Eigen::Vector3d p1 = idx_b.cast<double>() * res;
        Eigen::Vector3d d = p1 - p0;
        
        Eigen::Vector3d inv_d;
        for (int i = 0; i < 3; ++i) {
            // Avoid division by zero!
            inv_d(i) = (d(i) == 0.0) ? std::numeric_limits<double>::infinity() : 1.0 / d(i);
        }
        
        for (const auto& box : voxel_grid_.continuous_inflated_bounds) {
            Eigen::Vector3d t1 = (box.b_min - p0).cwiseProduct(inv_d);
            Eigen::Vector3d t2 = (box.b_max - p0).cwiseProduct(inv_d);
            
            Eigen::Vector3d t_min = t1.cwiseMin(t2);
            Eigen::Vector3d t_max = t1.cwiseMax(t2);
            
            double t_enter = t_min.maxCoeff();
            double t_exit = t_max.minCoeff();
            
            if (t_enter <= t_exit && t_exit >= 0.0 && t_enter <= 1.0) {
                return false; // Hit an obstacle
            }
        }
        return true;
    } 
    // --- 2. DISCRETE LIDAR ---
    else if (!voxel_grid_.occupied_voxels_inflated.empty()) {
        Eigen::Vector3d p0 = idx_a.cast<double>();
        Eigen::Vector3d p1 = idx_b.cast<double>();
        double dist = (p1 - p0).norm();
        if (dist == 0) return true;
        
        int steps = std::ceil(dist * 2.0);
        for (int i = 1; i < steps; ++i) {
            double t = (double)i / steps;
            Eigen::Vector3d point = p0 + t * (p1 - p0);
            Eigen::Vector3i voxel = point.array().round().cast<int>();
            
            if (voxel_grid_.occupied_voxels_inflated.find(voxel) != voxel_grid_.occupied_voxels_inflated.end()) {
                return false; // Hit a building block
            }
        }
        return true;
    }
    return true;
}

double AStarSFCPlanner::get_safe_extension_length(const Eigen::Vector3i& idx_a, const Eigen::Vector3i& idx_b, double requested_extension) {
    if (requested_extension <= 0.0) return 0.0;
    double res = voxel_grid_.voxel_resolution;
    
    // --- 1. CONTINUOUS FRONT-END ---
    if (!voxel_grid_.continuous_inflated_bounds.empty()) {
        Eigen::Vector3d p0 = idx_a.cast<double>() * res;
        Eigen::Vector3d p1 = idx_b.cast<double>() * res;
        Eigen::Vector3d d = p1 - p0;
        double dist = d.norm();
        if (dist == 0) return 0.0;
        
        Eigen::Vector3d dir_unit = d / dist;
        Eigen::Vector3d inv_d;
        for(int i = 0; i < 3; ++i) {
            inv_d(i) = (dir_unit(i) == 0.0) ? std::numeric_limits<double>::infinity() : 1.0 / dir_unit(i);
        }
        
        double min_safe_distance = requested_extension;
        for (const auto& box : voxel_grid_.continuous_inflated_bounds) {
            Eigen::Vector3d t1 = (box.b_min - p1).cwiseProduct(inv_d);
            Eigen::Vector3d t2 = (box.b_max - p1).cwiseProduct(inv_d);
            
            Eigen::Vector3d t_min = t1.cwiseMin(t2);
            Eigen::Vector3d t_max = t1.cwiseMax(t2);
            
            double t_enter = t_min.maxCoeff();
            double t_exit = t_max.minCoeff();
            
            if (t_enter <= t_exit && t_exit >= 0.0) {
                if (t_enter > 0 && t_enter < min_safe_distance) {
                    min_safe_distance = std::max(0.0, t_enter - 0.01);
                }
            }
        }
        return min_safe_distance;
    } 
    // --- 2. DISCRETE LIDAR ---
    else if (!voxel_grid_.occupied_voxels_inflated.empty()) {
        Eigen::Vector3d p0 = idx_a.cast<double>() * res;
        Eigen::Vector3d p1 = idx_b.cast<double>() * res;
        Eigen::Vector3d d = p1 - p0;
        double dist = d.norm();
        if (dist == 0) return 0.0;
        
        Eigen::Vector3d dir_unit = d / dist;
        double step_size = res / 2.0; // Half-voxel safety steps
        int num_steps = std::ceil(requested_extension / step_size);
        
        double min_safe_distance = requested_extension;
        for (int i = 1; i <= num_steps; ++i) {
            double t_dist = i * step_size;
            Eigen::Vector3d point = p1 + dir_unit * t_dist;
            Eigen::Vector3i voxel = (point / res).array().round().cast<int>();
            
            if (voxel_grid_.occupied_voxels_inflated.find(voxel) != voxel_grid_.occupied_voxels_inflated.end()) {
                min_safe_distance = std::max(0.0, t_dist - step_size);
                break;
            }
        }
        return min_safe_distance;
    }
    return requested_extension;
}

} // namespace path_finding
} // namespace trajectory_planner