#pragma once

#include <Eigen/Dense>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <cmath>
#include <cstdint>
#include <memory>
#include "../deps/nanoflann.hpp"

namespace trajectory_planner {
namespace mapping {

// ==========================================
// Custom Hash for Eigen::Vector3i
// ==========================================
struct Vector3iHash {
    // FNV-1a. The previous form was h(x) ^ (h(y)<<1) ^ (h(z)<<2), and since libstdc++'s
    // std::hash<int> is the identity that reduces to x ^ (y<<1) ^ (z<<2) -- which collides
    // constantly on the small, dense, grid-shaped indices this map is full of. Those
    // collisions turn every unordered_set probe into a bucket walk, so occupancy lookups
    // degrade toward O(n) exactly as the map grows.
    std::size_t operator()(const Eigen::Vector3i& v) const {
        std::size_t h = 1469598103934665603ULL;          // FNV offset basis
        h = (h ^ static_cast<std::uint32_t>(v.x())) * 1099511628211ULL;
        h = (h ^ static_cast<std::uint32_t>(v.y())) * 1099511628211ULL;
        h = (h ^ static_cast<std::uint32_t>(v.z())) * 1099511628211ULL;
        return h;
    }
};

// ==========================================
// Obstacle Bounding Box (Input Format)
// ==========================================
struct ObstacleBox {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3d min;
    Eigen::Vector3d max;
};

// ==========================================
// Adaptor for nanoflann KD-Tree
// ==========================================
struct Vector3dAdaptor {
    const std::vector<Eigen::Vector3d>& pts;
    Vector3dAdaptor(const std::vector<Eigen::Vector3d>& pts_) : pts(pts_) {}
    inline size_t kdtree_get_point_count() const { return pts.size(); }
    inline double kdtree_get_pt(const size_t idx, const size_t dim) const { return pts[idx](dim); }
    template <class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
};

using KDTreeType = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, Vector3dAdaptor>, 
    Vector3dAdaptor, 3>;

// ==========================================
// SparseVoxelGrid (Incremental + KD-Tree)
// ==========================================
class SparseVoxelGrid {
public:
    explicit SparseVoxelGrid(double resolution = 1.0, double inflation_radius = 0.0);

    /**
     * Incrementally updates the grid from a new set of obstacles.
     * Clears old data and repopulates, rebuilding the KD-tree.
     *
     * NOTE: this marks OCCUPIED cells only. Every other cell is left UNKNOWN, because a
     * bare list of obstacle boxes carries no information about what was *observed to be
     * empty*. Prefer update_from_scan() when the data came from a ranging sensor.
     */
    void update_from_obstacles(const std::vector<ObstacleBox>& obstacles);

    /**
     * Updates the grid from a ranging-sensor scan, producing a three-state map.
     *
     * Each beam is traversed from sensor_origin to its return point (Amanatides-Woo), and
     * every cell along the way is marked FREE -- that space was *measured* to be empty.
     * The return points themselves are marked OCCUPIED (and inflated by the drone radius
     * for planning). Anything neither traversed nor hit stays UNKNOWN.
     *
     * This is what lets the planner tell "seen and empty" apart from "never looked at",
     * instead of inferring it after the fact from line-of-sight raycasts.
     *
     * @param sensor_origin  Beam origin in world/NED metres.
     * @param hits           Beam return points in world/NED metres.
     */
    void update_from_scan(const Eigen::Vector3d& sensor_origin,
                          const std::vector<Eigen::Vector3d>& hits);

    /**
     * Enables accumulation of observations across scans (default: on).
     *
     * With a single scan the beam pattern is angular, so past ~9.5 m (3 deg beams, 0.5 m
     * voxels) neighbouring beams are further apart than a voxel and measured-free space
     * degenerates into a fan of thin lines separated by cells nothing happened to sample.
     * That makes the FREE/UNKNOWN boundary meaningless. Accumulating successive scans fills
     * those gaps in, because the vehicle moves and each scan samples different cells --
     * and it is also what lets the planner keep using space it saw a moment ago but is no
     * longer looking at.
     *
     * Staleness is handled by raycast clearing rather than by forgetting everything:
     * a beam that passes THROUGH a cell erases any occupancy there. An obstacle that moves
     * away is cleared the first time a beam reaches through its old position.
     *
     * @param window_radius_m  Cells further than this from the sensor are dropped, which
     *                         bounds both memory and the cost of the periodic prune.
     */
    void set_accumulate(bool on, double window_radius_m = 25.0) {
        accumulate_ = on;
        map_window_radius_ = window_radius_m;
    }

    /**
     * Limits frontier extraction to this radius around the last scan origin.
     *
     * The frontier is only ever consumed as a local barrier for corridor construction, so
     * extracting it across the whole retained map is wasted work that grows with flight
     * time. Cost scales with the CUBE of this radius, so trimming it to just beyond the
     * planning horizon is the single cheapest lever on front-end time.
     */
    void set_frontier_radius(double r) { frontier_radius_ = r; frontier_dirty_ = true; }

    /**
     * Focuses frontier extraction on a specific region instead of a sphere about the sensor.
     *
     * Corridors are built along the planned path, which is a tube, not a ball centred on the
     * vehicle. Sizing the extraction to the path's bounding sphere covers exactly the space
     * the SFC builder will query while skipping the rest of the retained map -- the
     * difference is several milliseconds per replan once the map has accumulated.
     */
    void set_frontier_focus(const Eigen::Vector3d& center, double radius) {
        frontier_focus_ = center;
        frontier_radius_ = radius;
        has_frontier_focus_ = true;
        frontier_dirty_ = true;
    }
    bool accumulating() const { return accumulate_; }

    /** True if this cell was traversed by a sensor beam (measured empty). */
    bool is_free(const Eigen::Vector3i& idx) const;

    /** True if this cell was neither traversed by a beam nor hit by one. */
    bool is_unknown(const Eigen::Vector3i& idx) const;

    /**
     * UNKNOWN cells that touch a FREE cell -- i.e. the surface where measured space ends.
     * This is the barrier corridors must not expand through. Returned as cell centres in
     * metres, ready to feed the SFC builder.
     */
    const std::vector<Eigen::Vector3d>& get_unknown_frontier() const {
        // Computed lazily: it costs roughly as much as the rest of the scan update put
        // together, and nothing on the hot path needs it yet.
        if (frontier_dirty_) {
            compute_unknown_frontier();
            frontier_dirty_ = false;
        }
        return unknown_frontier_;
    }

    /**
     * Whether get_obstacles_in_radius() also returns the FREE/UNKNOWN frontier, making unmeasured
     * space act as a solid barrier for SFC construction. OFF by default -- see the long note at
     * the frontier block in get_obstacles_in_radius() for the measurements behind that choice.
     */
    void set_frontier_blocks_corridors(bool on) { frontier_blocks_corridors_ = on; }
    bool frontier_blocks_corridors() const { return frontier_blocks_corridors_; }

    const std::unordered_set<Eigen::Vector3i, Vector3iHash>& get_free() const { return free_voxels_; }
    std::size_t num_free() const { return free_voxels_.size(); }

    /**
     * Returns true if the given voxel index is in the inflated occupied set.
     */
    bool is_occupied_inflated(const Eigen::Vector3i& idx) const;

    /**
     * Returns true if the given voxel index is in the raw (non-inflated) occupied set.
     */
    bool is_occupied_raw(const Eigen::Vector3i& idx) const;

    /**
     * O(log N) query: Distance to the nearest inflated obstacle voxel center.
     * Returns infinity if grid is empty.
     */
    double nearest_obstacle_distance(const Eigen::Vector3d& pt) const;

    /**
     * O(log N) query: Returns true if there is an inflated obstacle voxel 
     * within the given radius of pt.
     */
    bool is_occupied_radius(const Eigen::Vector3d& pt, double radius) const;

    /**
     * O(log N) query: Returns all inflated obstacle centers within radius.
     */
    std::vector<Eigen::Vector3d> get_obstacles_in_radius(const Eigen::Vector3d& pt, double radius) const;

    // Direct access to the underlying sets (for A* planner compatibility)
    const std::unordered_set<Eigen::Vector3i, Vector3iHash>& get_occupied_inflated() const { return occupied_voxels_inflated_; }
    const std::unordered_set<Eigen::Vector3i, Vector3iHash>& get_occupied_raw() const { return occupied_voxels_raw_; }
    
    // Virtual obstacles injected from A* shadowing (cleared before next scan)
    void set_virtual_obstacles(const std::vector<Eigen::Vector3d>& obs) { virtual_obstacles_ = obs; }
    void clear_virtual_obstacles() { virtual_obstacles_.clear(); }
    
    // Continuous inflated bounds for the LoS Smoother (AABB raycaster)
    const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& get_continuous_inflated_bounds() const {
        return continuous_inflated_bounds_;
    }

    double get_resolution() const { return voxel_resolution_; }
    double get_inflation_radius() const { return inflation_radius_; }
    std::size_t num_inflated() const { return occupied_voxels_inflated_.size(); }
    std::size_t num_raw() const { return occupied_voxels_raw_.size(); }

private:
    void rebuild_kdtree();

    /**
     * Marks every cell a beam passes through as FREE, stopping before the return cell, and
     * CLEARS any occupancy it finds on the way (see set_accumulate).
     */
    void raycast_mark_free(const Eigen::Vector3d& origin, const Eigen::Vector3d& endpoint);

    /** Drops accumulated cells outside map_window_radius_ of the sensor. */
    void prune_to_window(const Eigen::Vector3d& sensor_origin);

    /** Rebuilds the inflated set, continuous bounds and KD-tree from occupied_voxels_raw_. */
    void rebuild_derived_from_occupied();

    /** Recomputes unknown_frontier_ from the current free/occupied sets. */
    void compute_unknown_frontier() const;

    inline Eigen::Vector3i to_index(const Eigen::Vector3d& p) const {
        return Eigen::Vector3i(static_cast<int>(std::floor(p.x() / voxel_resolution_)),
                               static_cast<int>(std::floor(p.y() / voxel_resolution_)),
                               static_cast<int>(std::floor(p.z() / voxel_resolution_)));
    }
    inline Eigen::Vector3d to_center(const Eigen::Vector3i& idx) const {
        return idx.cast<double>() * voxel_resolution_
               + Eigen::Vector3d::Constant(voxel_resolution_ / 2.0);
    }

    double voxel_resolution_;
    double inflation_radius_;

    // Note these are deliberately NOT a single tri-state map. Inflation breaks the clean
    // partition: a cell can be genuinely FREE (a beam went through it) while still lying
    // inside the drone-radius halo of a nearby return, which makes it untraversable.
    // Occupancy and freedom therefore have to be tracked independently.
    std::unordered_set<Eigen::Vector3i, Vector3iHash> occupied_voxels_inflated_;
    std::unordered_set<Eigen::Vector3i, Vector3iHash> occupied_voxels_raw_;
    bool frontier_blocks_corridors_ = false;
    std::unordered_set<Eigen::Vector3i, Vector3iHash> free_voxels_;
    mutable std::vector<Eigen::Vector3d> unknown_frontier_;
    mutable bool frontier_dirty_ = true;

    bool accumulate_ = true;
    double map_window_radius_ = 25.0;
    double frontier_radius_ = 18.0;
    Eigen::Vector3d last_sensor_origin_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d frontier_focus_ = Eigen::Vector3d::Zero();
    mutable bool has_frontier_focus_ = false;
    // Cells whose FREE/OCCUPIED status actually changed in the last scan. Only these, and
    // their immediate neighbours, can have changed frontier membership.
    std::vector<Eigen::Vector3i> changed_cells_;
    mutable bool frontier_valid_ = false;
    mutable std::unordered_set<Eigen::Vector3i, Vector3iHash> frontier_set_;
    int scans_since_prune_ = 0;
    static constexpr int kScansPerPrune = 10;   // amortise the O(N) window prune
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> continuous_inflated_bounds_;

    std::vector<Eigen::Vector3d> inflated_centers_;
    std::unique_ptr<Vector3dAdaptor> kd_adaptor_;
    std::unique_ptr<KDTreeType> kdtree_;

    std::vector<Eigen::Vector3d> virtual_obstacles_;
};

} // namespace mapping
} // namespace trajectory_planner
