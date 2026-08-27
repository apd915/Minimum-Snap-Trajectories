#include "voxel_grid.hpp"
#include <cstdlib>
#include <cmath>
#include <limits>
#include <iostream>

namespace trajectory_planner {
namespace mapping {

SparseVoxelGrid::SparseVoxelGrid(double resolution, double inflation_radius)
    : voxel_resolution_(resolution), inflation_radius_(inflation_radius),
      kd_adaptor_(nullptr), kdtree_(nullptr) {}

void SparseVoxelGrid::update_from_scan(const Eigen::Vector3d& sensor_origin,
                                       const std::vector<Eigen::Vector3d>& hits) {
    last_sensor_origin_ = sensor_origin;
    changed_cells_.clear();
    if (!accumulate_) {
        free_voxels_.clear();
        occupied_voxels_raw_.clear();
        frontier_valid_ = false;
    }

    // 1. Sweep the beams. Each traversed cell is measured-empty, so it becomes FREE and any
    //    occupancy recorded there previously is cleared -- that is what keeps an
    //    accumulated map from filling up with ghosts of things that have moved.
    for (const auto& hit : hits) {
        raycast_mark_free(sensor_origin, hit);
    }

    // 2. Returns are OCCUPIED, and win over freedom: a beam grazing the corner of a cell
    //    where another beam terminated must not clear it.
    for (const auto& hit : hits) {
        const Eigen::Vector3i idx = to_index(hit);
        if (free_voxels_.erase(idx)) changed_cells_.push_back(idx);
        if (occupied_voxels_raw_.insert(idx).second) changed_cells_.push_back(idx);
    }

    // 3. Bound memory and query cost. Pruning walks the whole map, so amortise it rather
    //    than paying it inside every 10 Hz callback.
    if (accumulate_ && ++scans_since_prune_ >= kScansPerPrune) {
        prune_to_window(sensor_origin);
        scans_since_prune_ = 0;
        frontier_valid_ = false;   // wholesale removal, incremental update cannot cover it
    }

    // 4. Derived structures (inflated set / continuous bounds / KD-tree) follow occupancy.
    rebuild_derived_from_occupied();

    // 5. Frontier is computed lazily on demand -- see get_unknown_frontier().
    frontier_dirty_ = true;
}

void SparseVoxelGrid::prune_to_window(const Eigen::Vector3d& sensor_origin) {
    const double r2 = map_window_radius_ * map_window_radius_;
    auto out_of_window = [&](const Eigen::Vector3i& idx) {
        return (to_center(idx) - sensor_origin).squaredNorm() > r2;
    };
    for (auto it = free_voxels_.begin(); it != free_voxels_.end();) {
        it = out_of_window(*it) ? free_voxels_.erase(it) : std::next(it);
    }
    for (auto it = occupied_voxels_raw_.begin(); it != occupied_voxels_raw_.end();) {
        it = out_of_window(*it) ? occupied_voxels_raw_.erase(it) : std::next(it);
    }
}

void SparseVoxelGrid::rebuild_derived_from_occupied() {
    occupied_voxels_inflated_.clear();
    continuous_inflated_bounds_.clear();
    continuous_inflated_bounds_.reserve(occupied_voxels_raw_.size());

    const double half = voxel_resolution_ / 2.0;
    for (const auto& idx : occupied_voxels_raw_) {
        const Eigen::Vector3d c = to_center(idx);
        const Eigen::Vector3d lo = c - Eigen::Vector3d::Constant(half + inflation_radius_);
        const Eigen::Vector3d hi = c + Eigen::Vector3d::Constant(half + inflation_radius_);
        continuous_inflated_bounds_.emplace_back(lo, hi);

        const Eigen::Vector3i a = to_index(lo);
        const Eigen::Vector3i b = to_index(hi);
        for (int x = a.x(); x <= b.x(); ++x)
            for (int y = a.y(); y <= b.y(); ++y)
                for (int z = a.z(); z <= b.z(); ++z)
                    occupied_voxels_inflated_.insert(Eigen::Vector3i(x, y, z));
    }

    rebuild_kdtree();
}

void SparseVoxelGrid::raycast_mark_free(const Eigen::Vector3d& origin,
                                        const Eigen::Vector3d& endpoint) {
    Eigen::Vector3d d = endpoint - origin;
    const double len = d.norm();
    if (len < 1e-9) return;
    d /= len;

    Eigen::Vector3i cur = to_index(origin);
    const Eigen::Vector3i end = to_index(endpoint);
    if (cur == end) return;

    // Amanatides & Woo: tMax[i] is the ray parameter (in metres, since d is normalised) at
    // which the ray crosses the next voxel boundary on axis i; tDelta[i] is the spacing
    // between successive crossings on that axis.
    int step[3];
    double tMax[3], tDelta[3];
    const double kInf = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 3; ++i) {
        if (d[i] > 0.0) {
            step[i] = 1;
            tMax[i] = ((cur[i] + 1) * voxel_resolution_ - origin[i]) / d[i];
            tDelta[i] = voxel_resolution_ / d[i];
        } else if (d[i] < 0.0) {
            step[i] = -1;
            tMax[i] = (cur[i] * voxel_resolution_ - origin[i]) / d[i];
            tDelta[i] = -voxel_resolution_ / d[i];
        } else {
            step[i] = 0;
            tMax[i] = kInf;
            tDelta[i] = kInf;
        }
    }

    // Bound the walk: a malformed ray must never spin here inside a 10 Hz callback.
    const int max_steps = static_cast<int>(len / voxel_resolution_) + 4;
    for (int n = 0; n < max_steps && cur != end; ++n) {
        // Only a cell that was NOT already free represents new information. With an
        // accumulated map the overwhelming majority of swept cells are re-confirmations, so
        // tracking just the genuine transitions is what makes the incremental frontier
        // update cheap -- a few hundred cells per scan instead of the whole free set.
        if (free_voxels_.insert(cur).second) {
            changed_cells_.push_back(cur);
        }
        if (occupied_voxels_raw_.erase(cur)) {   // measured through it -> it is not solid
            changed_cells_.push_back(cur);
        }

        int axis = (tMax[0] < tMax[1]) ? ((tMax[0] < tMax[2]) ? 0 : 2)
                                       : ((tMax[1] < tMax[2]) ? 1 : 2);
        if (tMax[axis] > len) break;   // stepped past the return point
        cur[axis] += step[axis];
        tMax[axis] += tDelta[axis];
    }
}

void SparseVoxelGrid::compute_unknown_frontier() const {
    static const Eigen::Vector3i kNeighbors[6] = {
        Eigen::Vector3i( 1, 0, 0), Eigen::Vector3i(-1, 0, 0),
        Eigen::Vector3i( 0, 1, 0), Eigen::Vector3i( 0,-1, 0),
        Eigen::Vector3i( 0, 0, 1), Eigen::Vector3i( 0, 0,-1)
    };

    // A cell is on the frontier if it is UNKNOWN, touches measured-free space, and is not a
    // "pinhole". Pinholes exist because the beam pattern is angular: past ~9.5 m (3 deg
    // beams, 0.5 m voxels) neighbouring beams are more than a voxel apart, leaving unsampled
    // cells BETWEEN them. Those are not a boundary of knowledge -- the space really was swept
    // -- so a candidate ringed by free cells is rejected.
    constexpr int kMaxFreeNeighborsForFrontier = 2;
    const double fr2 = frontier_radius_ * frontier_radius_;
    const Eigen::Vector3d focus = has_frontier_focus_ ? frontier_focus_ : last_sensor_origin_;

    auto in_region = [&](const Eigen::Vector3i& idx) {
        return frontier_radius_ <= 0.0 || (to_center(idx) - focus).squaredNorm() <= fr2;
    };
    auto classify = [&](const Eigen::Vector3i& idx) {
        // Returns true if idx belongs on the frontier.
        if (free_voxels_.count(idx) || occupied_voxels_raw_.count(idx)) return false;
        if (!in_region(idx)) return false;
        int free_neighbors = 0;
        for (const auto& n : kNeighbors) if (free_voxels_.count(idx + n)) ++free_neighbors;
        return free_neighbors > 0 && free_neighbors <= kMaxFreeNeighborsForFrontier;
    };

    // --- Incremental path -------------------------------------------------------------
    // Only cells that actually changed state this scan, and their immediate neighbours, can
    // have changed frontier membership. With an accumulated map that is a few hundred cells
    // rather than the entire free set, which is the difference between ~0.5 ms and ~10 ms.
    if (frontier_valid_ && !changed_cells_.empty()) {
        for (const auto& c : changed_cells_) {
            frontier_set_.erase(c);
            if (classify(c)) frontier_set_.insert(c);
            for (const auto& n : kNeighbors) {
                const Eigen::Vector3i nb = c + n;
                frontier_set_.erase(nb);
                if (classify(nb)) frontier_set_.insert(nb);
            }
        }
    } else {
        // --- Full rebuild (first call, or after a prune removed cells wholesale) -------
        // Counts free neighbours DURING discovery -- each adjacent free cell contributes one
        // increment -- so a candidate is classified without ever re-probing its neighbourhood.
        // (Using classify() here instead costs ~8x more probes per free cell.)
        std::unordered_map<Eigen::Vector3i, int, Vector3iHash> counts;
        counts.reserve(free_voxels_.size() / 4);
        for (const auto& f : free_voxels_) {
            if (!in_region(f)) continue;
            for (const auto& n : kNeighbors) {
                const Eigen::Vector3i nb = f + n;
                if (free_voxels_.count(nb)) continue;
                if (occupied_voxels_raw_.count(nb)) continue;
                ++counts[nb];
            }
        }
        frontier_set_.clear();
        frontier_set_.reserve(counts.size());
        for (const auto& e : counts) {
            if (e.second > kMaxFreeNeighborsForFrontier) continue;   // pinhole
            frontier_set_.insert(e.first);
        }
        frontier_valid_ = true;
    }

    unknown_frontier_.clear();
    unknown_frontier_.reserve(frontier_set_.size());
    for (const auto& idx : frontier_set_) {
        if (!in_region(idx)) continue;   // focus may have moved since insertion
        unknown_frontier_.push_back(to_center(idx));
    }
}

bool SparseVoxelGrid::is_free(const Eigen::Vector3i& idx) const {
    return free_voxels_.count(idx) > 0;
}

bool SparseVoxelGrid::is_unknown(const Eigen::Vector3i& idx) const {
    return free_voxels_.count(idx) == 0 && occupied_voxels_raw_.count(idx) == 0;
}

void SparseVoxelGrid::update_from_obstacles(const std::vector<ObstacleBox>& obstacles) {
    // Clear old data
    occupied_voxels_inflated_.clear();
    occupied_voxels_raw_.clear();
    continuous_inflated_bounds_.clear();
    inflated_centers_.clear();

    for (const auto& obstacle : obstacles) {
        // Inflated bounds
        Eigen::Vector3d inflated_min = obstacle.min.array() - inflation_radius_;
        Eigen::Vector3d inflated_max = obstacle.max.array() + inflation_radius_;

        // Save the continuous inflated bounds for the LoS smoother
        continuous_inflated_bounds_.emplace_back(inflated_min, inflated_max);

        // Convert to voxel indices (floor division)
        Eigen::Vector3i inf_min_idx(
            static_cast<int>(std::floor(inflated_min.x() / voxel_resolution_)),
            static_cast<int>(std::floor(inflated_min.y() / voxel_resolution_)),
            static_cast<int>(std::floor(inflated_min.z() / voxel_resolution_))
        );
        Eigen::Vector3i inf_max_idx(
            static_cast<int>(std::floor(inflated_max.x() / voxel_resolution_)),
            static_cast<int>(std::floor(inflated_max.y() / voxel_resolution_)),
            static_cast<int>(std::floor(inflated_max.z() / voxel_resolution_))
        );

        // Raw (non-inflated) bounds
        Eigen::Vector3i raw_min_idx(
            static_cast<int>(std::floor(obstacle.min.x() / voxel_resolution_)),
            static_cast<int>(std::floor(obstacle.min.y() / voxel_resolution_)),
            static_cast<int>(std::floor(obstacle.min.z() / voxel_resolution_))
        );
        Eigen::Vector3i raw_max_idx(
            static_cast<int>(std::floor(obstacle.max.x() / voxel_resolution_)),
            static_cast<int>(std::floor(obstacle.max.y() / voxel_resolution_)),
            static_cast<int>(std::floor(obstacle.max.z() / voxel_resolution_))
        );

        // Fill inflated voxels
        for (int x = inf_min_idx.x(); x <= inf_max_idx.x(); ++x) {
            for (int y = inf_min_idx.y(); y <= inf_max_idx.y(); ++y) {
                for (int z = inf_min_idx.z(); z <= inf_max_idx.z(); ++z) {
                    occupied_voxels_inflated_.insert(Eigen::Vector3i(x, y, z));
                }
            }
        }

        // Fill raw voxels
        for (int x = raw_min_idx.x(); x <= raw_max_idx.x(); ++x) {
            for (int y = raw_min_idx.y(); y <= raw_max_idx.y(); ++y) {
                for (int z = raw_min_idx.z(); z <= raw_max_idx.z(); ++z) {
                    occupied_voxels_raw_.insert(Eigen::Vector3i(x, y, z));
                }
            }
        }
    }

    rebuild_kdtree();
}

void SparseVoxelGrid::rebuild_kdtree() {
    inflated_centers_.clear();
    inflated_centers_.reserve(occupied_voxels_inflated_.size());
    for (const auto& idx : occupied_voxels_inflated_) {
        Eigen::Vector3d pt = idx.cast<double>() * voxel_resolution_
                             + Eigen::Vector3d::Constant(voxel_resolution_ / 2.0);
        inflated_centers_.push_back(pt);
    }

    if (!inflated_centers_.empty()) {
        kd_adaptor_ = std::make_unique<Vector3dAdaptor>(inflated_centers_);
        kdtree_ = std::make_unique<KDTreeType>(3, *kd_adaptor_, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
        kdtree_->buildIndex();
    } else {
        kdtree_.reset();
        kd_adaptor_.reset();
    }
}

bool SparseVoxelGrid::is_occupied_inflated(const Eigen::Vector3i& idx) const {
    return occupied_voxels_inflated_.count(idx) > 0;
}

bool SparseVoxelGrid::is_occupied_raw(const Eigen::Vector3i& idx) const {
    return occupied_voxels_raw_.count(idx) > 0;
}

double SparseVoxelGrid::nearest_obstacle_distance(const Eigen::Vector3d& pt) const {
    if (!kdtree_ || inflated_centers_.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    
    double query_pt[3] = {pt.x(), pt.y(), pt.z()};
    unsigned int out_index = 0;
    double out_dist_sq = 0.0;
    
    kdtree_->knnSearch(&query_pt[0], 1, &out_index, &out_dist_sq);
    return std::sqrt(out_dist_sq);
}

bool SparseVoxelGrid::is_occupied_radius(const Eigen::Vector3d& pt, double radius) const {
    if (!kdtree_ || inflated_centers_.empty()) {
        return false;
    }
    
    double query_pt[3] = {pt.x(), pt.y(), pt.z()};
    double search_radius = radius * radius; // nanoflann uses squared radius
    std::vector<nanoflann::ResultItem<uint32_t, double>> ret_matches;
    
    nanoflann::SearchParameters params;
    size_t nMatches = kdtree_->radiusSearch(&query_pt[0], search_radius, ret_matches, params);
    
    return nMatches > 0;
}

std::vector<Eigen::Vector3d> SparseVoxelGrid::get_obstacles_in_radius(const Eigen::Vector3d& pt, double radius) const {
    std::vector<Eigen::Vector3d> result;
    double search_radius_sq = radius * radius;
    
    // 1. Physical Obstacles
    if (kdtree_ && !inflated_centers_.empty()) {
        double query_pt[3] = {pt.x(), pt.y(), pt.z()};
        std::vector<nanoflann::ResultItem<uint32_t, double>> ret_matches;
        nanoflann::SearchParameters params;
        
        kdtree_->radiusSearch(&query_pt[0], search_radius_sq, ret_matches, params);
        
        result.reserve(ret_matches.size() + virtual_obstacles_.size());
        for (const auto& match : ret_matches) {
            result.push_back(inflated_centers_[match.first]);
        }
    } else {
        result.reserve(virtual_obstacles_.size());
    }

    // 2. Unknown-space frontier -- OPT-IN, see set_frontier_blocks_corridors().
    //
    // The idea was that a corridor may only claim space actually measured empty, so the
    // FREE/UNKNOWN boundary acts as a barrier exactly like a physical obstacle. In practice
    // that is far too aggressive and it is OFF by default. Two reasons.
    //
    // It is inconsistent with the rest of the pipeline. A* is free to route THROUGH unknown
    // space -- it only pays unknown_traversal_cost -- so the search commits to a path and the
    // corridor builder then walls that same path in. The symptom is specific: the corridor
    // around a waypoint the sensor has not swept yet collapses to ZERO width while every
    // corridor behind it comes out full size. Measured on a straight 6.4 m leg, bounds
    // (+y -y +z -z) were 2.010 2.010 0.000 0.000 with the frontier included and
    // 3.394 0.625 2.010 2.010 with it excluded, on a bit-identical path.
    //
    // And the frontier is not the clean surface it sounds like. Beam divergence means that at
    // range the swept-free region is a fan of thin lines with unsampled cells between them.
    // compute_unknown_frontier() rejects those "pinholes", but only by counting FREE FACE
    // neighbours -- and beam lines run diagonally, so a gap cell scores 1-2 by that metric and
    // survives. Measured 679 surviving frontier cells within 0.75 m of one corridor axis, of
    // which the modal cell had NINE free neighbours in its 26-neighbourhood: embedded in
    // measured-free space, not on any boundary of knowledge.
    //
    // Safety against unknown space is handled where it belongs, and by mechanisms that do not
    // depend on this one: A* pays unknown_traversal_cost to enter it, replan_loop re-checks the
    // committed setpoints against the freshest map at 10 Hz, and the terminal-velocity logic in
    // trajectory_planner.cpp sizes end-of-plan speed so the vehicle can brake inside the space
    // it can actually see. For reference, SANDO builds its decomposition obstacle set from
    // OCCUPIED voxels (getVecOccupied) and treats unknown-boundary inflation as a separately
    // gated option -- it does not wall corridors off with unknown space either.
    if (frontier_blocks_corridors_) {
        for (const auto& f : get_unknown_frontier()) {
            if ((f - pt).squaredNorm() <= search_radius_sq) {
                result.push_back(f);
            }
        }
    }

    // 3. Virtual Obstacles (manual injection point; unused by the default pipeline)
    for (const auto& v_obs : virtual_obstacles_) {
        if ((v_obs - pt).squaredNorm() <= search_radius_sq) {
            result.push_back(v_obs);
        }
    }

    return result;
}

} // namespace mapping
} // namespace trajectory_planner
