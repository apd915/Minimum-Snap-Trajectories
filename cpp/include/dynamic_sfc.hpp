#pragma once

#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <memory>
#include "static_sfc.hpp"
#include "voxel_grid.hpp"

namespace trajectory_planner {
namespace sfc {

// ==========================================
// StandaloneWaypointsSFC (Port of dynamic_sfc.py)
// ==========================================
struct WaypointData {
    Eigen::Vector3d position;
    int parent;
    double cost;
    bool connectsToGoal;
};

class StandaloneWaypointsSFC {
public:
    explicit StandaloneWaypointsSFC(int numDimensions = 3)
        : numDimensions_(numDimensions) {}

    void add(const Eigen::Vector3d& position, int parent, double cost, bool connectsToGoal) {
        waypoints_.push_back({position, parent, cost, connectsToGoal});
    }

    void addSFC(const SFCResult& sfc) {
        corridors_.push_back(sfc);
    }

    const std::vector<WaypointData>& getAllWaypoints() const { return waypoints_; }
    const std::vector<SFCResult>& getAllFlightCorridors() const { return corridors_; }

    const Eigen::Vector3d& getPosition(int index) const { return waypoints_[index].position; }
    int getNumNodes() const { return static_cast<int>(waypoints_.size()); }

private:
    int numDimensions_;
    std::vector<WaypointData> waypoints_;
    std::vector<SFCResult> corridors_;
};

// ==========================================
// SpatialScanner (Port of SpatialScannerKD)
// Uses brute-force for now; swap to nanoflann later.
// ==========================================
class SpatialScanner {
public:
    explicit SpatialScanner(std::shared_ptr<mapping::SparseVoxelGrid> grid);

    /**
     * Vectorized Discrete Cylinder Search.
     * Returns all obstacle points inside a cylinder defined by pA→pB
     * with width W and extensions ext_start/ext_end.
     */
    std::vector<Eigen::Vector3d> get_broad_phase_obstacles(
        const Eigen::Vector3d& pA, const Eigen::Vector3d& pB,
        double W, double ext_start, double ext_end) const;

private:
    std::shared_ptr<mapping::SparseVoxelGrid> grid_;
};

// ==========================================
// AsymmetricBoxBuilder (Port of dynamic_sfc.py)
// ==========================================
class AsymmetricBoxBuilder {
public:
    AsymmetricBoxBuilder(double drone_physical_radius, double max_cp_drift, double voxel_resolution = 1.0);

    /**
     * Calculates local axes and uses True Segment Distance Sorting and
     * the Separating Axis Theorem (SAT) to shrink all 6 planes.
     * Returns (ux, uy, uz, bounds[6]).
     */
    std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d, Eigen::Matrix<double, 6, 1>>
    build_bounds(const Eigen::Vector3d& pA, const Eigen::Vector3d& pB,
                 const std::vector<Eigen::Vector3d>& obs_points,
                 double ext_start, double ext_end) const;

    /** Largest lateral half-extent any single face of the box can start at. */
    double max_lateral_extent() const { return 2.0 * max_drift_; }

private:
    double drone_radius_;
    double voxel_resolution_;
    double max_drift_;
};

// ==========================================
// OBBAdapters (Port of dynamic_sfc.py)
// ==========================================
class OBBAdapters {
public:
    /**
     * Translates local frame vectors into OSQP linear inequality constraints (Ax <= b).
     */
    static void get_osqp_matrices(
        const Eigen::Vector3d& pA,
        const Eigen::Vector3d& ux, const Eigen::Vector3d& uy, const Eigen::Vector3d& uz,
        const Eigen::Matrix<double, 6, 1>& bounds,
        Eigen::MatrixXd& A_out, Eigen::VectorXd& b_out);
};

// ==========================================
// AsymmetricSFCManager (Port of dynamic_sfc.py)
// ==========================================
class AsymmetricSFCManager {
public:
    AsymmetricSFCManager(std::shared_ptr<mapping::SparseVoxelGrid> grid,
                         double drone_physical_radius,
                         double max_cp_drift,
                         double voxel_resolution = 1.0);

    /**
     * Full pipeline: Scanner → Builder → OBB Adapter → SFCResult.
     */
    SFCResult generate_sfc(const Eigen::Vector3d& pA, const Eigen::Vector3d& pB,
                           double W, double ext_start, double ext_end) const;

private:
    SpatialScanner scanner_;
    AsymmetricBoxBuilder builder_;
};

} // namespace sfc
} // namespace trajectory_planner
