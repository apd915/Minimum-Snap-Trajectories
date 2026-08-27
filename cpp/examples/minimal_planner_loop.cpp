// A complete, runnable planning loop in ~200 lines. No ROS, no middleware.
//
// This is the reference for the four things every integration has to supply, which the
// library deliberately does not:
//
//   1. Feeding the map -- sensor ORIGIN plus returns, in world-frame NED
//   2. Evaluating the spline into setpoints (bspline_eval.hpp)
//   3. Deciding WHEN to re-plan
//   4. Stitching successive plans together so they stay continuous
//
// The vehicle and sensor here are fakes: the vehicle tracks setpoints perfectly, and the
// LiDAR is a raycast against a hardcoded world. Swap those two for real ones and the
// structure is unchanged.
//
// Build (from cpp/):
//   g++ -std=c++17 -O2 examples/minimal_planner_loop.cpp \
//       -Iinclude -I/usr/include/eigen3 -I/usr/local/include \
//       -Lbuild -lmin_snap_engine -L/usr/local/lib -losqp -o /tmp/minimal_loop
//   /tmp/minimal_loop
//
// Or simply `make` -- this is a CMake target and builds to build/minimal_planner_loop.

#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <deque>
#include <memory>
#include <vector>

#include "bspline_eval.hpp"
#include "trajectory_planner.hpp"
#include "voxel_grid.hpp"

using namespace trajectory_planner;

// ===========================================================================================
// Fake world + fake LiDAR. Replace with your sensor driver.
// ===========================================================================================
namespace {

// Two pillars and a wall, in NED (z negative-up, so these span from the ground upward).
bool is_solid(const Eigen::Vector3d& p) {
  if (p.z() >= 0.0) return true;                                        // ground
  // Wall with a gap either side. Kept narrow deliberately: this example demonstrates the
  // integration contract, not the planner's limits. A wall wide enough to force a detour
  // longer than planning_horizon will reliably strand the vehicle -- see the note at the
  // bottom of this file, and use tools/stress_planner.cpp for that kind of scenario.
  if (p.x() >= 14.0 && p.x() <= 15.0 && std::abs(p.y()) <= 2.0 && p.z() >= -6.0) return true;
  const double d1 = std::hypot(p.x() - 6.0, p.y() - 1.5);
  if (d1 <= 0.6 && p.z() >= -6.0) return true;
  const double d2 = std::hypot(p.x() - 9.0, p.y() + 2.0);
  if (d2 <= 0.8 && p.z() >= -6.0) return true;
  return false;
}

/// Livox-Mid-360-ish sweep: 360 deg horizontal, -7..+52 deg vertical, 3 deg steps.
std::vector<Eigen::Vector3d> lidar_scan(const Eigen::Vector3d& origin) {
  constexpr double kMaxRange = 40.0, kStep = 0.25;
  std::vector<Eigen::Vector3d> hits;
  for (int iv = 0; iv < 20; ++iv) {
    const double el = (-7.0 + iv * 3.0) * M_PI / 180.0;
    for (int ih = 0; ih < 120; ++ih) {
      const double az = (-180.0 + ih * 3.0) * M_PI / 180.0;
      const Eigen::Vector3d dir(std::cos(el) * std::cos(az),
                                std::cos(el) * std::sin(az),
                                -std::sin(el));
      for (double s = kStep; s < kMaxRange; s += kStep) {
        const Eigen::Vector3d q = origin + s * dir;
        if (is_solid(q)) { hits.push_back(q); break; }
      }
    }
  }
  return hits;
}

struct Setpoint {
  Eigen::Vector3d position, velocity, acceleration;
};

}  // namespace

// ===========================================================================================
int main() {
  // --- Vehicle / loop constants -------------------------------------------------------
  const double kDroneRadius = 0.49;   // circumscribed radius; corridors clear by this much
  const double kVoxel       = 0.5;
  const double kVMax        = 3.0;
  const double kAMax        = 2.0;
  const double kDispatchDt  = 0.02;   // 50 Hz setpoint output
  const double kReplanDt    = 0.10;   // 10 Hz planning
  const int    kStitchAhead = 10;     // plan from 10 setpoints (200 ms) into the future
  const size_t kMaxBuffered = 150;    // commit at most 3 s of setpoints ahead

  // --- Map ----------------------------------------------------------------------------
  auto grid = std::make_shared<mapping::SparseVoxelGrid>(kVoxel, kDroneRadius);

  // --- Planner configuration ----------------------------------------------------------
  FrontEndConfig cfg;
  cfg.voxel_resolution      = kVoxel;
  cfg.drone_physical_radius = kDroneRadius;
  cfg.map_bounds            = Eigen::Vector3d(100.0, 100.0, 15.0);
  cfg.min_altitude          = kDroneRadius + 0.1;
  cfg.planning_horizon      = 10.0;

  // --- Vehicle state (fake: tracks setpoints exactly) ----------------------------------
  Eigen::Vector3d vehicle_pos(0.0, 0.0, -3.0);   // 3 m altitude
  Eigen::Vector3d vehicle_vel = Eigen::Vector3d::Zero();
  const Eigen::Vector3d goal(20.0, 0.0, -3.0);

  std::deque<Setpoint> buffer;   // pending setpoints, drained at 50 Hz
  double sim_time = 0.0, last_plan_time = -1e9;
  bool last_reached_goal = false;
  int replans = 0, failures = 0;

  std::printf("%-7s %-26s %-9s %s\n", "t[s]", "vehicle position", "buffer", "event");

  // ===========================================================================================
  // MAIN LOOP
  // ===========================================================================================
  for (int tick = 0; tick < 1500; ++tick) {
    sim_time = tick * kDispatchDt;

    // -------------------------------------------------------------------------------------
    // (1) SENSOR UPDATE -- 10 Hz. Pass the beam ORIGIN, not just the returns: that is what
    //     lets the map record which space was measured EMPTY rather than merely unoccupied.
    // -------------------------------------------------------------------------------------
    if (tick % 5 == 0) {
      grid->update_from_scan(vehicle_pos, lidar_scan(vehicle_pos));
    }

    // -------------------------------------------------------------------------------------
    // (3) REPLAN POLICY -- when, not how.
    //
    //     Re-plan at 10 Hz, but SKIP while the committed trajectory is still long, still
    //     collision-free against the newest map, and still fresh. Without the freshness
    //     term the vehicle flies open-loop and ignores space it has just revealed; without
    //     the skip it re-plans every cycle and the trajectory visibly stutters.
    // -------------------------------------------------------------------------------------
    const bool due = (sim_time - last_plan_time) >= kReplanDt;
    if (due) {
      bool skip = false;
      if (buffer.size() >= 50 && last_reached_goal &&
          (sim_time - last_plan_time) < 0.5) {
        skip = true;
        for (size_t i = 0; i < buffer.size(); i += 10) {   // coarse re-validation
          if (grid->is_occupied_radius(buffer[i].position, kDroneRadius)) { skip = false; break; }
        }
      }

      if (!skip) {
        // ---------------------------------------------------------------------------------
        // (4) STITCH POINT -- plan from a point already in the buffer, not from where the
        //     vehicle is now. Handing the planner that point's position AND velocity AND
        //     acceleration as boundary conditions is what makes consecutive plans join
        //     smoothly instead of stepping.
        // ---------------------------------------------------------------------------------
        Eigen::Vector3d start_pos = vehicle_pos, start_vel = vehicle_vel;
        Eigen::Vector3d start_acc = Eigen::Vector3d::Zero();
        int stitch_index = 0;
        if (!buffer.empty()) {
          stitch_index = std::min<int>(static_cast<int>(buffer.size()) - 1, kStitchAhead);
          start_pos = buffer[stitch_index].position;
          start_vel = buffer[stitch_index].velocity;
          start_acc = buffer[stitch_index].acceleration;
        }

        TrajectoryPlanner planner(cfg, grid, kVMax, kAMax);
        auto result = planner.plan_mission(start_pos, goal, start_vel, start_acc);
        ++replans;

        if (result.success) {
          // -------------------------------------------------------------------------------
          // (2) EVALUATE -- turn control points into setpoints. TrajectorySampler builds the
          //     velocity/acceleration curves once, so sampling in this loop is cheap.
          // -------------------------------------------------------------------------------
          bspline::TrajectorySampler traj(result.control_points, result.knots, result.degree);
          if (traj.valid()) {
            // Keep everything up to the stitch point, discard the rest, append the new plan.
            buffer.erase(buffer.begin() + std::min<int>(stitch_index + 1, buffer.size()),
                         buffer.end());
            // Only commit a bounded lookahead. The planner happily returns a 20 s
            // trajectory, but buffering all of it means re-planning against a map that has
            // moved on -- and the tail gets discarded on the next cycle anyway. Committing
            // ~3 s keeps the buffer honest and the collision re-check cheap.
            for (double t = traj.start_time() + kDispatchDt;
                 t <= traj.end_time() && buffer.size() < kMaxBuffered;
                 t += kDispatchDt) {
              buffer.push_back({traj.position(t), traj.velocity(t), traj.acceleration(t)});
            }
            last_reached_goal = result.reached_goal;
            last_plan_time = sim_time;
            if (replans <= 3 || result.reached_goal) {
              std::printf("%-7.2f %-26s %-9zu replan: %zu corridors, %d pts, %.1f ms%s\n",
                          sim_time, "", buffer.size(), result.corridors.size(),
                          result.total_control_points, result.total_time_ms,
                          result.reached_goal ? " (reaches goal)" : "");
            }
          }
        } else {
          // A refusal is normal: goal unreachable, or the state is outside the kinematic
          // envelope. Hold what we have and try again -- do NOT clear the buffer.
          ++failures;
          last_plan_time = sim_time;
        }
      }
    }

    // -------------------------------------------------------------------------------------
    // DISPATCH -- 50 Hz. Pop one setpoint and "fly" it.
    // -------------------------------------------------------------------------------------
    if (!buffer.empty()) {
      const Setpoint sp = buffer.front();
      buffer.pop_front();
      vehicle_pos = sp.position;    // a real vehicle would track this imperfectly
      vehicle_vel = sp.velocity;
    }

    if (tick % 100 == 0) {
      std::printf("%-7.2f (%6.2f %6.2f %6.2f)   %-9zu\n", sim_time,
                  vehicle_pos.x(), vehicle_pos.y(), vehicle_pos.z(), buffer.size());
    }

    if ((vehicle_pos - goal).norm() < 1.0) {
      std::printf("\nReached goal at t = %.2f s  (%d replans, %d refusals)\n",
                  sim_time, replans, failures);
      return 0;
    }
  }

  std::printf("\nStopped at t = %.2f s, %.2f m from goal  (%d replans, %d refusals)\n",
              sim_time, (vehicle_pos - goal).norm(), replans, failures);
  return 0;
}

// ===========================================================================================
// NOTES
//
// * Runs are deterministic. A*'s budget is a node count (FrontEndConfig::astar_max_nodes),
//   not a wall-clock timer, so identical inputs give identical trajectories. That property is
//   worth protecting: when the budget was time-based, this same scenario reached the goal on
//   some runs and stranded the vehicle on others, purely from CPU scheduling jitter.
//
// * Refusals are normal. plan_mission() returns success == false when the goal is
//   unreachable, too close to stop at from the current speed, or the state is outside the
//   kinematic envelope. Hold the existing buffer and retry -- do not clear it.
//
// * The vehicle here tracks setpoints perfectly. A real one does not, and the corridor
//   guarantee only covers the REFERENCE trajectory: it holds for the vehicle only while
//   tracking error stays below drone_physical_radius. Measure that.
//
// * A detour longer than planning_horizon can strand the vehicle: the horizon-projected goal
//   points into the obstacle and the search never sees the way around. Raising the horizon
//   costs solve time super-linearly. This is a known limitation, not a bug in this example.
// ===========================================================================================
