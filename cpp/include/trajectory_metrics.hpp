#pragma once
//
// Quality metrics for a planned trajectory, for benchmarking and A/B comparison.
//
// Header-only; depends on Eigen and bspline_eval.hpp.
//
//   auto m = metrics::evaluate(result.control_points, result.knots, result.degree);
//   printf("%.1f m in %.1f s (%.2f m/s avg), jerk_rms %.2f\n",
//          m.path_length, m.duration, m.mean_speed, m.jerk_rms);
//
// A NOTE ON COMPARING NUMBERS
// --------------------------
// jerk_rms is included because SANDO's benchmark tooling reports a column of that name, so
// it is the one metric that can be lined up directly. Definitions still have to match to be
// meaningful: this computes sqrt(mean(||jerk||^2)) over UNIFORM TIME samples of the spline.
// An implementation sampling uniformly in arc length, or reporting per-axis rather than
// vector norm, will produce a different number for the same trajectory. Check before
// quoting a comparison.

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

#include "bspline_eval.hpp"

namespace trajectory_planner {
namespace metrics {

struct TrajectoryMetrics {
  bool valid = false;

  // --- Geometry / timing -------------------------------------------------------------
  double duration = 0.0;        ///< seconds (knots are unit-spaced, so this is real time)
  double path_length = 0.0;     ///< arc length along the curve, metres
  double straight_line = 0.0;   ///< distance from first to last point, metres
  double tortuosity = 0.0;      ///< path_length / straight_line; 1.0 is a straight line

  // --- Speed -------------------------------------------------------------------------
  double mean_speed = 0.0;      ///< path_length / duration, m/s
  double max_speed = 0.0;       ///< peak ||v||, m/s -- compare against v_max
  double max_accel = 0.0;       ///< peak ||a||, m/s^2 -- compare against a_max

  // --- Smoothness / effort -----------------------------------------------------------
  /// sqrt(mean(||jerk||^2)) over uniform time samples. Lower is smoother. The metric most
  /// directly comparable with other planners' published figures.
  double jerk_rms = 0.0;

  /// Integral of ||a||^2 dt. Standard control-effort cost: penalises sustained aggressive
  /// acceleration, and is what distinguishes a trajectory that merely *reaches* the goal
  /// from one that gets there without fighting itself.
  double control_effort = 0.0;

  /// Integral of ||a - g|| dt, i.e. specific thrust integrated over time (g is NED gravity,
  /// +9.81 on z). Proportional to total impulse the vehicle must produce, so it is a
  /// reasonable energy proxy that needs no vehicle model beyond hover thrust. NOT actual
  /// joules -- multiply by mass for impulse, and note real electrical power is closer to
  /// thrust^1.5, so this understates the cost of aggressive segments.
  double thrust_integral = 0.0;
};

/**
 * Compute metrics by sampling the spline uniformly in time.
 *
 * @param samples number of samples; 400 is plenty for a few-second trajectory, and the
 *                integrals converge well before that.
 */
inline TrajectoryMetrics evaluate(const Eigen::VectorXd& control_points,
                                  const Eigen::VectorXd& knots,
                                  int degree,
                                  int samples = 400) {
  TrajectoryMetrics m;
  if (!bspline::is_evaluable(control_points, knots, degree) || samples < 2) return m;

  bspline::TrajectorySampler traj(control_points, knots, degree);
  if (!traj.valid()) return m;

  // Jerk is the third derivative, so it needs degree >= 3 to exist at all.
  Eigen::VectorXd jerk_cps, jerk_knots;
  int jerk_degree = degree - 3;
  if (jerk_degree >= 0) {
    Eigen::VectorXd v = bspline::derivative_control_points(control_points, knots, degree);
    Eigen::VectorXd vk = bspline::derivative_knots(knots);
    Eigen::VectorXd a = bspline::derivative_control_points(v, vk, degree - 1);
    Eigen::VectorXd ak = bspline::derivative_knots(vk);
    jerk_cps = bspline::derivative_control_points(a, ak, degree - 2);
    jerk_knots = bspline::derivative_knots(ak);
  }

  const double t0 = traj.start_time();
  const double t1 = traj.end_time();
  m.duration = t1 - t0;
  if (m.duration <= 0.0) return m;

  const double dt = m.duration / (samples - 1);
  const Eigen::Vector3d gravity_ned(0.0, 0.0, 9.81);

  Eigen::Vector3d prev = traj.position(t0);
  const Eigen::Vector3d first = prev;
  Eigen::Vector3d last = prev;
  double jerk_sq_sum = 0.0;

  for (int i = 0; i < samples; ++i) {
    const double t = t0 + i * dt;
    const Eigen::Vector3d p = traj.position(t);
    const Eigen::Vector3d v = traj.velocity(t);
    const Eigen::Vector3d a = traj.acceleration(t);

    if (i > 0) m.path_length += (p - prev).norm();
    prev = p;
    last = p;

    m.max_speed = std::max(m.max_speed, v.norm());
    m.max_accel = std::max(m.max_accel, a.norm());

    // Trapezoidal weight: half at the endpoints.
    const double w = (i == 0 || i == samples - 1) ? 0.5 * dt : dt;
    m.control_effort  += w * a.squaredNorm();
    m.thrust_integral += w * (a - gravity_ned).norm();

    if (jerk_degree >= 0 && jerk_cps.size() > 0) {
      jerk_sq_sum += bspline::evaluate(t, jerk_cps, jerk_knots, jerk_degree).squaredNorm();
    }
  }

  m.jerk_rms = (jerk_degree >= 0 && jerk_cps.size() > 0)
                   ? std::sqrt(jerk_sq_sum / samples)
                   : 0.0;

  m.straight_line = (last - first).norm();
  m.tortuosity = (m.straight_line > 1e-6) ? m.path_length / m.straight_line : 0.0;
  m.mean_speed = m.path_length / m.duration;
  m.valid = true;
  return m;
}

/**
 * Running min / max / mean over a stream of values.
 *
 * For planning time the distribution matters more than any single number: a 10 Hz loop is
 * broken by the tail, not the average. Feed it total_time_ms each replan.
 */
class RunningStats {
 public:
  void add(double x) {
    if (n_ == 0 || x < min_) min_ = x;
    if (n_ == 0 || x > max_) max_ = x;
    ++n_;
    sum_ += x;
  }
  std::size_t count() const { return n_; }
  double min() const { return n_ ? min_ : 0.0; }
  double max() const { return n_ ? max_ : 0.0; }
  double mean() const { return n_ ? sum_ / static_cast<double>(n_) : 0.0; }
  void reset() { n_ = 0; sum_ = 0.0; min_ = 0.0; max_ = 0.0; }

 private:
  std::size_t n_ = 0;
  double sum_ = 0.0, min_ = 0.0, max_ = 0.0;
};

}  // namespace metrics
}  // namespace trajectory_planner
