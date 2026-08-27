#pragma once
//
// Evaluation of the B-spline that TrajectoryPlanner produces.
//
// plan_mission() returns control points and a knot vector, not a curve. Turning those into
// position/velocity/acceleration setpoints is de Boor's algorithm plus derivative control
// points -- short, but fiddly enough that every consumer re-deriving it independently is a
// good way to get subtly different bugs. This header is that code, lifted from a flying
// integration.
//
// Header-only; depends on Eigen and nothing else. No ROS, no linking.
//
//   using namespace trajectory_planner;
//   auto res = planner.plan_mission(...);
//   if (res.success) {
//       bspline::TrajectorySampler traj(res.control_points, res.knots, res.degree);
//       for (double t = traj.start_time(); t <= traj.end_time(); t += 0.02) {
//           Eigen::Vector3d p = traj.position(t);
//           Eigen::Vector3d v = traj.velocity(t);
//           Eigen::Vector3d a = traj.acceleration(t);
//       }
//   }
//
// TIME UNITS: the planner emits a unit-spaced knot vector, so knot time IS seconds and one
// spline segment is one second. start_time() is normally 0 and end_time() is the flight
// duration. Do not rescale.

#include <Eigen/Dense>
#include <algorithm>
#include <vector>

namespace trajectory_planner {
namespace bspline {

/**
 * True if this (control points, knots, degree) triple describes an evaluable curve.
 *
 * Worth checking once before sampling in a loop. A degree-p spline needs more than p control
 * points and more than p+1 knots; anything less leaves de Boor indexing before the start of
 * the control-point array. The planner can legitimately return short splines, so this is a
 * real case rather than a paranoid one.
 */
inline bool is_evaluable(const Eigen::VectorXd& control_points,
                         const Eigen::VectorXd& knots,
                         int degree) {
  if (degree < 1) return false;
  if (control_points.size() == 0 || knots.size() == 0) return false;
  if (control_points.size() % 3 != 0) return false;
  if (control_points.size() / 3 <= degree) return false;
  if (knots.size() <= degree + 1) return false;
  return true;
}

/** First valid time on the curve (the start of the clamped span). */
inline double start_time(const Eigen::VectorXd& knots, int degree) {
  return knots(degree);
}

/** Last valid time on the curve. */
inline double end_time(const Eigen::VectorXd& knots, int degree) {
  return knots(knots.size() - degree - 1);
}

/**
 * Evaluate the curve at time t via de Boor's algorithm.
 *
 * t is clamped into [start_time, end_time]; sampling past the end returns the endpoint
 * rather than extrapolating, which is what you want when a consumer's timer overshoots.
 * Returns zero for a non-evaluable spline -- check is_evaluable() first if you need to
 * distinguish "at the origin" from "invalid".
 */
inline Eigen::Vector3d evaluate(double t,
                                const Eigen::VectorXd& control_points,
                                const Eigen::VectorXd& knots,
                                int degree) {
  if (!is_evaluable(control_points, knots, degree)) return Eigen::Vector3d::Zero();

  const double t0 = start_time(knots, degree);
  const double t1 = end_time(knots, degree);
  t = std::max(t0, std::min(t, t1));

  // Locate the knot span containing t.
  int span = degree;
  if (t >= t1) {
    span = static_cast<int>(knots.size()) - degree - 2;
  } else {
    while (span < static_cast<int>(knots.size()) - degree - 1 && t >= knots(span + 1)) {
      ++span;
    }
  }

  // de Boor: repeatedly interpolate the p+1 control points influencing this span.
  std::vector<Eigen::Vector3d> d(degree + 1);
  for (int j = 0; j <= degree; ++j) {
    const int idx = (span - degree + j) * 3;
    d[j] = Eigen::Vector3d(control_points(idx), control_points(idx + 1), control_points(idx + 2));
  }
  for (int r = 1; r <= degree; ++r) {
    for (int j = degree; j >= r; --j) {
      const double denom = knots(span + j + 1 - r) - knots(span - degree + j);
      // Repeated knots give a zero denominator at clamped ends; alpha=0 is the correct limit.
      const double alpha = (denom > 1e-9) ? (t - knots(span - degree + j)) / denom : 0.0;
      d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j];
    }
  }
  return d[degree];
}

/**
 * Control points of the derivative curve.
 *
 * The derivative of a degree-p B-spline is a degree-(p-1) B-spline with one fewer control
 * point, over the knot vector with the first and last entries dropped. Apply once for
 * velocity, twice for acceleration. Returns empty if the input is too short to differentiate.
 */
inline Eigen::VectorXd derivative_control_points(const Eigen::VectorXd& control_points,
                                                 const Eigen::VectorXd& knots,
                                                 int degree) {
  if (control_points.size() < 6) return Eigen::VectorXd();
  const int n = static_cast<int>(control_points.size()) / 3;
  if (knots.size() < n + degree + 1) return Eigen::VectorXd();

  Eigen::VectorXd out((n - 1) * 3);
  for (int i = 0; i < n - 1; ++i) {
    const double denom = knots(i + degree + 1) - knots(i + 1);
    const double factor = (denom > 1e-9) ? (degree / denom) : 0.0;
    for (int k = 0; k < 3; ++k) {
      out(i * 3 + k) = factor * (control_points((i + 1) * 3 + k) - control_points(i * 3 + k));
    }
  }
  return out;
}

/** Knot vector of the derivative curve: the input with its first and last entries removed. */
inline Eigen::VectorXd derivative_knots(const Eigen::VectorXd& knots) {
  if (knots.size() <= 2) return Eigen::VectorXd();
  return knots.segment(1, knots.size() - 2);
}

/**
 * Samples position, velocity and acceleration from one planned trajectory.
 *
 * Builds the two derivative curves once on construction rather than per sample -- a 50 Hz
 * consumer would otherwise redo that work for every setpoint.
 */
class TrajectorySampler {
 public:
  TrajectorySampler() = default;

  TrajectorySampler(const Eigen::VectorXd& control_points,
                    const Eigen::VectorXd& knots,
                    int degree)
      : pos_(control_points), pos_knots_(knots), degree_(degree) {
    valid_ = is_evaluable(pos_, pos_knots_, degree_);
    if (!valid_) return;

    vel_ = derivative_control_points(pos_, pos_knots_, degree_);
    vel_knots_ = derivative_knots(pos_knots_);

    acc_ = derivative_control_points(vel_, vel_knots_, degree_ - 1);
    acc_knots_ = derivative_knots(vel_knots_);
  }

  /** False if the planner returned a spline too short to evaluate. Check before sampling. */
  bool valid() const { return valid_; }

  double start_time() const { return valid_ ? bspline::start_time(pos_knots_, degree_) : 0.0; }
  double end_time()   const { return valid_ ? bspline::end_time(pos_knots_, degree_) : 0.0; }
  double duration()   const { return end_time() - start_time(); }

  Eigen::Vector3d position(double t) const {
    return evaluate(t, pos_, pos_knots_, degree_);
  }

  /** Zero if the spline is too short to differentiate -- which is safe to command. */
  Eigen::Vector3d velocity(double t) const {
    return evaluate(t, vel_, vel_knots_, degree_ - 1);
  }

  Eigen::Vector3d acceleration(double t) const {
    return evaluate(t, acc_, acc_knots_, degree_ - 2);
  }

 private:
  Eigen::VectorXd pos_, pos_knots_;
  Eigen::VectorXd vel_, vel_knots_;
  Eigen::VectorXd acc_, acc_knots_;
  int degree_ = 0;
  bool valid_ = false;
};

}  // namespace bspline
}  // namespace trajectory_planner
