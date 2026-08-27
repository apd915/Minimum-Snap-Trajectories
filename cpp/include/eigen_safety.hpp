#pragma once
//
// Turns Eigen's internal bounds checks into C++ exceptions instead of deleting them.
//
// THE PROBLEM THIS SOLVES
// ----------------------
// Release builds define NDEBUG. Eigen's `eigen_assert` defaults to `assert`, so NDEBUG
// silently removes every bounds check on .block(), .segment(), .col(), .head(), etc. An
// out-of-range block does not fault -- it writes past the end of the matrix, corrupts the
// heap, and the process dies later somewhere unrelated with "double free or corruption".
//
// That is not hypothetical here: get_fast_cascaded_D_matrix() wrote a 4x8 boundary stencil
// into a 3x7 matrix for any trajectory short enough to need only 3 segments. It survived
// indefinitely because ordinary flight never produced a path that short, then started
// killing the node the moment the planning horizon was reduced. Debug builds caught it
// instantly; the flown build could not.
//
// WHAT THIS DOES
// --------------
// Defines eigen_assert to throw std::runtime_error. The checks are a compare-and-branch on
// sizes that are already in registers, so the cost is small (measured: see below), and the
// planner already runs every solve inside a try/catch -- so a dimension bug degrades into a
// logged, recoverable planning failure instead of memory corruption on an aircraft.
//
// Include this BEFORE any Eigen header. The CMake target does that for the whole library via
// a forced include, so individual sources do not need to.
//
// To opt out (e.g. to benchmark the difference):  -DMIN_SNAP_NO_EIGEN_SAFETY

#if !defined(MIN_SNAP_NO_EIGEN_SAFETY)

#ifdef EIGEN_CORE_H
#error "eigen_safety.hpp must be included BEFORE any Eigen header to take effect."
#endif

#include <stdexcept>
#include <string>

#define eigen_assert(condition)                                                        \
  do {                                                                                 \
    if (!(condition)) {                                                                \
      throw std::runtime_error(std::string("Eigen assertion failed: " #condition       \
                                           " at " __FILE__ ":") +                      \
                               std::to_string(__LINE__));                              \
    }                                                                                  \
  } while (0)

#endif  // MIN_SNAP_NO_EIGEN_SAFETY
