#include "min_snap_clamped.hpp"
#include <stdexcept>

using namespace std;
using namespace Eigen;

// ==========================================
// PUBLIC API
// ==========================================

MinSnapEvalClamped::MinSnapEvalClamped(int16_t num_segments, int8_t degree) {
    // 1. Intuitive Safety Checks
    if (degree < 4) {
        throw invalid_argument("Minimum Snap requires a polynomial of at least degree 4.");
    }
    if (num_segments < 3) {
        throw invalid_argument("To satisfy 6 physical constraints, you need at least 3 flight segments.");
    }
    
    degree_ = degree;
    update_segments(num_segments);
}

void MinSnapEvalClamped::update_segments(int16_t new_num_segments) {
    M_ = new_num_segments;
    num_control_points_ = M_ + degree_;
    double start_time = 0.0;
    
    knots_ = create_clamped_knot_points(num_control_points_, degree_, start_time);
    
    // Trigger the heavy math (We will write this next!)
    calculate_Q();
}