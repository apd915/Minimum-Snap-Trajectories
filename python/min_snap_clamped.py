'''
Visual and simple computational 
representation of snap minimization
using clamped B-Splines

TO DO:
- Optimize 4th derivative of position (snap)
- Optimize 2nd derivative of heading (course spline)

SNAP OPTIMIZATION:
- C_p = A_p @ Q_d4_M

HEADING OPTIMIZATION:
- C_psi = A_psi @ Q_d2_M

Q_d4_M = B_d3 @ U_1 @ (I - W_d4_M @ U_2 @ inv(U_2.T@W_d4_M@U_2) @ U_2.T)
Q_d2_M = U_1_psi.T @ (I - W_d2_M @ U_2_psi @ inv(U_2_psi.T@W_d2_M@U_2_psi) @ U_2_psi.T)

Will need to define matrices and how to calculate them
'''

import numpy as np
from numpy.linalg import inv
import time


# ==========================================
# MINIMUM SNAP EVALUATOR (3D POSITION)
# ==========================================

class MinSnapEval:
    '''
    Visual and simple computational 
    representation of snap minimization
    using clamped uniform B-Splines
    '''

    def __init__(self, num_control_points, degree):
        start_time = 0

        self.knots = self._create_clamped_knot_points(num_control_points, degree, start_time)
        
        B_d_3 = self._get_B_d3_matrix(degree)

        S_d4_M, snap_knots = self._get_S_matrix(degree, degree, self.knots, num_control_points)
        W_d4_M = self._get_W_matrix(S_d4_M, snap_knots)
        U1, U2 = self._get_U_matrices(num_control_points)

        Q_d4_M = B_d_3 @ U1.T @ (np.eye(num_control_points) - W_d4_M @ U2 @ inv(U2.T@W_d4_M@U2) @ U2.T)
        
        self.Q_d4_M = Q_d4_M

    def get_Q_matrix(self):
        return self.Q_d4_M

    # --- Internal Class Methods ---

    def _create_clamped_knot_points(self, num_ctrl_pts, degree, start_time):
        number_of_knot_points = num_ctrl_pts + degree + 1
        number_of_unique_knot_points = number_of_knot_points - 2*degree
        unique_knot_points = np.arange(0,number_of_unique_knot_points) + start_time
        knot_points = np.zeros(number_of_knot_points) + start_time
        knot_points[degree : degree + number_of_unique_knot_points] = unique_knot_points
        knot_points[degree + number_of_unique_knot_points: 2*degree + number_of_unique_knot_points] = unique_knot_points[-1]
        return knot_points

    def _get_B_d3_matrix(self, d):
        d_inv = 1.0 / d
        accel_scalar = 2.0 / (d * (d - 1))
        B_d3 = np.array([
            [1,  1,       1,             0,             0,        0],
            [0,  d_inv,   3 * d_inv,     0,             0,        0],
            [0,  0,       accel_scalar,  0,             0,        0],
            [0,  0,       0,             accel_scalar,  0,        0],
            [0,  0,       0,            -3 * d_inv,    -d_inv,    0],
            [0,  0,       0,             1,             1,        1]
        ])
        return B_d3

    def _get_D_matrix(self, degree, knots, num_control_points):
        num_derivative_cps = num_control_points - 1
        diag_values = np.zeros(num_derivative_cps)
        for i in range(num_derivative_cps):
            denominator = knots[i + degree + 1] - knots[i + 1]
            diag_values[i] = degree / denominator
        D_bar = np.diag(diag_values)
        zero_row = np.zeros((1, num_derivative_cps))
        block_1 = np.vstack((D_bar, zero_row))
        block_2 = np.vstack((zero_row, D_bar))
        D_matrix = -block_1 + block_2
        return D_matrix

    def _get_S_matrix(self, degree, derivative_level, knots, num_control_points):
        S_matrix = None
        current_degree = degree
        current_num_cp = num_control_points
        current_knots = np.copy(knots)
        
        for i in range(derivative_level):
            D_current = self._get_D_matrix(current_degree, current_knots, current_num_cp)
            if S_matrix is None:
                S_matrix = D_current
            else:
                S_matrix = np.dot(S_matrix, D_current)
                
            current_degree -= 1
            current_num_cp -= 1
            current_knots = current_knots[1:-1]
            
        return S_matrix, current_knots

    def _get_W_matrix(self, S_matrix, snap_knots):
        num_intervals = len(snap_knots) - 1
        dt_values = np.zeros(num_intervals)
        for i in range(num_intervals):
            dt_values[i] = snap_knots[i+1] - snap_knots[i]
        integral_matrix = np.diag(dt_values)
        print(integral_matrix)
        W_matrix = np.dot(S_matrix, np.dot(integral_matrix, S_matrix.T))
        return W_matrix

    def _get_U_matrices(self, num_control_points):
        I = np.eye(num_control_points)
        U1 = np.hstack((I[:, 0:3], I[:, -3:]))
        U2 = I[:, 3:-3]
        return U1, U2


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    
    # -------------------------
    # 1. RUN SNAP OPTIMIZATION
    # -------------------------
    snap_degree = 4
    snap_ctrl_pts = 11

    # Pre-compute the Q matrix (This simulates the drone "booting up" on the ground)
    print("Pre-computing Q Matrix...")
    min_snap_evaluator = MinSnapEval(snap_ctrl_pts, snap_degree)
    Q_d4_M = min_snap_evaluator.get_Q_matrix()

    print("\n--- Running Performance Test: 100 Random Trajectories ---")
    
    # Start the high-precision timer
    start_time = time.perf_counter()

    for i in range(1):
        # Generate random 3x1 column vectors for the states
        # The scalars give them reasonable physical ranges (e.g., 0 to 10 meters for position)
        # p0 = np.random.rand(3, 1) * 10 
        # v0 = np.random.rand(3, 1) * 5 - 2.5
        # a0 = np.random.rand(3, 1) * 2 - 1
        
        # pf = np.random.rand(3, 1) * 10 
        # vf = np.random.rand(3, 1) * 5 - 2.5
        # af = np.random.rand(3, 1) * 2 - 1

        p0 = np.array([[0],[0],[0]])
        v0 = np.array([[-10],[-10],[10]])
        a0 = np.array([[0],[0],[0]])
        pf = np.array([[10],[10],[10]])
        vf = np.array([[-10],[-10],[-10]])
        af = np.array([[0],[0],[0]])
        
        # Build the boundary constraint matrix
        A_p = np.hstack((p0, v0, a0, af, vf, pf))
        
        # Calculate the exact optimal 3D flight path in a single dot product
        C_p_snap = A_p @ Q_d4_M
        
    # Stop the timer
    end_time = time.perf_counter()
    
    # Calculate and print the results
    total_time = end_time - start_time
    avg_time = total_time / 100
    
    print(f"Total time for 100 trajectories: {total_time:.6f} seconds")
    print(f"Average time per trajectory: {avg_time:.6f} seconds ({avg_time * 1000:.3f} ms)")


    # print("\n--- 3D Minimum Snap Control Points ---")
    # print(C_p_snap)
    from utils.visualization import plot_trajectory, plot_course_trajectory
    plot_trajectory(C_p_snap, min_snap_evaluator.knots, snap_degree)


    # from utils.benchmarks import run_batch_performance_test_clamped, run_clamped_performance_benchmark
    # run_batch_performance_test_clamped()
    # run_clamped_performance_benchmark_clamped()