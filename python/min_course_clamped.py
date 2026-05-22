import numpy as np
import time
from numpy.linalg import inv

# ==========================================
# MINIMUM COURSE EVALUATOR (1D Course)
# ==========================================

class MinCourseEval:
    def __init__(self, num_control_points, degree):
        start_time = 0  

        self.knots = self._create_clamped_knot_points(num_control_points, degree, start_time)


        S_d2_M, snap_knots = self._get_S_matrix(degree, degree, self.knots, num_control_points)
        W_d2_M = self._get_W_matrix(S_d2_M, snap_knots)
        U1_psi, U2_psi = self._get_U_matrices_course(num_control_points)

        Q_d2_M = U1_psi.T @ (np.eye(num_control_points) - W_d2_M @ U2_psi @ inv(U2_psi.T@W_d2_M@U2_psi) @ U2_psi.T)
        
        self.Q_d2_M = Q_d2_M

    def get_Q_matrix(self):
        return self.Q_d2_M

    # --- Internal Class Methods ---

    def _create_clamped_knot_points(self, num_ctrl_pts, degree, start_time):
        number_of_knot_points = num_ctrl_pts + degree + 1
        number_of_unique_knot_points = number_of_knot_points - 2*degree
        unique_knot_points = np.arange(0,number_of_unique_knot_points) + start_time
        knot_points = np.zeros(number_of_knot_points) + start_time
        knot_points[degree : degree + number_of_unique_knot_points] = unique_knot_points
        knot_points[degree + number_of_unique_knot_points: 2*degree + number_of_unique_knot_points] = unique_knot_points[-1]
        return knot_points

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
        W_matrix = np.dot(S_matrix, np.dot(integral_matrix, S_matrix.T))
        return W_matrix

    def _get_U_matrices_course(self, num_control_points):
        I = np.eye(num_control_points)
        U1 = np.hstack((I[:, 0:1], I[:, -1:]))
        U2 = I[:, 1:-1]
        return U1, U2
    


if __name__ == '__main__':
    # # -------------------------
    # # RUN COURSE OPTIMIZATION
    # # -------------------------
    course_degree = 2
    course_ctrl_pts = 11

    # Pre-compute the Course Q matrix
    print("\nPre-computing Course Q Matrix...")
    min_course_evaluator = MinCourseEval(course_ctrl_pts, course_degree)
    Q_d2_M = min_course_evaluator.get_Q_matrix()
    
    print("\n--- Running Performance Test: 100 Random Course Trajectories ---")
    
    # Start the high-precision timer
    start_time_course = time.perf_counter()
    
    for i in range(100):
        # Generate random 1x1 column vectors for the yaw states
        # The math: [0.0 to 1.0] * 360 - 180 = Random angle between -180 and 180 degrees
        psi0 = np.random.rand(1, 1) * 360 - 180
        psif = np.random.rand(1, 1) * 360 - 180
        
        # Build the boundary constraint matrix (Just 2 items!)
        A_p_course = np.hstack((psi0, psif))
        
        # Calculate the exact optimal 1D yaw path in a single dot product
        C_p_course = A_p_course @ Q_d2_M
        
    # Stop the timer
    end_time_course = time.perf_counter()
    
    # Calculate and print the results
    total_time_course = end_time_course - start_time_course
    avg_time_course = total_time_course / 100
    
    print(f"Total time for 100 yaw trajectories: {total_time_course:.6f} seconds")
    print(f"Average time per yaw trajectory: {avg_time_course:.6f} seconds ({avg_time_course * 1000:.3f} ms)")

    # print("\n--- 1D Minimum Course Control Points ---")
    print(C_p_course)

    from utils.visualization import plot_course_trajectory
    plot_course_trajectory(C_p_course, min_course_evaluator.knots, course_degree)