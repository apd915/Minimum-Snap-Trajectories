'''
Visual and simple computational 
representation of snap minimization
using B-Splines

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
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

# ==========================================
# PLOTTING FUNCTIONS
# ==========================================

def plot_trajectory(ctrl_pts, knots, degree):
    # Transpose control points so scipy can read them as (num_points, dimensions)
    pts = ctrl_pts.T 
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Plot the Control Polygon (The mathematical "gravitational anchors")
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'ro--', alpha=0.4, label='Control Polygon')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='red', s=30)
    
    # 2. Evaluate and Plot the actual Minimum Snap B-Spline Curve
    spline = BSpline(knots, pts, degree)
    
    # Generate 100 smooth time steps between the start and end clamps
    t_smooth = np.linspace(knots[degree], knots[-degree-1], 100)
    curve = spline(t_smooth)
    
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], 'b-', linewidth=3, label='Min Snap Trajectory')
    
    # Plot Start and End points
    ax.scatter(*pts[0], c='green', s=100, marker='*', label='Start')
    ax.scatter(*pts[-1], c='purple', s=100, marker='*', label='End')
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('B-Spline Minimum Snap Trajectory')
    ax.legend()
    
    # Ensure axes are scaled equally so the curve isn't warped
    ax.set_box_aspect([1,1,1]) 
    plt.show()

def plot_course_trajectory(ctrl_pts, knots, degree):
    pts = ctrl_pts.T  # Shape becomes (N, 1)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Generate the physical time array for the X-axis of the control points
    cp_time = np.linspace(knots[0], knots[-1], len(pts))
    ax.plot(cp_time, pts, 'ro--', alpha=0.5, label='Course Control Polygon')
    
    # Evaluate smooth spline
    spline = BSpline(knots, pts, degree)
    t_smooth = np.linspace(knots[degree], knots[-degree-1], 100)
    curve = spline(t_smooth)
    
    ax.plot(t_smooth, curve, 'b-', linewidth=3, label='Optimized Course Trajectory')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Course Angle (Degrees or Rads)')
    ax.set_title('Minimum 2nd-Derivative Course Spline')
    ax.legend()
    ax.grid(True)
    plt.show()


# ==========================================
# MINIMUM SNAP EVALUATOR (3D POSITION)
# ==========================================

class MinSnapEval:
    '''
    Visual and simple computational 
    representation of snap minimization
    using B-Splines
    '''

    def __init__(self, initial_states, final_states, num_control_points, degree):
        start_time = 0
        
        # 1. Define your placeholder initial states (Start at origin, at rest)
        p0 = initial_states[0]  # Initial position (X, Y, Z)
        v0 = initial_states[1]  # Initial velocity
        a0 = initial_states[2]  # Initial acceleration

        # 2. Define your placeholder final states (End at target, at rest)
        pf = final_states[0]   # Final position (X, Y, Z)
        vf = final_states[1]   # Final velocity
        af = final_states[2]   # Final acceleration

        self.knots = self._create_clamped_knot_points(num_control_points, degree, start_time)

        # 3. Glue them together horizontally to build Ap
        # Order: [p0, v0, a0, af, vf, pf]
        A_p = np.hstack((p0, v0, a0, af, vf, pf))
        # print(f"A_p = {A_p}\n")
        B_d_3 = self._get_B_d3_matrix(degree)

        S_d4_M, snap_knots = self._get_S_matrix(degree, degree, self.knots, num_control_points)
        W_d4_M = self._get_W_matrix(S_d4_M, snap_knots)
        U1, U2 = self._get_U_matrices(num_control_points)

        Q_d4_M = B_d_3 @ U1.T @ (np.eye(num_control_points) - W_d4_M @ U2 @ inv(U2.T@W_d4_M@U2) @ U2.T)

        C_p = A_p @ Q_d4_M
        
        self.optimized_ctrl_pts = C_p

    def min_snap_ctrl_pts(self):
        return self.optimized_ctrl_pts

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
        W_matrix = np.dot(S_matrix, np.dot(integral_matrix, S_matrix.T))
        return W_matrix

    def _get_U_matrices(self, num_control_points):
        I = np.eye(num_control_points)
        U1 = np.hstack((I[:, 0:3], I[:, -3:]))
        U2 = I[:, 3:-3]
        return U1, U2


# ==========================================
# MINIMUM COURSE EVALUATOR (1D Course)
# ==========================================

class MinCourseEval:
    def __init__(self, initial_states, final_states, num_control_points, degree):
        start_time = 0
        
        # 1. Define your placeholder initial states
        psi0 = initial_states[0]  
        
        # 2. Define your placeholder final states
        psif = final_states[0]   

        self.knots = self._create_clamped_knot_points(num_control_points, degree, start_time)

        # 3. Glue them together horizontally to build Ap
        A_p = np.hstack((psi0, psif))

        S_d2_M, snap_knots = self._get_S_matrix(degree, degree, self.knots, num_control_points)
        W_d2_M = self._get_W_matrix(S_d2_M, snap_knots)
        U1_psi, U2_psi = self._get_U_matrices_course(num_control_points)

        Q_d2_M = U1_psi.T @ (np.eye(num_control_points) - W_d2_M @ U2_psi @ inv(U2_psi.T@W_d2_M@U2_psi) @ U2_psi.T)

        C_p = A_p @ Q_d2_M
        
        self.optimized_ctrl_pts = C_p

    def min_course_ctrl_pts(self):
        return self.optimized_ctrl_pts

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


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    
    # -------------------------
    # 1. RUN SNAP OPTIMIZATION
    # -------------------------
    snap_degree = 4
    snap_ctrl_pts = 11
    
    p0 = np.array([[0], [0], [0]])  # Initial position (X, Y, Z)
    v0 = np.array([[3], [0], [8]])  # Initial velocity
    a0 = np.array([[3], [0], [5]])  # Initial acceleration

    pf = np.array([[-10], [10], [15]]) # Final position (X, Y, Z)
    vf = np.array([[10], [0], [-10]])  # Final velocity
    af = np.array([[-5], [0], [-3]])   # Final acceleration

    initial_states_snap = np.array([p0, v0, a0])
    final_states_snap = np.array([pf, vf, af])
    min_snap_evaluator = MinSnapEval(initial_states_snap, final_states_snap, snap_ctrl_pts, snap_degree)

    snap_optimized_pts = min_snap_evaluator.min_snap_ctrl_pts()
    print("\n--- 3D Minimum Snap Control Points ---")
    print(snap_optimized_pts)

    plot_trajectory(snap_optimized_pts, min_snap_evaluator.knots, snap_degree)

    # -------------------------
    # 2. RUN COURSE OPTIMIZATION
    # -------------------------
    course_degree = 2
    course_ctrl_pts = 11

    psi0 = np.array([[0]])
    psif = np.array([[75]])

    initial_states_course = np.array([psi0])
    final_states_course = np.array([psif])

    min_course_evaluator = MinCourseEval(initial_states_course, final_states_course, course_ctrl_pts, course_degree)

    course_optimized_pts = min_course_evaluator.min_course_ctrl_pts()
    print("\n--- 1D Minimum Course Control Points ---")
    print(course_optimized_pts)

    plot_course_trajectory(course_optimized_pts, min_course_evaluator.knots, course_degree)