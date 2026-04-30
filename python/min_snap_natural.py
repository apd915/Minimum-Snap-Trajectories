"""
Visual and simple computational representation of snap minimization
using natural uniform B-Splines.

SNAP OPTIMIZATION:
- C_p = [S E] @ Q_d4_M
- S = [ p(0) dp(0)/dt d^2p(0)/dt^2 ]
- E = [ p(M) dp(M)/dt d^2p(M)/dt^2 ]

C++ for Deployment: When you eventually wrap this into a ROS C++ node
for the physical quadcopter, you will swap back to Cholesky, 
use fixed-size Eigen matrices, and achieve sub-microsecond 
trajectory generation.
"""

# ==========================================
# IMPORTS
# ==========================================
import time
import math
import numpy as np
from numpy import eye
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from b_spline_constants import M_STENCILS, S_STENCILS, D_STENCILS, T_STENCILS


# ==========================================
# CORE SOLVER CLASS
# ==========================================
class MinSnapEval:
    """
    Evaluator for generating Minimum Snap Trajectories using 
    Natural Uniform B-Splines via Singular Value Decomposition (SVD).
    """

    def __init__(self, num_segments, degree=4):
        """
        Initializes the solver based on the desired number of flight segments (base time).
        """
        # 1. Intuitive Safety Checks
        if degree < 4:
            raise ValueError(f"Minimum Snap requires a polynomial of at least degree 4. You provided degree {degree}.")
        if num_segments < 3:
            raise ValueError(f"To satisfy 6 physical constraints, you need at least 3 flight segments. You provided {num_segments}.")
            
        # 2. Automatically calculate the required "ghost" control points
        self.num_control_points = num_segments + degree
        
        # We store M internally since it's used to build the W/S matrices
        self.M = num_segments
        self.degree = degree
        
        # 3. Define the knot vector
        self.knots = np.arange(-self.degree, self.M + self.degree + 1)

        # 4. Extract the SVD boundary matrices (pass in the calculated control points)
        B_combined, U1, U2, Sigma, V = self._create_SVD(self.num_control_points)

        # 3. Generate the dynamically sized W (penalty) matrix for the 4th derivative (Snap)
        W = self._get_W_matrix(self.M)

        # 4. Compute the Q Matrix using the Linear Solver Optimization (Section 2.7)
        # We solve the system A_bar * X_bar = B_bar to avoid taking the direct inverse 
        # of large matrices, maximizing computational efficiency and stability.
        
        # A_bar = (U2^T * W * U2)^T
        A_bar = (U2.T @ W @ U2).T
        
        # B_bar = (W * U2)^T
        B_bar = (W @ U2).T
        
        # Solve the linear system
        X_bar = np.linalg.solve(A_bar, B_bar)
        
        # Transpose back to get our final X block
        X = X_bar.T
        
        # Construct the final generalized inverse mapping matrix (Q)
        self.Q = V @ inv(Sigma) @ U1.T @ (eye(self.num_control_points) - X @ U2.T)

    def get_Q_matrix(self):
        """Returns the pre-computed Q mapping matrix."""
        return self.Q
    

    def _get_M_matrix(self, degree):
        """
        Dynamically generates the Basis Mapping Matrix (M) for ANY degree uniform B-spline.
        Maps control points to polynomial coefficients for tau in [0, 1].
        """
        M = np.zeros((degree + 1, degree + 1), dtype=int)
        
        for r in range(degree + 1):
            for j in range(degree + 1):
                val = 0
                for k in range(degree - r + 1):
                    term1 = (-1)**k
                    term2 = math.comb(degree + 1, k)
                    term3 = math.comb(degree, j)
                    
                    # Handle 0^0 safely
                    base = degree - r - k
                    term4 = (base**j) if not (base == 0 and j == 0) else 1
                    
                    val += term1 * term2 * term3 * term4
                    
                M[r, j] = val
        
        scalar = math.factorial(degree)
        return M, scalar


    def _get_T_vector(self, degree, derivative_order, tau):
        """
        Dynamically generates the Time vector T(tau) and its derivatives.
        """
        T = np.zeros((degree + 1, 1))
        for i in range(degree + 1):
            power = degree - i
            if power >= derivative_order:
                # Calculate the cascaded derivative scalar using the power rule
                scalar = math.prod(range(power - derivative_order + 1, power + 1)) if derivative_order > 0 else 1
                T[i, 0] = scalar * (tau ** (power - derivative_order))
        return T
    
    def _get_boundary_states(self, M_matrix, degree):
        """
        Calculates the active constraint blocks for the start (tau=0) 
        and end (tau=1) of the trajectory using a hybrid lookup/dynamic approach.
        """
        # --- FAST PATH: Hardcoded Lookup ---
        if degree in T_STENCILS:
            T_start_combined = T_STENCILS[degree]['start']
            T_end_combined = T_STENCILS[degree]['end']
            
            # Execute a single matrix multiplication for all 3 constraints simultaneously
            B_d_M0 = M_matrix @ T_start_combined
            B_d_MM = M_matrix @ T_end_combined
            
            return B_d_M0, B_d_MM

        # --- FALLBACK PATH: Dynamic Generation ---
        # TAU = 0 (Start Constraints)
        start_pos = self._get_T_vector(degree, 0, 0)
        start_vel = self._get_T_vector(degree, 1, 0)
        start_acc = self._get_T_vector(degree, 2, 0)
        T_start_combined = np.hstack((start_pos, start_vel, start_acc))
        B_d_M0 = M_matrix @ T_start_combined

        # TAU = 1 (End Constraints)
        end_pos = self._get_T_vector(degree, 0, 1)
        end_vel = self._get_T_vector(degree, 1, 1)
        end_acc = self._get_T_vector(degree, 2, 1)
        T_end_combined = np.hstack((end_pos, end_vel, end_acc))
        B_d_MM = M_matrix @ T_end_combined

        return B_d_M0, B_d_MM


    def _create_SVD(self, num_control_points):
        """
        Constructs the boundary constraint matrices and performs SVD to isolate
        the null space (free control points) for snap optimization.
        """
        
        # --- 1. RESOLVE THE M MATRIX ---
        if self.degree in M_STENCILS:
            # Fast Path: Memory Lookup
            M_matrix = M_STENCILS[self.degree]
        else:
            # Fallback Path: Dynamic Generation
            print(f"Warning: Generating M^{self.degree} on the fly.")
            M_int, scalar = self._get_M_matrix(self.degree)
            M_matrix = M_int / scalar 

        # --- 2. GENERATE BOUNDARY BLOCKS ---
        B_d_M0, B_d_MM = self._get_boundary_states(M_matrix, self.degree) # block representing the active control points at t=0 and t=M
        

        # Initialize the full-size boundary matrices with zeros
        B_0_full = np.zeros((num_control_points, 3))
        B_M_full = np.zeros((num_control_points, 3))

        # Paste the active blocks into their respective ends# The injection window size is exactly degree + 1
        window_size = self.degree + 1
        B_0_full[0:window_size, :] = B_d_M0 
        B_M_full[-window_size:, :] = B_d_MM

        # Glue them together horizontally for the SVD: [B(0) B(M)]
        B_combined = np.hstack((B_0_full, B_M_full))

        # Run the Singular Value Decomposition
        U, s, Vh = np.linalg.svd(B_combined, full_matrices=True)

        # We have 6 total constraints (3 start + 3 end)
        num_constraints = B_combined.shape[1]

        # Extract the components required for the Q matrix
        # U1 represents the constrained space; U2 represents the free null space
        U1 = U[:, :num_constraints]
        U2 = U[:, num_constraints:]
        Sigma = np.diag(s)
        V = Vh.T

        return B_combined, U1, U2, Sigma, V
    
    
    def _get_fast_cascaded_D_matrix(self, M, degree, derivative_order):
        """
        Direct O(N) LUT implementation of the book's cascaded D matrix.
        Yields an (M+d) x (M+d-l) matrix matching the exact output of D^d * D^{d-1}...
        """
        if derivative_order not in D_STENCILS:
            raise NotImplementedError(f"Stencil for derivative {derivative_order} not hardcoded.")

        stencil = D_STENCILS[derivative_order]

        # The book's dimensions: mapping from (M + d - l) up to (M + d)
        rows = M + degree
        cols = M + degree - derivative_order

        D_cascaded = np.zeros((rows, cols))

        # The book's D matrix cascades column-wise
        for i in range(cols):
            D_cascaded[i : i + len(stencil), i] = stencil

        return D_cascaded
    
    def _get_S_matrix(self, M, k):
        """
        Dynamically populates the S matrix using the pre-calculated memory LUT.
        Applies exact boundary truncation patches for splines bleeding out of [0, M].
        """
        if k not in S_STENCILS:
            raise NotImplementedError(f"Integral stencil for k={k} is not yet hardcoded.")

        size = M + k
        S = np.zeros((size, size))
        stencil = S_STENCILS[k]
        
        # 1. Populate the main diagonal (s_0)
        S += np.diag(np.full(size, stencil[0]))
        
        # 2. Populate the sub and super diagonals (s_1, s_2, etc.)
        for i in range(1, len(stencil)):
            S += np.diag(np.full(size - i, stencil[i]), k=i)   # Super-diagonal
            S += np.diag(np.full(size - i, stencil[i]), k=-i)  # Sub-diagonal
            
        # 3. APPLY BOUNDARY TRUNCATION PATCHES
        if k == 1: # [[1/3, 2/3], [1/6]]
            # The first and last basis functions lose exactly half their area 
            # because they bleed outside the [0, M] integral bounds.
            S[0, 0] = 1/3
            S[-1, -1] = 1/3

        elif k == 2: # [[1/20, 1/2, 11/20], [13/120, 26/120], [1/120]]
            # Main Diagonal (b_0*b_0 and b_1*b_1)
            S[0, 0] = 1/20
            S[1, 1] = 1/2
            S[-1, -1] = 1/20
            S[-2, -2] = 1/2
            
            # First Off-Diagonal (b_0*b_1)
            S[0, 1] = S[1, 0] = 13/120
            S[-1, -2] = S[-2, -1] = 13/120
            
            # Note: The second off-diagonal (b_0*b_2) doesn't bleed into 
            # negative time, so it remains the infinite value of 1/120!

        elif k == 3: # [[1/252, 151/630, 599/1260, 151/315], [43/1680, 59/280, 397/1680], [1/84, 1/42], [1/5040]]
            # 1. Main Diagonal Patches (S_0,0 / S_1,1 / S_2,2)
            S[0, 0] = 1/252
            S[1, 1] = 151/630
            S[2, 2] = 599/1260
            
            S[-1, -1] = 1/252
            S[-2, -2] = 151/630
            S[-3, -3] = 599/1260

            # --- First Off-Diagonal Patches ---
            S[0, 1] = S[1, 0] = 43/1680
            S[1, 2] = S[2, 1] = 59/280   # (Which is exactly 354/1680)
            
            S[-1, -2] = S[-2, -1] = 43/1680
            S[-2, -3] = S[-3, -2] = 59/280

            # --- Second Off-Diagonal Patches ---
            S[0, 2] = S[2, 0] = 1/84

            S[-1, -3] = S[-3, -1] = 1/84


        elif k > 3:
            # Placeholder: The boundary patches for k=2 and higher are matrices 
            # (e.g., a 2x2 corner block for k=2) because the overlap bleeds further.
            raise NotImplementedError(f"Boundary patches for k={k} not yet hardcoded.")
        
        # print(f"S:\n{S}")
        return S

    def _get_W_matrix(self, M):
        """
        Generates the Minimum Snap Penalty matrix (W) for a Natural Uniform Spline.
        Cascades 4 derivative matrices to represent the 4th derivative (Snap).
        """
        snap_minimization = 4

        D_cascaded = self._get_fast_cascaded_D_matrix(M, self.degree, snap_minimization)

        k = self.degree-snap_minimization
        S = self._get_S_matrix(M, k)
        # print(f'S Matrix:\n{S}')
        
        # W = D_4th @ S @ D_4th.T
        W = D_cascaded @ S @ D_cascaded.T
        
        return W


# ==========================================
# UTILITY & VISUALIZATION
# ==========================================
def plot_trajectory(ctrl_pts, knots, degree):
    """
    Evaluates and plots the 3D Minimum Snap Trajectory and its control polygon.
    """
    pts = ctrl_pts.T 
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Plot the Control Polygon (The mathematical "gravitational anchors")
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'ro--', alpha=0.4, label='Control Polygon')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='red', s=30)
    
    # 2. Evaluate and Plot the actual Minimum Snap B-Spline Curve
    spline = BSpline(knots, pts, degree)
    t_smooth = np.linspace(knots[degree], knots[-degree-1], 100)
    curve = spline(t_smooth)
    
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], 'b-', linewidth=3, label='Min Snap Trajectory')
    
    # Plot Start and End constraint points
    ax.scatter(*curve[0], c='green', s=100, marker='*', label='Start')
    ax.scatter(*curve[-1], c='purple', s=100, marker='*', label='End')
    
    # Formatting
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('B-Spline Minimum Snap Trajectory')
    ax.legend()
    ax.set_box_aspect([1, 1, 1]) 
    plt.show()


# ==========================================
# BENCHMARKING FUNCTIONS
# ==========================================
def run_batch_performance_test():
    """
    Tests execution time scalability by simulating a drone rapidly 
    calculating massive batches of trajectories.
    """
    batch_sizes = [10, 100, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    execution_times = []
    
    snap_degree = 4
    snap_ctrl_pts = 11

    print("\nPre-computing Q Matrix once for all batches...")
    evaluator = MinSnapEval(snap_ctrl_pts, snap_degree)
    Q_d4_M = evaluator.get_Q_matrix()
    
    print("\nRunning Batch Execution Test...")
    for num_trajectories in batch_sizes:
        print(f"Calculating {num_trajectories:,} random trajectories...")
        
        start_exec = time.perf_counter()
        
        for _ in range(num_trajectories):
            # Generate random boundaries
            p0, pf = np.random.rand(3, 1) * 10, np.random.rand(3, 1) * 10
            v0, vf = np.random.rand(3, 1) * 5 - 2.5, np.random.rand(3, 1) * 5 - 2.5
            a0, af = np.random.rand(3, 1) * 2 - 1, np.random.rand(3, 1) * 2 - 1
            
            S = np.hstack((p0, v0, a0))
            E = np.hstack((pf, vf, af))
            SE = np.hstack((S, E))

            # The real-time mapping math
            C_p_snap = SE @ Q_d4_M
            
        end_exec = time.perf_counter()
        execution_times.append(end_exec - start_exec)

    # Plot batch results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(batch_sizes, execution_times, 'g-o', linewidth=2, markersize=6)
    ax.set_title('Batch Processing Time for Natural Uniform Minimum Snap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Trajectories Computed', fontsize=12)
    ax.set_ylabel('Total Computation Time (seconds)', fontsize=12)
    ax.ticklabel_format(style='plain', axis='x')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.show()


def run_performance_benchmark(max_control_points=100, iterations=1000, degree=4):
    """
    Benchmarks the setup (offline) and execution (real-time) time of the solver 
    as the number of control points scales up.
    """
    print(f"\n🚀 Starting Benchmark: 7 to {max_control_points} Control Points")
    print(f"   Running {iterations} random trajectories per step...\n")
    
    ctrl_pts_range = range(7, max_control_points + 1, 2)
    setup_times_ms = []
    exec_times_us = []
    
    for num_pts in ctrl_pts_range:
        # 1. SETUP PHASE (Boot-up Math)
        start_setup = time.perf_counter()
        evaluator = MinSnapEval(num_pts, degree)
        Q_matrix = evaluator.get_Q_matrix()
        end_setup = time.perf_counter()
        
        setup_times_ms.append((end_setup - start_setup) * 1000)
        
        # 2. EXECUTION PHASE (Real-time Math)
        start_exec = time.perf_counter()
        for _ in range(iterations):
            p0, pf = np.random.rand(3, 1) * 10, np.random.rand(3, 1) * 10
            v0, vf = np.random.rand(3, 1) * 5 - 2.5, np.random.rand(3, 1) * 5 - 2.5
            a0, af = np.random.rand(3, 1) * 2 - 1, np.random.rand(3, 1) * 2 - 1
            
            S, E = np.hstack((p0, v0, a0)), np.hstack((pf, vf, af))
            SE = np.hstack((S, E))
            C_optimal = SE @ Q_matrix 
            
        end_exec = time.perf_counter()
        
        avg_exec_us = ((end_exec - start_exec) / iterations) * 1_000_000
        exec_times_us.append(avg_exec_us)
        
        print(f"Pts: {num_pts:3d} | Setup: {setup_times_ms[-1]:6.2f} ms | Exec: {avg_exec_us:6.3f} µs")

    # Plot Scaling Results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(ctrl_pts_range, setup_times_ms, 'r-o', linewidth=2)
    ax1.set_title('Boot-up Time (Q Matrix Generation)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Number of Control Points')
    ax1.set_ylabel('Time (Milliseconds)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax2.plot(ctrl_pts_range, exec_times_us, 'b-o', linewidth=2)
    ax2.set_title('Real-Time Execution (SE @ Q)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Number of Control Points')
    ax2.set_ylabel('Time (Microseconds)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    
    plt.suptitle('Minimum Snap B-Spline Performance Scaling', fontsize=16)
    plt.tight_layout()
    plt.show()


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # ----------------------------------------------------
    # DEMO: SINGLE FLIGHT PATH GENERATION
    # ----------------------------------------------------
    degree = 4
    snap_num_segments = 1000

    print("Pre-computing Q Matrix...")
    min_snap_evaluator = MinSnapEval(snap_num_segments, degree)
    knots = min_snap_evaluator.knots
    Q_d4_M = min_snap_evaluator.get_Q_matrix()
    
    # for i in range(1,8):
    #     Mi, scalar = min_snap_evaluator._get_M_matrix(i)
    #     print(f"M^{i}: 1/{scalar}\n{Mi}\n")

    start_time = time.perf_counter()

    for i in range(100):
        # Generate random start and end conditions
        p0 = np.random.rand(3, 1) * 10 
        v0 = np.random.rand(3, 1) * 5 - 2.5
        a0 = np.random.rand(3, 1) * 2 - 1
        
        pf = np.random.rand(3, 1) * 10 
        vf = np.random.rand(3, 1) * 5 - 2.5
        af = np.random.rand(3, 1) * 2 - 1
        
        S = np.hstack((p0, v0, a0))
        E = np.hstack((pf, vf, af))
        SE = np.hstack((S, E))

        # Core optimization calculation
        C_p_snap = SE @ Q_d4_M
        
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time = total_time / 100
    
    print(f"Total time for 100 trajectories: {total_time:.6f} seconds")
    print(f"Average time per trajectory: {avg_time:.6f} seconds ({avg_time * 1000:.3f} ms)")

    # Plot the last trajectory from the loop
    plot_trajectory(C_p_snap, min_snap_evaluator.knots, degree)


    # ----------------------------------------------------
    # OPTIONAL BENCHMARKS
    # (Uncomment the lines below to run them)
    # ----------------------------------------------------
    
    # run_batch_performance_test()
    # run_performance_benchmark(max_control_points=100)