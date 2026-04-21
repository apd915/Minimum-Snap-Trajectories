"""
Visual and simple computational representation of snap minimization
using natural uniform B-Splines.

SNAP OPTIMIZATION:
- C_p = [S E] @ Q_d4_M
- S = [ p(0) dp(0)/dt d^2p(0)/dt^2 ]
- E = [ p(M) dp(M)/dt d^2p(M)/dt^2 ]
"""

# ==========================================
# IMPORTS
# ==========================================
import time
import numpy as np
from numpy import eye
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline


# ==========================================
# MATHEMATICAL CONSTANTS
# ==========================================
# M4: The basis matrix for a degree 4 (quartic) B-spline.
# Used to map control points to specific spatial/derivative constraints.
M4 = np.array([
    [ 1,  -4,   6,  -4,   1],
    [-4,  12,  -6, -12,  11],
    [ 6, -12,  -6,  12,  11],
    [-4,   4,   6,   4,   1],
    [ 1,   0,   0,   0,   0]
]) / 24.0


# ==========================================
# CORE SOLVER CLASS
# ==========================================
class MinSnapEval:
    """
    Evaluator for generating Minimum Snap Trajectories using 
    Natural Uniform B-Splines via Singular Value Decomposition (SVD).
    """

    def __init__(self, num_control_points, degree):
        """
        Initializes the solver and pre-computes the optimal Q matrix.
        """
        # We enforce a minimum of 7 points because we have 6 boundary constraints (Start/End: Pos, Vel, Acc).
        min_control_points = 6
        if num_control_points <= min_control_points:
            raise ValueError(
                f"For 6 constraints, a degree {degree} spline requires at least {min_control_points + 1} "
                f"control points to optimize. You provided {num_control_points}."
            )

        # 1. Extract the SVD boundary matrices
        B_combined, U1, U2, Sigma, V = self._create_SVD(num_control_points)

        # 2. Calculate M (number of time segments) dynamically
        M = num_control_points - degree
        
        # Define the natural uniform knot vector (-degree to M + degree)
        self.knots = np.arange(-degree, M + degree + 1)

        # 3. Generate the dynamically sized W (penalty) matrix for the 4th derivative (Snap)
        W = self._get_W_matrix(M)

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
        self.Q = V @ inv(Sigma) @ U1.T @ (eye(num_control_points) - X @ U2.T)

    def get_Q_matrix(self):
        """Returns the pre-computed Q mapping matrix."""
        return self.Q

    def _create_SVD(self, num_control_points):
        """
        Constructs the boundary constraint matrices and performs SVD to isolate
        the null space (free control points) for snap optimization.
        """
        # --- TAU = 0 (Start Constraints) ---
        T_pos_0 = np.array([[0], [0], [0], [0], [1]])
        T_vel_0 = np.array([[0], [0], [0], [1], [0]])
        T_acc_0 = np.array([[0], [0], [2], [0], [0]])

        start_pos = M4 @ T_pos_0
        start_vel = M4 @ T_vel_0
        start_acc = M4 @ T_acc_0

        # 5x3 block representing the active control points at t=0
        B_d_M0 = np.hstack((start_pos, start_vel, start_acc))

        # --- TAU = 1 (End Constraints) ---
        T_pos_1 = np.array([[1], [1], [1], [1], [1]])
        T_vel_1 = np.array([[4], [3], [2], [1], [0]])
        T_acc_1 = np.array([[12], [6], [2], [0], [0]])

        end_pos = M4 @ T_pos_1
        end_vel = M4 @ T_vel_1
        end_acc = M4 @ T_acc_1

        # 5x3 block representing the active control points at t=M
        B_d_MM = np.hstack((end_pos, end_vel, end_acc))

        # Initialize the full-size boundary matrices with zeros
        B_0_full = np.zeros((num_control_points, 3))
        B_M_full = np.zeros((num_control_points, 3))

        # Paste the active blocks into their respective ends
        B_0_full[0:5, :] = B_d_M0 
        B_M_full[-5:, :] = B_d_MM

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

    def _get_uniform_D_matrix(self, num_intervals, k):
        """
        Generates the discrete derivative operator matrix D_M^k.
        For natural uniform splines, this simplifies to pure subtraction.
        """
        rows = num_intervals + k
        cols = num_intervals + k - 1
        
        D = np.zeros((rows, cols))
        
        # Main diagonal (-1) represents the current control point
        D[:-1, :] -= np.eye(cols)
        
        # Sub-diagonal (+1) represents the next control point
        D[1:, :] += np.eye(cols)
        
        return D

    def _get_W_matrix(self, M):
        """
        Generates the Minimum Snap Penalty matrix (W) for a Natural Uniform Spline.
        Cascades 4 derivative matrices to represent the 4th derivative (Snap).
        """
        D1 = self._get_uniform_D_matrix(M, 1)
        D2 = self._get_uniform_D_matrix(M, 2)
        D3 = self._get_uniform_D_matrix(M, 3)
        D4 = self._get_uniform_D_matrix(M, 4)
        
        # Matrix multiplication reads right-to-left
        D_4th = D4 @ D3 @ D2 @ D1
        
        # W = D_4th * (Identity) * D_4th.T
        W = D_4th @ D_4th.T
        
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
    snap_degree = 4
    snap_ctrl_pts = 11

    print("Pre-computing Q Matrix...")
    min_snap_evaluator = MinSnapEval(snap_ctrl_pts, snap_degree)
    knots = min_snap_evaluator.knots
    Q_d4_M = min_snap_evaluator.get_Q_matrix()

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

    print("\n==========================================")
    print("🚀 OPTIMAL CONTROL POINTS (C*) GENERATED:")
    print("==========================================")
    print(C_p_snap)
    print("Shape:", C_p_snap.shape)

    # Plot the last trajectory from the loop
    plot_trajectory(C_p_snap, min_snap_evaluator.knots, snap_degree)


    # ----------------------------------------------------
    # OPTIONAL BENCHMARKS
    # (Uncomment the lines below to run them)
    # ----------------------------------------------------
    
    # run_batch_performance_test()
    # run_performance_benchmark(max_control_points=100)