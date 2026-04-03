'''
Visual and simple computational 
representation of snap minimization
using natural uniform B-Splines

SNAP OPTIMIZATION:
- C_p = [S E] @ Q_d4_M

- S = [ p(0) dp(0)/dt d^2p(0)/dt^2 ]
- E = [ p(M) dp(M)/dt d^2p(M)/dt^2 ]


'''
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
import time
import numpy as np
from numpy import eye
from numpy.linalg import inv

M4 = np.array([
    [ 1,  -4,   6,  -4,   1],
    [-4,  12,  -6, -12,  11],
    [ 6, -12,  -6,  12,  11],
    [-4,   4,   6,   4,   1],
    [ 1,   0,   0,   0,   0]
]) / 24.0

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
    ax.scatter(*curve[0], c='green', s=100, marker='*', label='Start')
    ax.scatter(*curve[-1], c='purple', s=100, marker='*', label='End')
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('B-Spline Minimum Snap Trajectory')
    ax.legend()
    
    # Ensure axes are scaled equally so the curve isn't warped
    ax.set_box_aspect([1,1,1]) 
    plt.show()


class MinSnapEval:
    '''
    Visual and simple computational 
    representation of snap minimization
    using clamped uniform B-Splines
    '''

    def __init__(self, num_control_points, degree):
        min_control_points = 6
        if num_control_points <= min_control_points:
            raise ValueError(f"A degree {degree} spline requires at least {degree + 1} control points. You provided {num_control_points}.")

        B_combined, U1, U2, Sigma, V = self._create_SVD(num_control_points)

        # 2. Calculate M dynamically
        M = num_control_points - degree
        self.knots = np.arange(-degree, M + degree + 1)

        # 3. Generate the dynamically sized W matrix
        W = self._get_W_matrix(M)

        # A_bar = (U2^T * W * U2)^T
        A_bar = (U2.T @ W @ U2).T
        
        # B_bar = (W * U2)^T
        B_bar = (W @ U2).T
        
        # Solve the linear system A_bar * X_bar = B_bar
        X_bar = np.linalg.solve(A_bar, B_bar)
        
        # Transpose back to get our final X block
        X = X_bar.T
        
        # Calculate Q using the solved X block instead of the inv() function
        self.Q = V @ inv(Sigma) @ U1.T @ (eye(num_control_points) - X @ U2.T)


    def get_Q_matrix(self):
        return self.Q


    def _create_SVD(self, num_control_points):
        # 2. Define the T arrays for tau = 0 (Start)
        T_pos_0 = np.array([[0], [0], [0], [0], [1]])
        T_vel_0 = np.array([[0], [0], [0], [1], [0]])
        T_acc_0 = np.array([[0], [0], [2], [0], [0]])

        # Multiply M4 by the T arrays to get the start states
        start_pos = M4 @ T_pos_0
        start_vel = M4 @ T_vel_0
        start_acc = M4 @ T_acc_0

        # Stack them horizontally to create the 5x3 start block
        B_d_M0 = np.hstack((start_pos, start_vel, start_acc))

        # 3. Define the T arrays for tau = 1 (End)
        T_pos_1 = np.array([[1], [1], [1], [1], [1]])
        T_vel_1 = np.array([[4], [3], [2], [1], [0]])
        T_acc_1 = np.array([[12], [6], [2], [0], [0]])

        # Multiply M4 by the T arrays to get the end states
        end_pos = M4 @ T_pos_1
        end_vel = M4 @ T_vel_1
        end_acc = M4 @ T_acc_1

        # Stack them horizontally to create the 5x3 end block
        B_d_MM = np.hstack((end_pos, end_vel, end_acc))

        # 1. Initialize the full-size boundary matrices with zeros
        # Shape will be (11, 3) based on your inputs
        B_0_full = np.zeros((num_control_points, 3))
        B_M_full = np.zeros((num_control_points, 3))

        # 2. Paste the 5x3 blocks into their proper positions
        # Start block goes at the very top (first 5 control points)
        B_0_full[0:5, :] = B_d_M0 
        
        # End block goes at the very bottom (last 5 control points)
        B_M_full[-5:, :] = B_d_MM

        # 3. Glue them together horizontally for the SVD: [B(0) B(M)]
        # Shape becomes (11, 6)
        B_combined = np.hstack((B_0_full, B_M_full))

        # 4. Run the Singular Value Decomposition (Theorem 2.6)
        U, s, Vh = np.linalg.svd(B_combined, full_matrices=True)

        # 5. We have 6 total constraints (3 start + 3 end)
        num_constraints = B_combined.shape[1]

        # 6. Extract the components required for the Q matrix
        U1 = U[:, :num_constraints]
        U2 = U[:, num_constraints:]
        Sigma = np.diag(s)
        V = Vh.T

        return B_combined, U1, U2, Sigma, V
    

    def _get_uniform_D_matrix(self, num_intervals, k):
        """
        Generates the discrete derivative matrix D_M^k.
        Shape is (M + k) x (M + k - 1)
        """
        rows = num_intervals + k
        cols = num_intervals + k - 1
        
        D = np.zeros((rows, cols))
        
        # Put -1 on the main diagonal
        D[:-1, :] -= np.eye(cols)
        
        # Put +1 on the sub-diagonal (shifted down by 1 row)
        D[1:, :] += np.eye(cols)
        
        return D

    def _get_W_matrix(self, M):
        """
        Generates the Minimum Snap W matrix for a Natural Uniform Spline.
        """
        # 1. Generate the cascading D matrices for degrees 1 through 4
        D1 = self._get_uniform_D_matrix(M, 1)
        D2 = self._get_uniform_D_matrix(M, 2)
        D3 = self._get_uniform_D_matrix(M, 3)
        D4 = self._get_uniform_D_matrix(M, 4)
        
        # 2. Multiply them all together to get the 4th derivative operator
        # Note: Matrix multiplication reads right-to-left
        D_4th = D4 @ D3 @ D2 @ D1
        
        # 3. W = D_4th * (Identity) * D_4th.T
        W = D_4th @ D_4th.T
        
        return W

def run_batch_performance_test():
    # The number of random trajectories we want to compute in each batch
    batch_sizes = [10, 100, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    
    execution_times = []
    
    # Static physics parameters
    snap_degree = 4
    snap_ctrl_pts = 11

    print("\nPre-computing Q Matrix once for all batches...")
    evaluator = MinSnapEval(snap_ctrl_pts, snap_degree)
    Q_d4_M = evaluator.get_Q_matrix()
    
    print("\nRunning Batch Execution Test...")
    
    for num_trajectories in batch_sizes:
        print(f"Calculating {num_trajectories:,} random trajectories...")
        
        # --- START TIMER ---
        start_exec = time.perf_counter()
        
        # Simulate the drone rapidly calculating new paths
        for _ in range(num_trajectories):
            p0 = np.random.rand(3, 1) * 10 
            v0 = np.random.rand(3, 1) * 5 - 2.5
            a0 = np.random.rand(3, 1) * 2 - 1
            
            pf = np.random.rand(3, 1) * 10 
            vf = np.random.rand(3, 1) * 5 - 2.5
            af = np.random.rand(3, 1) * 2 - 1
            
            # Build the boundary constraint matrix
            S = np.hstack((p0, v0, a0))
            E = np.hstack((pf, vf, af))
            SE = np.hstack((S, E))

            # C* = [S E] @ Q
            # This single dot product instantly bends all 11 control points to minimize snap!
            C_p_snap = SE @ Q_d4_M
            
        # --- STOP TIMER ---
        end_exec = time.perf_counter()
        
        total_time = end_exec - start_exec
        execution_times.append(total_time)

    # ==========================================
    # PLOT THE RESULTS
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(batch_sizes, execution_times, 'g-o', linewidth=2, markersize=6)
    
    # Format the graph
    ax.set_title('Batch Processing Time for Natural Uniform Minimum Snap Trajectories', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Trajectories Computed', fontsize=12)
    ax.set_ylabel('Total Computation Time (seconds)', fontsize=12)
    
    # Use a standard decimal format for the X-axis instead of scientific notation
    ax.ticklabel_format(style='plain', axis='x')
    
    # Add a grid and start axes at 0
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.show()


def run_performance_benchmark(max_control_points=100, iterations=1000, degree=4):
    """
    Benchmarks the setup and execution time of the Minimum Snap B-Spline solver.
    """
    print(f"🚀 Starting Benchmark: 7 to {max_control_points} Control Points")
    print(f"   Running {iterations} random trajectories per number of Control Points...\n")
    
    # We must start at degree 7 - as we have 6 states we must satisfy
    ctrl_pts_range = range(7, max_control_points + 1, 2)
    
    setup_times_ms = []
    exec_times_us = []
    
    for num_pts in ctrl_pts_range:
        # ==========================================
        # 1. MEASURE SETUP TIME (SVD & Solver)
        # ==========================================
        start_setup = time.perf_counter()
        evaluator = MinSnapEval(num_pts, degree)
        Q_matrix = evaluator.get_Q_matrix()
        end_setup = time.perf_counter()
        
        # Convert to milliseconds
        setup_times_ms.append((end_setup - start_setup) * 1000)
        
        # ==========================================
        # 2. PRE-GENERATE RANDOM STATES
        # We do this outside the timer so we are STRICTLY 
        # timing the matrix math, not numpy's random engine.
        # ==========================================
        
        start_exec = time.perf_counter()
        for _ in range(iterations):
            p0 = np.random.rand(3, 1) * 10 
            v0 = np.random.rand(3, 1) * 5 - 2.5
            a0 = np.random.rand(3, 1) * 2 - 1
            pf = np.random.rand(3, 1) * 10 
            vf = np.random.rand(3, 1) * 5 - 2.5
            af = np.random.rand(3, 1) * 2 - 1
            
            S = np.hstack((p0, v0, a0))
            E = np.hstack((pf, vf, af))
            SE = np.hstack((S, E))
            C_optimal = SE @ Q_matrix 
            
        end_exec = time.perf_counter()
        
        # Calculate average time per trajectory in MICROSECONDS
        total_exec_time = end_exec - start_exec
        avg_exec_us = (total_exec_time / iterations) * 1_000_000
        exec_times_us.append(avg_exec_us)
        
        print(f"Pts: {num_pts:3d} | Setup: {setup_times_ms[-1]:6.2f} ms | Exec: {avg_exec_us:6.3f} µs")

    # ==========================================
    # 4. PLOT THE RESULTS
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Setup Time
    ax1.plot(ctrl_pts_range, setup_times_ms, 'r-o', linewidth=2)
    ax1.set_title('Boot-up Time (Q Matrix Generation)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Number of Control Points')
    ax1.set_ylabel('Time (Milliseconds)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Execution Time
    ax2.plot(ctrl_pts_range, exec_times_us, 'b-o', linewidth=2)
    ax2.set_title('Real-Time Execution (SE @ Q)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Number of Control Points')
    ax2.set_ylabel('Time (Microseconds)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Optional: Set Y-axis to start at 0 for fair visual scaling
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    
    plt.suptitle('Minimum Snap B-Spline Performance Scaling (1000 Iterations/# of Control Points)', fontsize=16)
    plt.tight_layout()
    plt.show()

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
    knots = min_snap_evaluator.knots
    Q_d4_M = min_snap_evaluator.get_Q_matrix()


    # Start the high-precision timer
    start_time = time.perf_counter()

    for i in range(100):
        p0 = np.random.rand(3, 1) * 10 
        v0 = np.random.rand(3, 1) * 5 - 2.5
        a0 = np.random.rand(3, 1) * 2 - 1
        
        pf = np.random.rand(3, 1) * 10 
        vf = np.random.rand(3, 1) * 5 - 2.5
        af = np.random.rand(3, 1) * 2 - 1
        
        # Build the boundary constraint matrix
        S = np.hstack((p0, v0, a0))
        E = np.hstack((pf, vf, af))
        SE = np.hstack((S, E))

        # C* = [S E] @ Q
        # This single dot product instantly bends all 11 control points to minimize snap!
        C_p_snap = SE @ Q_d4_M
        
    # Stop the timer
    end_time = time.perf_counter()

    # Calculate and print the results
    total_time = end_time - start_time
    avg_time = total_time / 100
    
    print(f"Total time for 100 trajectories: {total_time:.6f} seconds")
    print(f"Average time per trajectory: {avg_time:.6f} seconds ({avg_time * 1000:.3f} ms)")

    print("\n==========================================")
    print("🚀 OPTIMAL CONTROL POINTS (C*) GENERATED:")
    print("==========================================")
    print(C_p_snap)
    print("Shape:", C_p_snap.shape)

    plot_trajectory(C_p_snap, min_snap_evaluator.knots, snap_degree)


# run_batch_performance_test()

# run_performance_benchmark()