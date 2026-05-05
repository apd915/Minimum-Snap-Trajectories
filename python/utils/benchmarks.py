import time
import numpy as np
import matplotlib.pyplot as plt
from min_snap_natural import MinSnapEval

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