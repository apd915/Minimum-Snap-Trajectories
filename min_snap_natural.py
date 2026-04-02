'''
Visual and simple computational 
representation of snap minimization
using natural uniform B-Splines

SNAP OPTIMIZATION:
- C_p = [S E] @ Q_d4_M

- S = [ p(0) dp(0)/dt d^2p(0)/dt^2 ]
- E = [ p(M) dp(M)/dt d^2p(M)/dt^2 ]


'''
import numpy as np

M4 = np.array([
    [ 1,  -4,   6,  -4,   1],
    [-4,  12,  -6, -12,  11],
    [ 6, -12,  -6,  12,  11],
    [-4,   4,   6,   4,   1],
    [ 1,   0,   0,   0,   0]
]) / 24.0


class MinSnapEval:
    '''
    Visual and simple computational 
    representation of snap minimization
    using clamped uniform B-Splines
    '''

    def __init__(self, num_control_points, degree):
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

        print("Start Block (tau=0):\n", B_d_M0)
        print("\nEnd Block (tau=1):\n", B_d_MM)


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
    # Q_d4_M = min_snap_evaluator.get_Q_matrix()


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


    