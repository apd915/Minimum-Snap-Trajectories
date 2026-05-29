import numpy as np
from scipy.optimize import minimize
import scipy.sparse as sparse
import osqp
from core.minvo_bounds_clamped import MINVO_CLAMPED_STENCILS
from core.minvo_bounds import MINVO_STENCILS


def build_minvo_sparse_matrix(D_matrix, F_matrix, current_degree):
    """
    Creates a massive sparse matrix that mathematically replicates a sliding window.
    When multiplied by the control points, it applies the MINVO transformation 
    simultaneously across the entire trajectory.
    """
    num_deriv_points = D_matrix.shape[0]
    num_windows = num_deriv_points - current_degree
    num_minvo_points_per_window = F_matrix.shape[0]
    
    rows = []
    cols = []
    data = []
    
    row_idx = 0
    # Replicate the "sliding window" by shifting the column index
    for s in range(num_windows):
        for i in range(num_minvo_points_per_window):
            for j in range(current_degree + 1):
                val = F_matrix[i, j]
                if val != 0:
                    rows.append(row_idx)
                    cols.append(s + j)  # <--- This is the shift!
                    data.append(val)
            row_idx += 1
            
    # Assemble the Sliding Window matrix
    M_window = sparse.coo_matrix((data, (rows, cols)), shape=(row_idx, num_deriv_points)).tocsc()
    
    # Multiply it by the Derivative matrix to get the final MINVO Constraint Matrix
    A_minvo = M_window @ sparse.csc_matrix(D_matrix)
    return A_minvo


def build_minvo_kinodynamic_matrix(D_matrix, current_degree, spline_type="clamped"):
    """
    Compiles the massive sparse matrix that applies MINVO kinematics simultaneously across the trajectory.
    Routes logic based on whether the spline is "natural" (sliding window) or "clamped" (boundary distorted).
    """
    num_deriv_points = D_matrix.shape[0]
    total_spans = num_deriv_points - current_degree
    
    rows = []
    cols = []
    data = []
    row_idx = 0
    
    # -----------------------------------------------------
    # PATH A: The Clamped Stencil Logic (MADER)
    # -----------------------------------------------------
    if spline_type == "clamped":
        stencils = MINVO_CLAMPED_STENCILS[current_degree]
        
        for span_idx in range(total_spans):
            # 1. Grab the correct stencil based on physical position
            if span_idx < current_degree:
                F_matrix = stencils[f'start_{span_idx}']
            elif span_idx >= total_spans - current_degree:
                end_idx = total_spans - 1 - span_idx
                F_matrix = stencils[f'end_{end_idx}']
            else:
                F_matrix = stencils['interior']
                
            # 2. Apply it to the sparse mapping
            num_minvo_pts = F_matrix.shape[0]
            for i in range(num_minvo_pts):
                for j in range(current_degree + 1):
                    val = F_matrix[i, j]
                    if abs(val) > 1e-9:
                        rows.append(row_idx)
                        cols.append(span_idx + j) 
                        data.append(val)
                row_idx += 1
                
    # -----------------------------------------------------
    # PATH B: The Natural Stencil Logic (Standard Sliding Window)
    # -----------------------------------------------------
    elif spline_type == "natural":
        F_matrix = MINVO_STENCILS[current_degree]
        num_minvo_pts = F_matrix.shape[0]
        
        for span_idx in range(total_spans):
            for i in range(num_minvo_pts):
                for j in range(current_degree + 1):
                    val = F_matrix[i, j]
                    if abs(val) > 1e-9:
                        rows.append(row_idx)
                        cols.append(span_idx + j)
                        data.append(val)
                row_idx += 1
                
    else:
        raise ValueError("spline_type must be 'clamped' or 'natural'")
        
    # Assemble the Mapping matrix
    M_mapping = sparse.coo_matrix((data, (rows, cols)), shape=(row_idx, num_deriv_points)).tocsc()
    
    # Multiply it by the Calculus Derivative matrix to get the final Kinodynamic Bounds!
    A_minvo = M_mapping @ sparse.csc_matrix(D_matrix)
    return A_minvo


def run_qp_solver(objective_matrix, equality_constraints, inequality_constraints, A_eq, degree, spline_type="clamped", use_minvo=True):
    """
    Executes a lightning-fast OSQP 3D Minimum Snap optimization.
    Enforces Boundary Conditions, Safe Flight Corridors (SFCs), and Kinodynamic Limits.
    """
    # OSQP strictly requires Compressed Sparse Column (CSC) format for speed
    W = sparse.csc_matrix(objective_matrix)
    SE = equality_constraints 
    
    # Unpack our 6-item inequality tuple
    D_vel, D_accel, V_max, A_max, A_sfc, b_sfc = inequality_constraints
    N = W.shape[0]

    # ==========================================
    # 1. OBJECTIVE MATRICES (P and q)
    # ==========================================
    # Our variables are interleaved: [x0, y0, z0, x1, y1, z1...]
    # We use a Kronecker product to magically expand our 1D W matrix into 3D!
    P_3D = sparse.kron(W, sparse.eye(3)).tocsc()
    
    # OSQP minimizes (1/2 * x^T * P * x). Since our cost is (x^T * W * x), 
    # we must multiply by 2 to balance the equation.
    P_osqp = 2 * P_3D
    q_osqp = np.zeros(3 * N)

    # ==========================================
    # 2. EQUALITY CONSTRAINTS (Boundary Conditions)
    # ==========================================
    # Expand A_eq to 3D just like we did for W
    A_eq_3D = sparse.kron(A_eq, sparse.eye(3)).tocsc()
    
    # Flatten the Start/End matrix. For equalities, lower bound == upper bound.
    l_eq = SE.T.flatten()
    u_eq = SE.T.flatten()

    # ==========================================
    # 3. GEOMETRIC CONSTRAINTS (Safe Flight Corridors)
    # ==========================================
    # Math: A_sfc @ P <= b_sfc
    A_sfc_sparse = sparse.csc_matrix(A_sfc)
    
    # Lower bound is negative infinity, upper bound is the glass walls (b_sfc)
    l_sfc = np.full(b_sfc.shape, -np.inf) 
    u_sfc = b_sfc

    # ==========================================
    # 4. KINODYNAMIC CONSTRAINTS (Toggle MINVO vs Standard)
    # ==========================================
    # if use_minvo:
    #     print("[OSQP] Formatting Matrices with MINVO Kinodynamic Bounds...")
    #     # Grab the correct MINVO transformation matrices
    #     F_vel = MINVO_STENCILS[degree - 1]
    #     F_accel = MINVO_STENCILS[degree - 2]

    #     # Create the base MINVO constraint matrices for 1D
    #     A_vel_1D = build_minvo_sparse_matrix(D_vel, F_vel, degree - 1)
    #     A_accel_1D = build_minvo_sparse_matrix(D_accel, F_accel, degree - 2)
    # else:
    #     print("[OSQP] Formatting Matrices with Standard Convex Hull Bounds...")
    #     # Fall back to raw derivative control points without the sliding window
    #     A_vel_1D = sparse.csc_matrix(D_vel)
    #     A_accel_1D = sparse.csc_matrix(D_accel)
    if use_minvo:
        print(f"[OSQP] Formatting Matrices with {spline_type.upper()} MINVO Kinodynamic Bounds...")
        # Use the new smart compiler!
        A_vel_1D = build_minvo_kinodynamic_matrix(D_vel, degree - 1, spline_type)
        A_accel_1D = build_minvo_kinodynamic_matrix(D_accel, degree - 2, spline_type)

    # Expand them to 3D (X, Y, Z)
    A_vel_3D = sparse.kron(A_vel_1D, sparse.eye(3)).tocsc()
    A_accel_3D = sparse.kron(A_accel_1D, sparse.eye(3)).tocsc()

    # Absolute bounds for Velocity: -V_max <= V <= V_max
    # These dynamically size themselves to match whatever matrix was generated above!
    l_vel = np.full(A_vel_3D.shape[0], -V_max)
    u_vel = np.full(A_vel_3D.shape[0], V_max)

    # Absolute bounds for Acceleration: -A_max <= A <= A_max
    l_accel = np.full(A_accel_3D.shape[0], -A_max)
    u_accel = np.full(A_accel_3D.shape[0], A_max)

    # ==========================================
    # 5. ASSEMBLE THE MASTER MATRICES
    # ==========================================
    # Stack the matrices vertically, and the bound vectors horizontally
    A_osqp = sparse.vstack([A_eq_3D, A_sfc_sparse, A_vel_3D, A_accel_3D]).tocsc()
    l_osqp = np.hstack([l_eq, l_sfc, l_vel, l_accel])
    u_osqp = np.hstack([u_eq, u_sfc, u_vel, u_accel])

    # ==========================================
    # 6. EXECUTE OSQP
    # ==========================================
    prob = osqp.OSQP()
    
    # Setup the problem. 'verbose=True' will show you the exact solve time!
    prob.setup(P=P_osqp, q=q_osqp, A=A_osqp, l=l_osqp, u=u_osqp, verbose=True)
    
    result = prob.solve()
    
    # OSQP returns status_val == 1 if successful, or 2 if solved but inaccurate
    if result.info.status_val not in [1, 2]:
        raise ValueError(f"OSQP Failed: {result.info.status}")
        
    # Reshape back to the original (3, N) format for plotting and trajectory logic
    return result.x.reshape(-1, 3).T