import numpy as np
from scipy.optimize import minimize
from core.minvo_bounds import MINVO_STENCILS

def run_qp_solver(objective_matrix, equality_constraints, inequality_constraints, initial_guess, A_eq, degree):
    """
    Executes a fully coupled 3D Minimum Snap optimization, enforcing 
    Safe Flight Corridors (SFCs), velocity, and acceleration limits.
    """
    W = objective_matrix
    SE = equality_constraints 
    
    # 1. Unpack our new 6-item tuple!
    D_vel, D_accel, V_max, A_max, A_sfc, b_sfc = inequality_constraints
    
    # Grab the correct MINVO transformation matrices
    F_vel = MINVO_STENCILS[degree - 1]
    F_accel = MINVO_STENCILS[degree - 2]
    
    # 2. Flatten the initial guess from (3, N) to interleaved (3N,)
    # Output format: [x0, y0, z0, x1, y1, z1, ...]
    C_init_flat = initial_guess.T.flatten()

    # ==========================================
    # CONSTRAINT DEFINITIONS
    # ==========================================

    def cost_function(C_flat):
        # Reshape back to (N, 3) to calculate costs per axis
        P = C_flat.reshape(-1, 3) 
        cost_x = P[:, 0].T @ W @ P[:, 0]
        cost_y = P[:, 1].T @ W @ P[:, 1]
        cost_z = P[:, 2].T @ W @ P[:, 2]
        return cost_x + cost_y + cost_z

    def equality_constraint(C_flat):
        # P is (N, 3). A_eq is (6, N). 
        # A_eq @ P gives a (6, 3) matrix. SE is (3, 6), so we transpose it to match.
        P = C_flat.reshape(-1, 3)
        return (A_eq @ P - SE.T).flatten()

    def sfc_constraint(C_flat):
        # SciPy requires inequalities to be formulated as >= 0
        # Math: A * x <= b   -->   b - A * x >= 0
        return b_sfc - (A_sfc @ C_flat)

    def velocity_constraint(C_flat):
        P = C_flat.reshape(-1, 3)
        
        # Apply MINVO to each axis
        V_minvo_x = apply_minvo_transform(D_vel @ P[:, 0], F_vel, degree - 1)
        V_minvo_y = apply_minvo_transform(D_vel @ P[:, 1], F_vel, degree - 1)
        V_minvo_z = apply_minvo_transform(D_vel @ P[:, 2], F_vel, degree - 1)
        
        # Return headroom: Max allowable - actual
        cx = V_max - np.abs(V_minvo_x)
        cy = V_max - np.abs(V_minvo_y)
        cz = V_max - np.abs(V_minvo_z)
        
        # Concatenate into one massive array of constraints
        return np.concatenate((cx, cy, cz))
        
    def acceleration_constraint(C_flat):
        P = C_flat.reshape(-1, 3)
        
        A_minvo_x = apply_minvo_transform(D_accel @ P[:, 0], F_accel, degree - 2)
        A_minvo_y = apply_minvo_transform(D_accel @ P[:, 1], F_accel, degree - 2)
        A_minvo_z = apply_minvo_transform(D_accel @ P[:, 2], F_accel, degree - 2)
        
        cx = A_max - np.abs(A_minvo_x)
        cy = A_max - np.abs(A_minvo_y)
        cz = A_max - np.abs(A_minvo_z)
        return np.concatenate((cx, cy, cz))

    # ==========================================
    # SOLVER EXECUTION
    # ==========================================
    
    constraints = [
        {'type': 'eq', 'fun': equality_constraint},
        {'type': 'ineq', 'fun': sfc_constraint},       # Added the Glass Walls!
        {'type': 'ineq', 'fun': velocity_constraint},
        {'type': 'ineq', 'fun': acceleration_constraint}
    ]
    
    print("Executing Coupled 3D Optimization...")
    result = minimize(
        fun=cost_function,
        x0=C_init_flat,             
        method='SLSQP',        
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-6, 'disp': True}
    )
    
    if not result.success:
        raise ValueError(f"QP Solver Failed: {result.message}")
        
    # Reshape back to the original (3, N) format for plotting and trajectory logic
    return result.x.reshape(-1, 3).T


def apply_minvo_transform(ctrl_1D, F_matrix, current_degree):
    """
    Slides the MINVO transformation matrix across a 1D array of control points.
    Returns a flat array of all MINVO boundary points.
    """
    num_segments = len(ctrl_1D) - current_degree
    minvo_points = []
    
    for s in range(num_segments):
        window = ctrl_1D[s : s + current_degree + 1]
        V_local = F_matrix @ window
        minvo_points.extend(V_local)
        
    return np.array(minvo_points)