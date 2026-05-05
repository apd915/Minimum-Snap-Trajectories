import numpy as np
from scipy.optimize import minimize
from minvo_bounds import MINVO_STENCILS

# Added "degree" to the parameters
def run_qp_solver(objective_matrix, equality_constraints, inequality_constraints, initial_guess, A_eq, degree):
    """
    Splits the 3D minimum snap problem into decoupled X, Y, and Z optimizations.
    """
    W = objective_matrix
    SE = equality_constraints 
    D_vel, D_accel, V_max, A_max = inequality_constraints
    
    C_x_init = initial_guess[0, :]
    C_y_init = initial_guess[1, :]
    C_z_init = initial_guess[2, :]
    
    b_eq_x = SE[0, :]
    b_eq_y = SE[1, :]
    b_eq_z = SE[2, :]
    
    # Added A_eq as the second argument to all three calls!
    print("Optimizing X-axis...")
    C_x_opt = _optimize_single_axis(W, A_eq, b_eq_x, D_vel, D_accel, V_max, A_max, degree, MINVO_STENCILS, C_x_init)
    
    print("Optimizing Y-axis...")
    C_y_opt = _optimize_single_axis(W, A_eq, b_eq_y, D_vel, D_accel, V_max, A_max, degree, MINVO_STENCILS, C_y_init)
    
    print("Optimizing Z-axis...")
    C_z_opt = _optimize_single_axis(W, A_eq, b_eq_z, D_vel, D_accel, V_max, A_max, degree, MINVO_STENCILS, C_z_init)
    
    return np.vstack((C_x_opt, C_y_opt, C_z_opt))


def _optimize_single_axis(W, A_eq, b_eq, D_vel, D_accel, V_max, A_max, base_degree, minvo_stencils, C_init):
    """
    Executes the SciPy SLSQP solver for a single dimension.
    """
    # Grab the correct matrices for the derivatives
    F_vel = minvo_stencils[base_degree - 1]
    F_accel = minvo_stencils[base_degree - 2]

    # 1. The Objective Function 
    def cost_function(C_1D):
        return C_1D.T @ W @ C_1D

    # 2. Equality Constraints
    def equality_constraint(C_1D):
        return (A_eq @ C_1D) - b_eq 

    # 3. Inequality Constraints (MINVO Box Limits)
    def velocity_constraint(C_1D):
        # Get standard derivative control points
        V_ctrl = D_vel @ C_1D 
        
        # Apply MINVO transformation using the lowered degree
        V_minvo = apply_minvo_transform(V_ctrl, F_vel, base_degree - 1) 
        
        # Return Headroom
        return V_max - np.abs(V_minvo)
        
    def acceleration_constraint(C_1D):
        A_ctrl = D_accel @ C_1D
        
        # Apply MINVO transformation using the lowered degree
        A_minvo = apply_minvo_transform(A_ctrl, F_accel, base_degree - 2) 
        
        return A_max - np.abs(A_minvo)

    # 4. SciPy Configuration & Execution
    constraints = [
        {'type': 'eq', 'fun': equality_constraint},
        {'type': 'ineq', 'fun': velocity_constraint},
        {'type': 'ineq', 'fun': acceleration_constraint}
    ]
    
    result = minimize(
        fun=cost_function,
        x0=C_init,             
        method='SLSQP',        
        constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-6, 'disp': False}
    )
    
    if not result.success:
        raise ValueError(f"QP Solver Failed: {result.message}")
        
    return result.x


def apply_minvo_transform(ctrl_1D, F_matrix, current_degree):
    """
    Slides the MINVO transformation matrix across a 1D array of control points.
    Returns a flat array of all MINVO boundary points.
    """
    # Calculate how many segments we are evaluating
    num_segments = len(ctrl_1D) - current_degree
    
    # We will store the evaluated points here
    minvo_points = []
    
    for s in range(num_segments):
        # 1. Grab the sliding window for this specific segment
        window = ctrl_1D[s : s + current_degree + 1]
        
        # 2. Multiply by the MINVO matrix (F_matrix @ window)
        # Because we pre-transposed F, this works perfectly for 1D arrays too!
        V_local = F_matrix @ window
        
        # 3. Add these points to our master list
        minvo_points.extend(V_local)
        
    return np.array(minvo_points)