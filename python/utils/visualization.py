
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from core.minvo_bounds import MINVO_STENCILS
import numpy as np

# ==========================================
# UTILITY & VISUALIZATION
# ==========================================
def plot_trajectory(ctrl_pts, knots, degree, minvo_stencils=None):
    """
    Evaluates and plots the 3D Minimum Snap Trajectory and its control polygon.
    """
    pts = ctrl_pts.T 
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Plot the Control Polygon (The mathematical "gravitational anchors")
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'ro--', alpha=0.4, label='Control Polygon')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='red', s=30)

    # 2. Plot the MINVO Polygons (The tight "shrink-wrap")
    if MINVO_STENCILS is not None and degree in MINVO_STENCILS:
        F = minvo_stencils[degree]
        # Calculate how many segments make up this flight path
        num_segments = len(pts) - degree
        
        # Define a list of colors to cycle through
        colors = ['g', 'c', 'm', 'y', 'orange'] 
        
        # Slide the window across each segment!
        for s in range(num_segments):
            C_local = pts[s : s + degree + 1]
            V_local = F @ C_local
            V_closed = np.vstack((V_local, V_local[0]))
            
            # Pick a color based on the segment index
            c = colors[s % len(colors)]
            
            label = 'MINVO Bounds' if s == 0 else ""
            
            # Plot using the cycling color!
            ax.plot(V_closed[:, 0], V_closed[:, 1], V_closed[:, 2], 
                    color=c, linestyle='--', marker='.', alpha=0.6, linewidth=1.5, label=label)
            # ax.scatter(V_local[:, 0], V_local[:, 1], V_local[:, 2], c='green', s=20)
    
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


def plot_kinematics(C_p, knots, degree, V_max, A_max):
    """
    Evaluates the continuous velocity and acceleration profiles and plots 
    them against the physical solver limits.
    """
    pts = C_p.T  # Transpose to N x 3 for SciPy
    
    # 1. Construct the continuous position spline
    pos_spline = BSpline(knots, pts, degree)
    
    # 2. Extract the analytical derivative splines
    vel_spline = pos_spline.derivative(nu=1)
    acc_spline = pos_spline.derivative(nu=2)
    
    # 3. Evaluate smoothly over the flight time
    t_smooth = np.linspace(knots[degree], knots[-degree-1], 500)
    velocities = vel_spline(t_smooth)
    accelerations = acc_spline(t_smooth)
    
    # 4. Visualization Setup
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # --- VELOCITY PLOT ---
    ax1.plot(t_smooth, velocities[:, 0], 'r-', linewidth=2, label='Vx')
    ax1.plot(t_smooth, velocities[:, 1], 'g-', linewidth=2, label='Vy')
    ax1.plot(t_smooth, velocities[:, 2], 'b-', linewidth=2, label='Vz')
    
    # Draw the physical constraints
    ax1.axhline(V_max, color='k', linestyle='--', linewidth=2, label=f'Limit (±{V_max} m/s)')
    ax1.axhline(-V_max, color='k', linestyle='--', linewidth=2)
    
    ax1.set_title('Velocity Profiles vs. Physical Limits', fontweight='bold')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.4)
    
    # --- ACCELERATION PLOT ---
    ax2.plot(t_smooth, accelerations[:, 0], 'r-', linewidth=2, label='Ax')
    ax2.plot(t_smooth, accelerations[:, 1], 'g-', linewidth=2, label='Ay')
    ax2.plot(t_smooth, accelerations[:, 2], 'b-', linewidth=2, label='Az')
    
    # Draw the physical constraints
    ax2.axhline(A_max, color='k', linestyle='--', linewidth=2, label=f'Limit (±{A_max} m/s²)')
    ax2.axhline(-A_max, color='k', linestyle='--', linewidth=2)
    
    ax2.set_title('Acceleration Profiles vs. Physical Limits', fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Acceleration (m/s²)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.4)
    
    plt.tight_layout()
    plt.show()
