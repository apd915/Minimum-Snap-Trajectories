
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from minvo_bounds import MINVO_STENCILS
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
