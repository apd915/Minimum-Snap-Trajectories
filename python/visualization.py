
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
import numpy as np

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
