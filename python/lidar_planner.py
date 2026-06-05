import open3d as o3d
import plotly.graph_objects as go
import numpy as np
import time
from planning.astar_sfc import AStar_SFC_Planner

FILE_PATH = "/home/apd915/Minimum-Snap-Trajectories/data/vicon_easy.ply"  

# 1. MOCK VOXEL GRID CLASS
# Your A* expects an object with these two specific attributes
class VoxelGridMap:
    def __init__(self, resolution, occupied_set):
        self.voxel_resolution = resolution
        self.occupied_voxels_inflated = occupied_set

def run_real_lidar_astar():
    voxel_size = 0.04
    
    # --- 1. INGESTION ---
    pcd = o3d.io.read_point_cloud(FILE_PATH)
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    points = np.asarray(downsampled_pcd.points)
    
    grid_indices = np.floor(points / voxel_size).astype(int)
    occupied_voxels = set(tuple(idx) for idx in grid_indices)
    print(f"Map ingested: {len(occupied_voxels)} occupied voxels.\n")

    # --- 2. CALCULATE MAP BOUNDS ---
    # Find the physical walls of the Vicon Room to constrain A*
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    bounds = [
        [min_bound[0], max_bound[0]],
        [min_bound[1], max_bound[1]],
        [min_bound[2], max_bound[2]]
    ]
    
    print(f"Map Bounds (Meters):")
    print(f"X: {bounds[0][0]:.2f} to {bounds[0][1]:.2f}")
    print(f"Y: {bounds[1][0]:.2f} to {bounds[1][1]:.2f}")
    print(f"Z: {bounds[2][0]:.2f} to {bounds[2][1]:.2f}\n")

    # --- 3. RUN A* PLANNER ---
    voxel_map = VoxelGridMap(voxel_size, occupied_voxels)
    planner = AStar_SFC_Planner(voxel_map, bounds)

    # Pick a safe start and goal inside the Vicon Room
    # (We keep Z at 1.0 meters to fly over floor clutter)
    start_pos = (0.0, 0.0, 1.0) 
    goal_pos = (0.0, 13.0, 1.0)   

    print(f"Routing from {start_pos} to {goal_pos}...")
    
    start_time = time.perf_counter()
    path = planner.search(start_pos, goal_pos)
    end_time = time.perf_counter()

    if path:
        print(f"\n[ SUCCESS ] Path found in {(end_time - start_time)*1000:.2f} ms!")
        print(f"Waypoints: {len(path)}")
        for wp in path[:5]:
            print(f"  -> {wp}")
        print("  -> ...")
    else:
        print("\n[ FAILED ] No path found. Start/Goal might be inside an obstacle!")

    return occupied_voxels, path


def visualize_path_plotly(occupied_voxels, path):
    print("Generating interactive 3D visualization in your browser...")
    
    # 1. Extract obstacle coordinates
    obs_x, obs_y, obs_z = [], [], []
    for voxel in occupied_voxels:
        # Multiply by your voxel size (0.2) to scale it back to meters
        obs_x.append(voxel[0] * 0.2)
        obs_y.append(voxel[1] * 0.2)
        obs_z.append(voxel[2] * 0.2)
        
    # 2. Extract path coordinates
    path_x, path_y, path_z = [], [], []
    if path:
        for p in path:
            path_x.append(p[0])
            path_y.append(p[1])
            path_z.append(p[2])

    # 3. Create the 3D Scatter plot for obstacles (The Vicon Room)
    obstacles_trace = go.Scatter3d(
        x=obs_x, y=obs_y, z=obs_z,
        mode='markers',
        marker=dict(size=2, color='gray', opacity=0.5),
        name='LiDAR Obstacles'
    )

    # 4. Create the 3D Scatter plot for the A* Path
    path_trace = go.Scatter3d(
        x=path_x, y=path_y, z=path_z,
        mode='lines+markers',
        marker=dict(size=5, color='red'),
        line=dict(color='green', width=5),
        name='A* Trajectory'
    )

    # 5. Render the figure
    fig = go.Figure(data=[obstacles_trace, path_trace])
    fig.update_layout(
        title="Vicon Room 1 - LiDAR Map & A* Path",
        scene=dict(aspectmode='data') # Keeps the 3D scale realistic 1:1:1
    )
    
    # Save the interactive map as a webpage instead of trying to auto-open it
    output_file = "vicon_trajectory.html"
    fig.write_html(output_file)
    print(f"Saved successfully! Open '{output_file}' in your Windows File Explorer to view.")

if __name__ == "__main__":
    occupied_voxels, path = run_real_lidar_astar()
    visualize_path_plotly(occupied_voxels, path)