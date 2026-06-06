import open3d as o3d
import numpy as np

# Replace this with the name of the file you pulled from the folder
FILE_PATH = "/home/apd915/Documents/B-Splines/bspline_generator/Minimum-Snap-Trajectories/data/vicon_easy 1.ply"  

def test_lidar_ingestion(file_path, voxel_size=0.04):
    print(f"Loading 3D point cloud from: {file_path}")
    pcd = o3d.io.read_point_cloud(file_path)
    
    if pcd.is_empty():
        print("Error: Point cloud is empty. Check the file path.")
        return
        
    print(f"Original cloud contains: {len(pcd.points)} points.")

    # 1. THE DOWNSAMPLE (Crucial for A* performance)
    print(f"Downsampling to {voxel_size}m voxels...")
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # Paint it gray so the lighting looks clean in the visualizer
    downsampled_pcd.paint_uniform_color([0.5, 0.5, 0.5]) 
    
    print(f"Downsampled cloud contains: {len(downsampled_pcd.points)} points.")

    # 2. BRIDGING LOGIC FOR A* (Preparing the data)
    points = np.asarray(downsampled_pcd.points)
    
    # Convert physical meters into integer grid indices
    grid_indices = np.floor(points / voxel_size).astype(int)
    
    # Create the O(1) lookup set that your astar_sfc.py expects!
    occupied_voxels = set(tuple(idx) for idx in grid_indices)
    print(f"Generated {len(occupied_voxels)} unique occupied voxels for A*.")

    # 3. VISUALIZE
    print("Opening 3D viewer. Use your mouse to rotate and zoom!")
    # Press 'Q' or Esc to close the window
    o3d.visualization.draw_geometries([downsampled_pcd], window_name="Vicon Room 1 - LiDAR Scan")

if __name__ == "__main__":
    test_lidar_ingestion(FILE_PATH, voxel_size=0.2)