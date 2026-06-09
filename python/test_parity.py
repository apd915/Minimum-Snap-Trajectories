from planning.astar_sfc import AStar_SFC_Planner
import time

class MockVoxelGrid:
    def __init__(self):
        self.voxel_resolution = 1.0
        self.occupied_voxels_inflated = set()
        self.continuous_inflated_bounds = []

# 1. Create a 10x10x10 environment
grid = MockVoxelGrid()

# 2. Build a wall blocking the direct path (at X=2, blocking Y=0->5, Z=0->5)
for y in range(6):
    for z in range(6):
        grid.occupied_voxels_inflated.add((2, y, z))

# 3. Bounds formatted for your Python class ([x_min, x_max], [y_min, y_max], [z_min, z_max])
bounds = [[0, 10], [0, 10], [0, 10]]

planner = AStar_SFC_Planner(grid, bounds)

print("--- PYTHON BENCHMARK ---")

# 1. Benchmark the Search Phase
start_time = time.perf_counter()
path = planner.search((0, 0, 0), (5, 5, 5))
search_time = (time.perf_counter() - start_time) * 1_000_000  # Convert to microseconds

print(f"Raw Path Length: {len(path)}")

# 2. Benchmark the Smoothing Phase
start_time = time.perf_counter()
smoothed = planner.sfc_smoother()
smooth_time = (time.perf_counter() - start_time) * 1_000_000  # Convert to microseconds

print(f"Smoothed Waypoints: {smoothed}")
print("-" * 25)
print(f"[ PYTHON ] Search Time: {search_time:.2f} µs")
print(f"[ PYTHON ] Smooth Time: {smooth_time:.2f} µs")
print("-" * 25)
