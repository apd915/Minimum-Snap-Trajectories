import numpy as np

class RingBufferGrid:
    """
    A dense, fixed-size 3D array used for blazing fast local A* planning.
    Uses modulo arithmetic to map global coordinates to the local buffer.
    """
    def __init__(self, size_x: int, size_y: int, size_z: int):
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        
        # 0 = Free/Unknown, 1 = Occupied, 2 = Inflated Buffer
        self.grid = np.zeros((size_x, size_y, size_z), dtype=np.uint8)
        self.global_origin = (0, 0, 0)
        
    def is_occupied(self, x: int, y: int, z: int) -> bool:
        """
        O(1) lookup mapping global index to local grid index.
        """
        rx = x % self.size_x
        ry = y % self.size_y
        rz = z % self.size_z
        
        return self.grid[rx, ry, rz] > 0
        
    def set_occupied(self, x: int, y: int, z: int, state: int = 1):
        """
        Sets a specific global coordinate as occupied in the ring buffer.
        """
        rx = x % self.size_x
        ry = y % self.size_y
        rz = z % self.size_z
        
        self.grid[rx, ry, rz] = state
