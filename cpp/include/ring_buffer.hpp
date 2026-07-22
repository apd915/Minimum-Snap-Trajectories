#pragma once

#include <vector>
#include <cstdint>
#include <cstring>

namespace trajectory_planner {
namespace mapping {

// ==========================================
// RingBufferGrid (Port of ring_buffer.py)
// ==========================================
// A dense, fixed-size 3D array for O(1) local collision checks.
// Uses modulo arithmetic to map global voxel indices into the buffer.
// Backed by a flat std::vector<uint8_t> for maximum CPU cache locality.
//
// Values: 0 = Free/Unknown, 1 = Occupied, 2 = Inflated Buffer
class RingBufferGrid {
public:
    RingBufferGrid(int size_x, int size_y, int size_z)
        : size_x_(size_x), size_y_(size_y), size_z_(size_z),
          grid_(static_cast<std::size_t>(size_x) * size_y * size_z, 0) {}

    /**
     * O(1) inline lookup mapping a global voxel index to the local ring buffer.
     * Uses modulo to wrap coordinates into the fixed-size array.
     */
    inline bool is_occupied(int x, int y, int z) const {
        // C++ modulo can return negative for negative inputs, so we fix it
        int rx = ((x % size_x_) + size_x_) % size_x_;
        int ry = ((y % size_y_) + size_y_) % size_y_;
        int rz = ((z % size_z_) + size_z_) % size_z_;
        return grid_[static_cast<std::size_t>(rx) * size_y_ * size_z_ + ry * size_z_ + rz] > 0;
    }

    /**
     * Sets a specific global coordinate as occupied (or a given state) in the buffer.
     */
    inline void set_occupied(int x, int y, int z, uint8_t state = 1) {
        int rx = ((x % size_x_) + size_x_) % size_x_;
        int ry = ((y % size_y_) + size_y_) % size_y_;
        int rz = ((z % size_z_) + size_z_) % size_z_;
        grid_[static_cast<std::size_t>(rx) * size_y_ * size_z_ + ry * size_z_ + rz] = state;
    }

    /**
     * Clears the entire buffer to 0 (free space). Used when shifting the origin.
     */
    void clear() {
        std::memset(grid_.data(), 0, grid_.size());
    }

    int size_x() const { return size_x_; }
    int size_y() const { return size_y_; }
    int size_z() const { return size_z_; }
    std::size_t total_voxels() const { return grid_.size(); }
    std::size_t memory_bytes() const { return grid_.size() * sizeof(uint8_t); }

private:
    int size_x_, size_y_, size_z_;
    std::vector<uint8_t> grid_;
};

} // namespace mapping
} // namespace trajectory_planner
