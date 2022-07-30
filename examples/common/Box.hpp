#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Hatrix {

class Box {
 public:
  // Store the dimension, diameter, center, begin and end indices of particles
  int64_t ndim, num_particles, begin_index, end_index;
  double diameter;
  std::string morton_index;
  // Each number in center corresponds to the x, y, and z coordinate
  std::vector<double> center;

  Box() = default;

  Box(const int64_t _num_particles, const int64_t _begin_index, const int64_t _end_index,
      const double _diameter, const double center_x, const std::string& _morton_index)
      : ndim(1), num_particles(_num_particles), begin_index(_begin_index),
        end_index(_end_index), diameter(_diameter), morton_index(_morton_index) {
    center.push_back(center_x);
  }

  Box(const int64_t _num_particles, const int64_t _begin_index, const int64_t _end_index,
      const double _diameter, const double center_x, const double center_y,
      const std::string& _morton_index)
      : ndim(2), num_particles(_num_particles), begin_index(_begin_index),
        end_index(_end_index), diameter(_diameter), morton_index(_morton_index) {
    center.push_back(center_x);
    center.push_back(center_y);
  }

  Box(const int64_t _num_particles, const int64_t _begin_index, const int64_t _end_index,
      const double _diameter, const double center_x, const double center_y,
      const double center_z, const std::string& _morton_index)
      : ndim(3), num_particles(_num_particles), begin_index(_begin_index),
        end_index(_end_index), diameter(_diameter), morton_index(_morton_index) {
    center.push_back(center_x);
    center.push_back(center_y);
    center.push_back(center_z);
  }

  double distance_from(const Box& other) const {
    double dist = 0;
    for (int64_t k = 0; k < ndim; k++) {
      dist += (center[k] - other.center[k]) *
              (center[k] - other.center[k]);
    }
    return dist;
  }
};

} // namespace Hatrix

