#pragma once

#include <vector>
#include <string>

namespace Hatrix {
  class Box {
  public:
    double diameter;
    int64_t ndim, num_particles, start_index, stop_index;
    // Store the center, start and end co-ordinates of this box. Each number
    // in corresponds to the x, y, and z co-oridinate.
    std::string morton_index;
    std::vector<double> center;

    Box();
    Box(double _diameter, double center_x, int64_t _start_index, int64_t _stop_index,
        std::string _morton_index, int64_t _num_particles);
    Box(double diameter, double center_x, double center_y, double start_index,
        double stop_index, std::string _morton_index, int64_t num_particles);
    Box(double diameter, double center_x, double center_y, double center_z, double start_index,
        double stop_index, std::string _morton_index, int64_t num_particles);
    double distance_from(const Box& b) const;
  };
}
