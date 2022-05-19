#pragma once

#include <vector>

#include "Particle.hpp"
#include "Box.hpp"

#include "Hatrix/Hatrix.h"

namespace Hatrix {
  class Domain {
  public:
    std::vector<Hatrix::Particle> particles;
    std::vector<Hatrix::Box> boxes;
    int64_t N, ndim;
  private:
    // https://www.csd.uwo.ca/~mmorenom/cs2101a_moreno/Barnes-Hut_Algorithm.pdf
    void orthogonal_recursive_bisection_1dim(int64_t start, int64_t end,
                                             std::string morton_index, int64_t nleaf);
    void orthogonal_recursive_bisection_2dim(int64_t start, int64_t end,
                                             std::string morton_index, int64_t nleaf,
                                             int64_t axis);
    void orthogonal_recursive_bisection_3dim(int64_t start, int64_t end,
                                             std::string morton_index,
                                             int64_t nleaf, int64_t axis);
  public:
    Domain(int64_t N, int64_t ndim);
    void generate_circular_particles(double min_val, double max_val);
    void divide_domain_and_create_particle_boxes(int64_t nleaf);
    void generate_grid_particles();
    void print_file(std::string file_name);
  };
}
