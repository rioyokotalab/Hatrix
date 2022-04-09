#pragma once

#include <vector>

#include "Particle.hpp"
#include "Box.hpp"

namespace Hatrix {
  class Domain {
  public:
    std::vector<Hatrix::Particle> particles;
    std::vector<Hatrix::Box> boxes;
    int N, ndim;
  private:
    // https://www.csd.uwo.ca/~mmorenom/cs2101a_moreno/Barnes-Hut_Algorithm.pdf
    void orthogonal_recursive_bisection_1dim(int start, int end,
                                             std::string morton_index, int nleaf);
    void orthogonal_recursive_bisection_2dim(int start, int end,
                                             std::string morton_index, int nleaf,
                                             int axis);
    void orthogonal_recursive_bisection_3dim(int start, int end,
                                             std::string morton_index,
                                             int nleaf, int axis);
  public:
    Domain(int N, int ndim);
    void generate_particles(double min_val, double max_val);
    void divide_domain_and_create_particle_boxes(int nleaf);
    void generate_starsh_grid_particles();
    void generate_starsh_electrodynamics_particles();
    void print_file(std::string file_name);
    Matrix generate_rank_heat_map() const;
  };
}
