#pragma once

#include <vector>

#include "Particle.hpp"
#include "Box.hpp"

#include "Hatrix/Hatrix.h"

namespace Hatrix {
  // Hierarchy of sorted particles.
  class Cell {
  public:
    std::vector<Cell> cells;
    std::vector<double> center;
    int64_t start_index, end_index;
    double radius;

    Cell(std::vector<double> _center, int64_t pstart, int64_t pend, double _radius);
    Cell() = delete;
  };

  class Domain {
  public:
    std::vector<Hatrix::Particle> particles;
    std::vector<Hatrix::Box> boxes;
    int64_t N, ndim;
    Cell * tree;

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
    // The direction parameter determines whether the copying of data is happening into
    // bodies or buffer array.
    void split_cell(Cell* cell, int64_t start_index, int64_t end_index,
                    const int64_t max_nleaf,
                    std::vector<Hatrix::Particle>& bodies,
                    std::vector<Hatrix::Particle>& buffer,
                    bool direction=false);
    int
    get_quadrant(std::vector<double>& p_coords,
                 std::vector<double>& c_coords);
  public:
    Domain(int64_t N, int64_t ndim);
    void generate_circular_particles(double min_val, double max_val);
    void divide_domain_and_create_particle_boxes(int64_t nleaf);
    void generate_grid_particles();
    void read_col_file_3d(const std::string& geometry_file);
    void print_file(std::string file_name);

    // Build tree using co-oridinate sorting similar to exafmm. Uses the new Cell struct.
    void build_tree(const int64_t max_nleaf);
  };
}
