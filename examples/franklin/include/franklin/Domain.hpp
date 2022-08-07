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
    int64_t start_index, end_index, level;
    uint32_t level_index;
    double radius;

    Cell(std::vector<double> _center, int64_t pstart, int64_t pend, double _radius);
    Cell();

    // print the structure of the tree.
    void print();
  };

  class Domain {
  public:
    std::vector<Hatrix::Particle> particles;
    Cell tree;
    int64_t N, ndim;

  private:
    void
    orb_split(Cell& cell,
              const int64_t pstart,
              const int64_t pend,
              const int64_t max_nleaf,
              const int64_t dim,
              const int64_t level,
              const uint32_t level_index);
  public:
    Domain(int64_t N, int64_t ndim);
    void generate_circular_particles(double min_val, double max_val);
    void generate_grid_particles();
    void read_col_file_3d(const std::string& geometry_file);
    void print_file(std::string file_name);

    // Build tree using co-oridinate sorting similar to exafmm. Uses the new Cell struct.
    void build_tree(const int64_t max_nleaf);
  };
}
