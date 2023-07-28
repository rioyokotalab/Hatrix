#pragma once

#include <vector>

#include "Particle.hpp"
#include "Hatrix/Hatrix.h"

namespace Hatrix {
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

    void
    search_tree_for_nodes(const Cell& tree,
                          const int64_t level_index,
                          const int64_t level,
                          int64_t &pstart, int64_t &pend) const;
  public:
    Domain(int64_t N, int64_t ndim);
    void generate_circular_particles(double min_val, double max_val);
    void generate_grid_particles();
    void read_col_file_3d(const std::string& geometry_file);
    void read_col_file_2d(const std::string& geometry_file);
    // Read this kind of file: https://open-babel.readthedocs.io/en/latest/FileFormats/XYZ_cartesian_coordinates_format.html
    void read_xyz_chemical_file(const std::string& geometry_file,
                                const int64_t num_electrons_per_atom);

    // Sort bodies using Hilbert curves. Specialized for ELSES blocks. Then generate a
    // tree with near and far blocks.
    void build_elses_tree(const int64_t molecule_size);

    void print_file(std::string file_name);

    // Build tree using co-oridinate sorting similar to exafmm. Uses the new Cell struct.
    void build_tree(const int64_t max_nleaf);

    int64_t cell_size(int64_t level_index, int64_t level) const;
  };
}
