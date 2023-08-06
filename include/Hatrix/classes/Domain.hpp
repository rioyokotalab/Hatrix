#pragma once

#include <vector>

#include "Particle.hpp"
#include "Hatrix/Hatrix.h"

namespace Hatrix {
  class Domain {
  public:
    int64_t N, ndim;
    Cell tree;

    std::vector<Hatrix::Particle> particles;
    std::vector<Cell> tree_list;
    std::vector<double> Xmin, Xmax; // store the min and max co-ordinates of the whole domain.

  private:

    int64_t level_offset(int64_t level);
    int64_t get_hilbert_id_child(int64_t hilbert_key);
    int64_t get_level(int64_t hilbert_key);

    std::vector<int64_t> int_index_3d(const std::vector<double>& X,
                                      const std::vector<Cell>& cell_list,
                                      const int64_t level);
    int64_t hilbert_index(std::vector<int64_t>& iX, const int64_t level,
                          const bool offset = true);

    void
    search_tree_for_nodes(const Cell& tree,
                          const int64_t level_index,
                          const int64_t level,
                          int64_t &pstart, int64_t &pend) const;
    double get_axis_min(int64_t start_index, int64_t end_index, int64_t axis);
    double get_axis_max(int64_t start_index, int64_t end_index, int64_t axis);
    void sort_elses_bodies(const int64_t molecule_size);
    void sort_particles_and_build_tree(Particle *buffer, Particle* bodies,
                                       int64_t start_index, int64_t end_index,
                                       int64_t cell_list_index, std::vector<Cell>& cell_list,
                                       int64_t nleaf, int64_t level, bool direction);
  public:
    Domain(int64_t N, int64_t ndim);
    int64_t build_bottom_up_binary_tree(const int64_t molecule_size);

    void generate_circular_particles(double min_val, double max_val);
    void generate_grid_particles();
    void read_col_file_3d(const std::string& geometry_file);
    void read_col_file_2d(const std::string& geometry_file);
    // Read this kind of file: https://open-babel.readthedocs.io/en/latest/FileFormats/XYZ_cartesian_coordinates_format.html
    void read_xyz_chemical_file(const std::string& geometry_file,
                                const int64_t num_electrons_per_atom);

    // Sort bodies using Hilbert curves. Specialized for ELSES blocks. Then generate a
    // tree with near and far blocks.
    int64_t build_elses_tree(const int64_t molecule_size);

    int64_t sort_generic_geometry_particles(const int64_t nleaf);

    void print_file(std::string file_name);

    int64_t cell_size(int64_t level_index, int64_t level) const;
  };
}
