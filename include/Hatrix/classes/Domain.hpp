#pragma once

#include <vector>
#include <string>

#include "Particle.hpp"
#include "Cell.hpp"

namespace Hatrix {
  class Domain {
  public:
    // Number of points in this domain.
    int64_t N;

    // Number of dimensions of this domain.
    int64_t ndim;

    // Head node of the tree that makes up this domain.
    Cell tree;

    // Vector storing all the particles that exist in this domain.
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
    // Construct a domain where N denotes the number of points and ndim is the
    // number of dimension. ndim >=1 && ndim <= 3.
    Domain(int64_t N, int64_t ndim);

    // Recursively sort the points according to a simple cardinal sorting algorithm
    // that sorts according to the longest axis in every individual cell. The points
    // at the leaf level are not sorted.
    //
    // This function will result in the generation of a tree in the 'tree' variable.
    void cardinal_sort_and_cell_generation(const int64_t nleaf);
    void sector_sort(const int64_t nleaf);
    int64_t build_bottom_up_binary_tree(const int64_t molecule_size);

    // Generate points corresponding to a circular domain.
    //
    // If ndim == 1 :: generate a single line of equidistant points.
    // If ndim == 2 :: generate a unit circle with equidistant points
    //   on the circumference.
    // If ndim == 3 :: generate a unit sphere with equidistant points
    //   on the surface.
    void generate_circular_particles();

    // Generate points corresponding to a square grid domain.
    //
    // If ndim == 1 :: generate a single line of equidistant points.
    // If ndim == 2 :: generate a unit square with equidistant points on each side.
    // If ndim == 3 :: generate a unit cube with equidistant points on the surface.
    void generate_grid_particles();

    // Read geometry_file containing a 3D geometry. The file should not have any header
    // and each point should be on a new line in the following format:
    // x y z value
    //
    // A single space must separate each value. Only the first N points will be read.
    void read_col_file_3d(const std::string& geometry_file);

    // Read geometry_file containing a 2D geometry. The file should not have any header
    // and each point should be on a new line in the following format:
    // x y value
    //
    // A single space must separate each value. Only the first N points will be read.
    void read_col_file_2d(const std::string& geometry_file);

    // Read this kind of file:
    // https://open-babel.readthedocs.io/en/latest/FileFormats/XYZ_cartesian_coordinates_format.html
    void read_xyz_chemical_file(const std::string& geometry_file,
                                const int64_t num_electrons_per_atom);

    // Sort bodies using Hilbert curves. Specialized for ELSES blocks. Then generate a
    // tree with near and far blocks.
    int64_t build_elses_tree(const int64_t molecule_size);

    void sort_generic_geometry_particles(const int64_t start_index,
                                         const int64_t end_index,
                                         const int64_t nleaf);

    // Print the co-ordinates in the particles array to file file_name. Each
    // row corresponds to a space separated list of co-ordinates for each point.
    void print_file(std::string file_name);

    // Return the number of points at level 'level' for the cell corresponding to
    // the cell of index level_index.
    int64_t cell_size(int64_t level_index, int64_t level) const;
  };
}
