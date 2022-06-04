#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <tuple>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cassert>
#include <random>
#include <string>
#include <iomanip>
#include <functional>
#include <fstream>
#include <chrono>

#include "Hatrix/Hatrix.h"

using vec = std::vector<int64_t>;

// Construction of BLR2 strong admis matrix based on geometry based admis condition.
enum MATRIX_TYPES {BLR2_MATRIX=0, H2_MATRIX=1};
double PV = 1e-3;

namespace Hatrix {

std::function<double(const std::vector<double>& coords_row,
                     const std::vector<double>& coords_col)> kernel_function;

class Particle {
 public:
  double value;
  std::vector<double> coords;

  Particle(double x, double _value);
  Particle(double x, double y, double _value);
  Particle(double x, double y, double z, double _value);
  Particle(std::vector<double> coords, double _value);
};

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
                                           std::string morton_index, int64_t nleaf, int64_t axis);
  void orthogonal_recursive_bisection_3dim(int64_t start, int64_t end, std::string morton_index,
                                           int64_t nleaf, int64_t axis);
 public:
  Domain(int64_t N, int64_t ndim);
  void generate_particles(double min_val, double max_val);
  void divide_domain_and_create_particle_boxes(int64_t nleaf);
  void generate_starsh_grid_particles();
  void generate_starsh_electrodynamics_particles();
  void print_file(std::string file_name);
  Matrix generate_rank_heat_map();
};

class H2 {
 public:
  int64_t N, nleaf, n_blocks, lr_max_rank;
  double accuracy, admis;
  std::string admis_kind;
  int64_t matrix_type;
  int64_t height;
  RowLevelMap U;
  ColLevelMap V;
  RowColLevelMap<Matrix> D, S;
  RowColLevelMap<bool> is_admissible;
  std::vector<int64_t> level_blocks;
  int64_t min_rank, max_rank;

 private:
  int64_t find_all_dense_row();
  void coarsen_blocks(int64_t level);

  int64_t geometry_admis_non_leaf(int64_t nblocks, int64_t level);
  int64_t calc_geometry_based_admissibility(const Domain& domain);
  void calc_diagonal_based_admissibility(int64_t level);

  int64_t get_block_size_row(const Domain& domain, int64_t parent, int64_t level);
  int64_t get_block_size_col(const Domain& domain, int64_t parent, int64_t level);
  bool row_has_admissible_blocks(int64_t row, int64_t level);
  bool col_has_admissible_blocks(int64_t col, int64_t level);
  Matrix generate_block_row(int64_t block, int64_t block_size,
                            const Domain& domain, int64_t level,
                            const Matrix& rand);
  std::tuple<Matrix, Matrix>
  generate_row_cluster_bases(int64_t block, int64_t block_size,
                             const Domain& domain, int64_t level,
                             const Matrix& rand);
  Matrix generate_block_column(int64_t block, int64_t block_size,
                               const Domain& domain, int64_t level,
                               const Matrix& rand);
  std::tuple<Matrix, Matrix>
  generate_column_cluster_bases(int64_t block, int64_t block_size,
                                const Domain& domain, int64_t level,
                                const Matrix& rand);
  void generate_leaf_nodes(const Domain& domain, const Matrix& rand);

  std::tuple<Matrix, Matrix>
  generate_U_transfer_matrix(Matrix& Ubig_child1, Matrix& Ubig_child2, int64_t node,
                             int64_t block_size, const Domain& domain, int64_t level,
                             const Matrix& rand);
  std::tuple<Matrix, Matrix>
  generate_V_transfer_matrix(Matrix& Vbig_child1, Matrix& Vbig_child2, int64_t node,
                             int64_t block_size, const Domain& domain, int64_t level,
                             const Matrix& rand);
  std::tuple<RowLevelMap, ColLevelMap>
  generate_transfer_matrices(const Domain& domain, int64_t level, const Matrix& rand,
                             RowLevelMap& Uchild, ColLevelMap& Vchild);
  Matrix get_Ubig(int64_t node, int64_t level);
  Matrix get_Vbig(int64_t node, int64_t level);
  void actually_print_structure(int64_t level);

 public:
  H2(const Domain& domain, const int64_t N, const int64_t nleaf,
     const int64_t lr_max_rank, const double accuracy, const double admis,
     const std::string& admis_kind, const int64_t matrix_type);
  double construction_absolute_error(const Domain& domain);
  void print_structure();
  double low_rank_block_ratio();
};

double laplace_kernel(const std::vector<double>& coords_row,
                      const std::vector<double>& coords_col) {
  const int64_t ndim = coords_row.size();
  double rij = 0;
  for (int64_t k = 0; k < ndim; ++k) {
    rij += pow(coords_row[k] - coords_col[k], 2);
  }
  const double out = 1 / (std::sqrt(rij) + PV);

  return out;
}

void generate_p2p_interactions(const Domain& domain,
                               int64_t irow, int64_t icol,
                               Matrix& out) {
  assert(out.rows == domain.boxes[irow].num_particles);
  assert(out.cols == domain.boxes[icol].num_particles);
  for (int64_t i = 0; i < domain.boxes[irow].num_particles; ++i) {
    for (int64_t j = 0; j < domain.boxes[icol].num_particles; ++j) {
      int64_t source = domain.boxes[irow].start_index;
      int64_t target = domain.boxes[icol].start_index;

      out(i, j) = kernel_function(domain.particles[source+i].coords,
                                  domain.particles[target+j].coords);
    }
  }
}

// Generates p2p interactions between the particles of two boxes specified by irow
// and icol. ndim specifies the dimensionality of the particles present in domain.
// Uses a laplace kernel for generating the interaction.
Matrix generate_p2p_interactions(const Domain& domain,
                                 int64_t irow, int64_t icol) {
  Matrix out(domain.boxes[irow].num_particles, domain.boxes[icol].num_particles);
  generate_p2p_interactions(domain, irow, icol, out);
  return out;
}

std::vector<int64_t> leaf_indices(int64_t node, int64_t level, int64_t height) {
  std::vector<int64_t> indices;
  if (level == height) {
    std::vector<int64_t> leaf_index{node};
    return leaf_index;
  }

  auto c1_indices = leaf_indices(node * 2, level + 1, height);
  auto c2_indices = leaf_indices(node * 2 + 1, level + 1, height);

  c1_indices.insert(c1_indices.end(), c2_indices.begin(), c2_indices.end());

  return c1_indices;
}

void generate_p2p_interactions(const Domain& domain,
                               int64_t irow, int64_t icol,
                               int64_t level, int64_t height,
                               Matrix& out) {
  if (level == height) {
    generate_p2p_interactions(domain, irow, icol, out);
  }

  std::vector<int64_t> leaf_rows = leaf_indices(irow, level, height);
  std::vector<int64_t> leaf_cols = leaf_indices(icol, level, height);

  int64_t nrows = 0, ncols = 0;
  for (int64_t i = 0; i < leaf_rows.size(); ++i) {
    nrows += domain.boxes[leaf_rows[i]].num_particles; }
  for (int64_t i = 0; i < leaf_cols.size(); ++i) {
    ncols += domain.boxes[leaf_cols[i]].num_particles; }

  assert(out.rows == nrows);
  assert(out.cols == ncols);

  std::vector<Particle> source_particles, target_particles;
  for (int64_t i = 0; i < leaf_rows.size(); ++i) {
    int64_t source_box = leaf_rows[i];
    int64_t source = domain.boxes[source_box].start_index;
    for (int64_t n = 0; n < domain.boxes[source_box].num_particles; ++n) {
      source_particles.push_back(domain.particles[source + n]);
    }
  }

  for (int64_t i = 0; i < leaf_cols.size(); ++i) {
    int64_t target_box = leaf_cols[i];
    int64_t target = domain.boxes[target_box].start_index;
    for (int64_t n = 0; n < domain.boxes[target_box].num_particles; ++n) {
      target_particles.push_back(domain.particles[target + n]);
    }
  }

  for (int64_t i = 0; i < nrows; ++i) {
    for (int64_t j = 0; j < ncols; ++j) {
      out(i, j) = kernel_function(source_particles[i].coords,
                                  target_particles[j].coords);
    }
  }
}

Matrix generate_p2p_interactions(const Domain& domain,
                                 int64_t irow, int64_t icol,
                                 int64_t level, int64_t height) {
  if (level == height) {
    return generate_p2p_interactions(domain, irow, icol);
  }

  std::vector<int64_t> leaf_rows = leaf_indices(irow, level, height);
  std::vector<int64_t> leaf_cols = leaf_indices(icol, level, height);

  int64_t nrows = 0, ncols = 0;
  for (int64_t i = 0; i < leaf_rows.size(); ++i) {
    nrows += domain.boxes[leaf_rows[i]].num_particles; }
  for (int64_t i = 0; i < leaf_cols.size(); ++i) {
    ncols += domain.boxes[leaf_cols[i]].num_particles; }

  Matrix out(nrows, ncols);
  generate_p2p_interactions(domain, irow, icol, level, height, out);

  return out;
}

Matrix generate_p2p_matrix(const Domain& domain) {
  int64_t rows =  domain.particles.size();
  int64_t cols =  domain.particles.size();
  Matrix out(rows, cols);

  std::vector<Particle> particles;

  for (int64_t irow = 0; irow < domain.boxes.size(); ++irow) {
    int64_t source = domain.boxes[irow].start_index;
    for (int64_t n = 0; n < domain.boxes[irow].num_particles; ++n) {
      particles.push_back(domain.particles[source + n]);
    }
  }


  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      out(i, j) = kernel_function(particles[i].coords,
                                  particles[j].coords);
    }
  }

  return out;
}

Particle::Particle(double x, double _value) : value(_value)  {
  coords.push_back(x);
}

Particle::Particle(double x, double y, double _value) : value(_value) {
  coords.push_back(x);
  coords.push_back(y);
}

Particle::Particle(double x, double y, double z, double _value) : value(_value)  {
  coords.push_back(x);
  coords.push_back(y);
  coords.push_back(z);
}

Particle::Particle(std::vector<double> _coords, double _value) :
    coords(_coords), value(_value) {}

Box::Box() {}

Box::Box(double _diameter, double center_x, int64_t _start_index, int64_t _stop_index,
         std::string _morton_index, int64_t _num_particles) :
    diameter(_diameter),
    ndim(1),
    num_particles(_num_particles),
    start_index(_start_index),
    stop_index(_stop_index),
    morton_index(_morton_index) {
  center.push_back(center_x);
}

Box::Box(double diameter, double center_x, double center_y, double start_index,
         double stop_index, std::string _morton_index, int64_t num_particles) :
    diameter(diameter),
    ndim(2),
    num_particles(num_particles),
    start_index(start_index),
    stop_index(stop_index),
    morton_index(_morton_index) {
  center.push_back(center_x);
  center.push_back(center_y);
}

Box::Box(double diameter, double center_x, double center_y, double center_z, double start_index,
         double stop_index, std::string _morton_index, int64_t num_particles) :
    diameter(diameter),
    ndim(3),
    num_particles(num_particles),
    start_index(start_index),
    stop_index(stop_index),
    morton_index(_morton_index) {
  center.push_back(center_x);
  center.push_back(center_y);
  center.push_back(center_z);
}


double Box::distance_from(const Box& b) const {
  double dist = 0;
  for (int64_t k = 0; k < ndim; ++k) {
    dist += pow(b.center[k] - center[k], 2);
  }
  return std::sqrt(dist);
}

void
Domain::print_file(std::string file_name) {
  std::vector<char> coords{'x', 'y', 'z'};

  std::ofstream file;
  file.open(file_name, std::ios::app | std::ios::out);
  for (int64_t k = 0; k < ndim; ++k) {
    file << coords[k] << ",";
  }
  file << std::endl;

  for (int64_t i = 0; i < N; ++i) {
    for (int64_t k = 0; k < ndim; ++k) {
      file << particles[i].coords[k] << ",";
    }
    file << std::endl;
  }

  file.close();
}

Matrix Domain::generate_rank_heat_map() {
  int64_t nblocks = boxes.size();
  Matrix out(nblocks, nblocks);

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < nblocks; ++j) {
      Matrix block = Hatrix::generate_p2p_interactions(*this, i, j);

      Matrix Utemp, Stemp, Vtemp;
      std::tie(Utemp, Stemp, Vtemp) = error_svd(block, 1e-9);
      int64_t rank = Stemp.rows;

      out(i, j) = rank;
    }
  }

  return out;
}

// https://www.csd.uwo.ca/~mmorenom/cs2101a_moreno/Barnes-Hut_Algorithm.pdf
void Domain::orthogonal_recursive_bisection_1dim(int64_t start,
                                                 int64_t end,
                                                 std::string morton_index,
                                                 int64_t nleaf) {
  // Sort the particles only by the X axis since that is the only axis that needs to be bisected.
  std::sort(particles.begin()+start,
            particles.begin()+end, [](const Particle& lhs, const Particle& rhs) {
              return lhs.coords[0] < rhs.coords[0];
            });

  int64_t num_points = end - start;
  // found a box with the correct number of points.
  if (num_points <= nleaf) {
    auto start_coord_x = particles[start].coords[0];
    auto end_coord_x = particles[end-1].coords[0];
    auto center_x = (start_coord_x + end_coord_x) / 2;
    auto diameter = end_coord_x - start_coord_x;
    boxes.push_back(Box(diameter, center_x, start, end-1, morton_index, num_points));
  }
  else {                    // recurse further and split again.
    int64_t middle = (start + end) / 2;
    // first half
    orthogonal_recursive_bisection_1dim(start, middle, morton_index + "0", nleaf);
    // second half
    orthogonal_recursive_bisection_1dim(middle, end, morton_index + "1", nleaf);
  }
}

void
Domain::orthogonal_recursive_bisection_2dim(int64_t start,
                                            int64_t end,
                                            std::string morton_index,
                                            int64_t nleaf,
                                            int64_t axis) {
  std::sort(particles.begin() + start,
            particles.begin() + end, [&](const Particle& lhs, const Particle& rhs) {
              return lhs.coords[axis] < rhs.coords[axis];
            });

  int64_t num_points = end - start;
  if (num_points <= nleaf) {
    if (axis == ndim-1) {
      int64_t start_index = start;
      int64_t end_index = end - 1;

      double diameter = 0;
      for (int64_t k = 0; k < ndim; ++k) {
        diameter += pow(particles[start_index].coords[k] - particles[end_index].coords[k], 2);
      }
      diameter = std::sqrt(diameter);

      double center_x = (particles[start_index].coords[0] + particles[end_index].coords[0]) / 2;
      double center_y = (particles[start_index].coords[1] + particles[end_index].coords[1]) / 2;

      boxes.push_back(Box(diameter,
                          center_x,
                          center_y,
                          start_index,
                          end_index,
                          morton_index,
                          num_points));
    }
    else {
      orthogonal_recursive_bisection_2dim(start, end, morton_index, nleaf, (axis + 1) % ndim);
    }
  }
  else {
    int64_t middle = (start + end) / 2;
    orthogonal_recursive_bisection_2dim(start,
                                        middle,
                                        morton_index + "0",
                                        nleaf,
                                        (axis + 1) % ndim);
    orthogonal_recursive_bisection_2dim(middle, end, morton_index + "1", nleaf, (axis + 1) % ndim);
  }
}

void
Domain::orthogonal_recursive_bisection_3dim(int64_t start, int64_t end, std::string morton_index,
                                            int64_t nleaf, int64_t axis) {
  std::sort(particles.begin() + start,
            particles.begin() + end,
            [&](const Particle& lhs, const Particle& rhs) {
              return lhs.coords[axis] < rhs.coords[axis];
            });
  int64_t num_points = end - start;
  if (num_points <= nleaf) {
    if (axis == ndim-1) {
      int64_t start_index = start;
      int64_t end_index = end - 1;

      double diameter = 0;
      for (int64_t k = 0; k < ndim; ++k) {
        diameter += pow(particles[start_index].coords[k] - particles[end_index].coords[k], 2);
      }
      diameter = std::sqrt(diameter);

      double center_x = (particles[start_index].coords[0] + particles[end_index].coords[0]) / 2;
      double center_y = (particles[start_index].coords[1] + particles[end_index].coords[1]) / 2;
      double center_z = (particles[start_index].coords[2] + particles[end_index].coords[2]) / 2;

      boxes.push_back(Box(diameter,
                          center_x,
                          center_y,
                          center_z,
                          start_index,
                          end_index,
                          morton_index,
                          num_points));
    }
    else {
      orthogonal_recursive_bisection_3dim(start, end, morton_index, nleaf, (axis+1) % ndim);
    }
  }
  else {
    int64_t middle = (start + end) / 2;
    orthogonal_recursive_bisection_3dim(start, middle, morton_index + "0", nleaf, (axis+1)%ndim);
    orthogonal_recursive_bisection_3dim(middle, end, morton_index + "1", nleaf, (axis+1)%ndim);
  }
}


Domain::Domain(int64_t N, int64_t ndim) : N(N), ndim(ndim) {
  if (ndim <= 0) {
    std::cout << "invalid ndim : " << ndim << std::endl;
    abort();
  }

}

void Domain::generate_starsh_grid_particles() {
  int64_t side = ceil(pow(N, 1.0 / ndim)); // size of each size of the grid.
  int64_t total = side;
  for (int64_t i = 1; i < ndim; ++i) { total *= side; }

  int64_t ncoords = ndim * side;
  std::vector<double> coord(ncoords);

  for (int64_t i = 0; i < side; ++i) {
    double val = double(i) / side;
    for (int64_t j = 0; j < ndim; ++j) {
      coord[j * side + i] = val;
    }
  }

  std::vector<int64_t> pivot(ndim, 0);

  int64_t k = 0;
  for (int64_t i = 0; i < N; ++i) {
    std::vector<double> points(ndim);
    for (k = 0; k < ndim; ++k) {
      points[k] = coord[pivot[k] + k * side];
    }
    particles.push_back(Hatrix::Particle(points, 0));

    k = ndim - 1;
    pivot[k]++;

    while(pivot[k] == side) {
      pivot[k] = 0;
      if (k > 0) {
        --k;
        pivot[k]++;
      }
    }
  }
}

void Domain::generate_particles(double min_val, double max_val) {
  double range = max_val - min_val;

  if (ndim == 1) {
    auto vec = equally_spaced_vector(N, min_val, max_val);
    for (int64_t i = 0; i < N; ++i) {
      particles.push_back(Hatrix::Particle(vec[i], min_val + (double(i) / double(range))));
    }
  }
  else if (ndim == 2) {
    // Generate a unit circle with N points on the circumference.
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 2.0 * M_PI);
    double radius = 1.0;
    for (int64_t i = 0; i < N; ++i) {
      double theta = (i * 2.0 * M_PI) / N ;
      double x = radius * cos(theta);
      double y = radius * sin(theta);

      particles.push_back(Hatrix::Particle(x, y, min_val + (double(i) / double(range))));
    }
  }
  else if (ndim == 3) {
    // Generate a unit sphere geometry with N points on the surface.
    // http://www.cpp.re/forum/windows/262648/
    // https://neil-strickland.staff.shef.ac.uk/courses/MAS243/lectures/handout10.pdf
    // std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(1); // Standard mersenne_twister_engine seeded with 1 every time.
    std::uniform_real_distribution<> dis(0.0, 2.0 * M_PI);
    double radius = 1.0;
    for (int64_t i = 0; i < N; ++i) {
      // double phi = dis(gen);
      // double theta = dis(gen);
      double phi = dis(gen);
      double theta = dis(gen);

      double x = radius * sin(phi) * cos(theta);
      double y = radius * sin(phi) * sin(theta);
      double z = radius * cos(phi);

      particles.push_back(Hatrix::Particle(x, y, z,
                                           min_val + (double(i) / double(range))));
    }
  }
}

void Domain::divide_domain_and_create_particle_boxes(int64_t nleaf) {
  if (ndim == 1) {
    orthogonal_recursive_bisection_1dim(0, N, std::string(""), nleaf);
  }
  else if (ndim == 2) {
    orthogonal_recursive_bisection_2dim(0, N, std::string(""), nleaf, 0);
  }
  else if (ndim == 3) {
    orthogonal_recursive_bisection_3dim(0, N, std::string(""), nleaf, 0);
  }
}

int64_t H2::find_all_dense_row() {
  int64_t nblocks = level_blocks[height];

  for (int64_t i = 0; i < nblocks; ++i) {
    bool all_dense_row = true;
    for (int64_t j = 0; j < nblocks; ++j) {
      if (!is_admissible.exists(i, j, height) ||
          (is_admissible.exists(i, j, height) && is_admissible(i, j, height))) {
        all_dense_row = false;
      }
    }

    if (all_dense_row) {
      return i;
    }
  }

  return -1;
}

void H2::coarsen_blocks(int64_t level) {
  int64_t child_level = level + 1;
  int64_t nblocks = pow(2, level);
  for (int64_t i = 0; i < nblocks; ++i) {
    std::vector<int64_t> row_children({i * 2, i * 2 + 1});
    for (int64_t j = 0; j < nblocks; ++j) {
      std::vector<int64_t> col_children({j * 2, j * 2 + 1});

      bool admis_block = true;
      for (int64_t c1 = 0; c1 < 2; ++c1) {
        for (int64_t c2 = 0; c2 < 2; ++c2) {
          if (is_admissible.exists(row_children[c1], col_children[c2], child_level) &&
              !is_admissible(row_children[c1], col_children[c2], child_level)) {
            admis_block = false;
          }
        }
      }

      if (admis_block) {
        for (int64_t c1 = 0; c1 < 2; ++c1) {
          for (int64_t c2 = 0; c2 < 2; ++c2) {
            is_admissible.erase(row_children[c1], col_children[c2], child_level);
          }
        }
      }

      is_admissible.insert(i, j, level, std::move(admis_block));
    }
  }
}

int64_t H2::geometry_admis_non_leaf(int64_t nblocks, int64_t level) {
  int64_t child_level = level - 1;
  level_blocks.push_back(nblocks);

  if (nblocks == 1) { return level; }

  for (int64_t i = 0; i < nblocks; ++i) {
    std::vector<int64_t> row_children({i * 2, i * 2 + 1});
    for (int64_t j = 0; j < nblocks; ++j) {
      std::vector<int64_t> col_children({j * 2, j * 2 + 1});

      bool admis_block = true;
      for (int64_t c1 = 0; c1 < 2; ++c1) {
        for (int64_t c2 = 0; c2 < 2; ++c2) {
          if (is_admissible.exists(row_children[c1], col_children[c2], child_level) &&
              !is_admissible(row_children[c1], col_children[c2], child_level)) {
            admis_block = false;
          }
        }
      }

      if (admis_block) {
        for (int64_t c1 = 0; c1 < 2; ++c1) {
          for (int64_t c2 = 0; c2 < 2; ++c2) {
            is_admissible.erase(row_children[c1], col_children[c2], child_level);
          }
        }
      }

      is_admissible.insert(i, j, level, std::move(admis_block));
    }
  }

  return geometry_admis_non_leaf(nblocks/2, level+1);
}

int64_t H2::calc_geometry_based_admissibility(const Domain& domain) {
  int64_t nblocks = domain.boxes.size();
  level_blocks.push_back(nblocks);
  int64_t level = 0;
  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < nblocks; ++j) {
      is_admissible.insert(i, j, level,
                           std::min(domain.boxes[i].diameter, domain.boxes[j].diameter) <=
                           admis * domain.boxes[i].distance_from(domain.boxes[j]));
    }
  }

  if (matrix_type == BLR2_MATRIX) {
    level_blocks.push_back(1);
    return 1;
  }
  else {
    return geometry_admis_non_leaf(nblocks / 2, level+1);
  }
}

void H2::calc_diagonal_based_admissibility(int64_t level) {
  int64_t nblocks = pow(2, level); // pow since we are using diagonal based admis.
  level_blocks.push_back(nblocks);
  if (level == 0) { return; }
  if (level == height) {
    for (int64_t i = 0; i < nblocks; ++i) {
      for (int64_t j = 0; j < nblocks; ++j) {
        is_admissible.insert(i, j, level, std::abs(i - j) > admis);
      }
    }
  }
  else {
    coarsen_blocks(level);
  }

  calc_diagonal_based_admissibility(level-1);
}

int64_t H2::get_block_size_row(const Domain& domain, int64_t parent, int64_t level) {
  if (level == height) {
    return domain.boxes[parent].num_particles;
  }
  int64_t child_level = level + 1;
  int64_t child1 = parent * 2;
  int64_t child2 = parent * 2 + 1;

  return get_block_size_row(domain, child1, child_level) +
      get_block_size_row(domain, child2, child_level);
}

int64_t H2::get_block_size_col(const Domain& domain, int64_t parent, int64_t level) {
  if (level == height) {
    return domain.boxes[parent].num_particles;
  }
  int64_t child_level = level + 1;
  int64_t child1 = parent * 2;
  int64_t child2 = parent * 2 + 1;

  return get_block_size_col(domain, child1, child_level) +
      get_block_size_col(domain, child2, child_level);
}

bool H2::row_has_admissible_blocks(int64_t row, int64_t level) {
  bool has_admis = false;
  for (int64_t i = 0; i < level_blocks[level]; ++i) {
    if (!is_admissible.exists(row, i, level) ||
        (is_admissible.exists(row, i, level) && is_admissible(row, i, level))) {
      has_admis = true;
      break;
    }
  }
  return has_admis;
}

bool H2::col_has_admissible_blocks(int64_t col, int64_t level) {
  bool has_admis = false;
  for (int64_t j = 0; j < level_blocks[level]; ++j) {
    if (!is_admissible.exists(j, col, level) ||
        (is_admissible.exists(j, col, level) && is_admissible(j, col, level))) {
      has_admis = true;
      break;
    }
  }

  return has_admis;
}

Matrix H2::generate_block_row(int64_t block, int64_t block_size,
                              const Domain& domain, int64_t level,
                              const Matrix& rand) {
  int64_t nblocks = level_blocks[level];
  auto rand_splits = rand.split(nblocks, 1);

  Matrix block_row(block_size, rand.cols);
  for (int64_t j = 0; j < nblocks; ++j) {
    if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) { continue; }
    // Accumulate sample of each block
    matmul(generate_p2p_interactions(domain, block, j, level, height), rand_splits[j],
           block_row, false, false, 1.0, 1.0);
  }
  return block_row;
}

std::tuple<Matrix, Matrix>
H2::generate_row_cluster_bases(int64_t block, int64_t block_size,
                               const Domain& domain, int64_t level,
                               const Matrix& rand) {
  Matrix block_row = generate_block_row(block, block_size, domain, level, rand);
  Matrix Ui, Vi;
  std::tie(Ui, Vi) = truncated_pivoted_qr(block_row, accuracy);
  int64_t rank = Ui.cols;
  min_rank = std::min(min_rank, rank);
  max_rank = std::max(max_rank, rank);
  return {std::move(Ui), std::move(Vi)};
}

Matrix H2::generate_block_column(int64_t block, int64_t block_size,
                                 const Domain& domain, int64_t level,
                                 const Matrix& rand) {
  int64_t nblocks = level_blocks[level];
  auto rand_splits = rand.split(nblocks, 1);

  Matrix block_column(rand.cols, block_size);
  for (int64_t i = 0; i < nblocks; ++i) {
    if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) { continue; }
    // Accumulate sample of each block
    matmul(rand_splits[i], generate_p2p_interactions(domain, i, block, level, height),
           block_column, true, false, 1.0, 1.0);
  }
  return block_column;
}

std::tuple<Matrix, Matrix>
H2::generate_column_cluster_bases(int64_t block, int64_t block_size,
                                  const Domain& domain, int64_t level,
                                  const Matrix& rand) {
  Matrix block_column_T = transpose(generate_block_column(block, block_size, domain, level, rand));
  Matrix Ui, Vi;
  std::tie(Ui, Vi) = Hatrix::truncated_pivoted_qr(block_column_T, accuracy);
  int64_t rank = Ui.cols;
  min_rank = std::min(min_rank, rank);
  max_rank = std::max(max_rank, rank);
  return {std::move(Ui), std::move(Vi)};
}

void H2::generate_leaf_nodes(const Domain& domain, const Matrix& rand) {
  int64_t nblocks = level_blocks[height];
  // Generate inadmissible leaf blocks
  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < nblocks; ++j) {
      if (is_admissible.exists(i, j, height) && !is_admissible(i, j, height)) {
        D.insert(i, j, height,
                 generate_p2p_interactions(domain, i, j));
      }
    }
  }
  // Generate leaf level U
  for (int64_t i = 0; i < nblocks; ++i) {
    Matrix Utemp, _;
    std::tie(Utemp, _) =
        generate_row_cluster_bases(i, domain.boxes[i].num_particles, domain, height, rand);
    U.insert(i, height, std::move(Utemp));
  }
  // Generate leaf level V
  for (int64_t j = 0; j < nblocks; ++j) {
    Matrix Vtemp, _;
    std::tie(Vtemp, _) =
        generate_column_cluster_bases(j, domain.boxes[j].num_particles, domain, height, rand);
    V.insert(j, height, std::move(Vtemp));
  }
  // Generate S coupling matrices
  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < nblocks; ++j) {
      if (is_admissible.exists(i, j, height) && is_admissible(i, j, height)) {
        Matrix dense = generate_p2p_interactions(domain, i, j);

        S.insert(i, j, height,
                 matmul(matmul(U(i, height), dense, true, false),
                        V(j, height)));
      }
    }
  }
}

std::tuple<Matrix, Matrix>
H2::generate_U_transfer_matrix(Matrix& Ubig_child1, Matrix& Ubig_child2, int64_t node,
                               int64_t block_size, const Domain& domain, int64_t level,
                               const Matrix& rand) {
  Matrix block_row = generate_block_row(node, block_size, domain, level, rand);
  auto block_row_splits = block_row.split(2, 1);

  Matrix temp(Ubig_child1.cols + Ubig_child2.cols, block_row.cols);
  auto temp_splits = temp.split(vec{Ubig_child1.cols}, vec{});

  matmul(Ubig_child1, block_row_splits[0], temp_splits[0], true, false, 1, 0);
  matmul(Ubig_child2, block_row_splits[1], temp_splits[1], true, false, 1, 0);
  Matrix Utransfer, Vi;
  std::tie(Utransfer, Vi) = truncated_pivoted_qr(temp, accuracy);

  int64_t rank = Utransfer.cols;
  min_rank = std::min(min_rank, rank);
  max_rank = std::max(max_rank, rank);
  return {std::move(Utransfer), std::move(Vi)};
}

std::tuple<Matrix, Matrix>
H2::generate_V_transfer_matrix(Matrix& Vbig_child1, Matrix& Vbig_child2, int64_t node,
                               int64_t block_size, const Domain& domain, int64_t level,
                               const Matrix& rand) {
  Matrix block_column_T = transpose(generate_block_column(node, block_size, domain, level, rand));
  auto block_column_T_splits = block_column_T.split(2, 1);

  Matrix temp(Vbig_child1.cols + Vbig_child2.cols, block_column_T.cols);
  auto temp_splits = temp.split(vec{Vbig_child1.cols}, vec{});

  matmul(Vbig_child1, block_column_T_splits[0], temp_splits[0], true, false, 1, 0);
  matmul(Vbig_child2, block_column_T_splits[1], temp_splits[1], true, false, 1, 0);
  Matrix Vtransfer, Vi;
  std::tie(Vtransfer, Vi) = truncated_pivoted_qr(temp, accuracy);

  int64_t rank = Vtransfer.cols;
  min_rank = std::min(min_rank, rank);
  max_rank = std::max(max_rank, rank);
  return {std::move(Vtransfer), std::move(Vi)};
}

std::tuple<RowLevelMap, ColLevelMap>
H2::generate_transfer_matrices(const Domain& domain, int64_t level, const Matrix& rand,
                               RowLevelMap& Uchild, ColLevelMap& Vchild) {
  // Generate the actual bases for the upper level and pass it to this
  // function again for generating transfer matrices at successive levels.
  RowLevelMap Ubig_parent;
  ColLevelMap Vbig_parent;

  int64_t nblocks = level_blocks[level];
  for (int64_t node = 0; node < nblocks; ++node) {
    int64_t child1 = node * 2;
    int64_t child2 = node * 2 + 1;
    int64_t child_level = level + 1;
    int64_t block_size = get_block_size_row(domain, node, level);

    if (row_has_admissible_blocks(node, level) && height != 1) {
      // Generate row cluster transfer matrix.
      Matrix& Ubig_child1 = Uchild(child1, child_level);
      Matrix& Ubig_child2 = Uchild(child2, child_level);
      Matrix Utransfer, _;
      std::tie(Utransfer, _) = generate_U_transfer_matrix(Ubig_child1, Ubig_child2,
                                                          node, block_size, domain, level, rand);
      U.insert(node, level, std::move(Utransfer));

      // Generate the full bases to pass onto the parent.
      auto Utransfer_splits = U(node, level).split(vec{Ubig_child1.cols}, vec{});
      Matrix Ubig(block_size, U(node, level).cols);
      auto Ubig_splits = Ubig.split(vec{Ubig_child1.rows}, vec{});

      matmul(Ubig_child1, Utransfer_splits[0], Ubig_splits[0]);
      matmul(Ubig_child2, Utransfer_splits[1], Ubig_splits[1]);
      Ubig_parent.insert(node, level, std::move(Ubig));
    }

    if (col_has_admissible_blocks(node, level) && height != 1) {
      // Generate column cluster transfer matrix.
      Matrix& Vbig_child1 = Vchild(child1, child_level);
      Matrix& Vbig_child2 = Vchild(child2, child_level);
      Matrix Vtransfer, _;
      std::tie(Vtransfer, _) = generate_V_transfer_matrix(Vbig_child1, Vbig_child2,
                                                          node, block_size, domain, level, rand);
      V.insert(node, level, std::move(Vtransfer));

      // Generate the full bases for passing onto the upper level.
      auto Vtransfer_splits = V(node, level).split(vec{Vbig_child1.cols}, vec{});
      Matrix Vbig(block_size, V(node, level).cols);
      auto Vbig_splits = Vbig.split(vec{Vbig_child1.rows}, vec{});

      matmul(Vbig_child1, Vtransfer_splits[0], Vbig_splits[0]);
      matmul(Vbig_child2, Vtransfer_splits[1], Vbig_splits[1]);
      Vbig_parent.insert(node, level, std::move(Vbig));
    }
  }

  for (int64_t row = 0; row < nblocks; ++row) {
    for (int64_t col = 0; col < nblocks; ++col) {
      if (is_admissible.exists(row, col, level) && is_admissible(row, col, level)) {
        Matrix D = generate_p2p_interactions(domain, row, col, level, height);

        S.insert(row, col, level, matmul(matmul(Ubig_parent(row, level), D, true, false),
                                         Vbig_parent(col, level)));
      }
    }
  }
  return {Ubig_parent, Vbig_parent};
}

Matrix H2::get_Ubig(int64_t node, int64_t level) {
  if (level == height) {
    return U(node, level);
  }

  int64_t child1 = node * 2;
  int64_t child2 = node * 2 + 1;
  Matrix Ubig_child1 = get_Ubig(child1, level+1);
  Matrix Ubig_child2 = get_Ubig(child2, level+1);

  int64_t block_size = Ubig_child1.rows + Ubig_child2.rows;
  Matrix Ubig(block_size, U(node, level).cols);
  auto Ubig_splits = Ubig.split(vec{Ubig_child1.rows}, vec{});
  auto U_splits = U(node, level).split(vec{Ubig_child1.cols}, vec{});

  matmul(Ubig_child1, U_splits[0], Ubig_splits[0]);
  matmul(Ubig_child2, U_splits[1], Ubig_splits[1]);
  return Ubig;
}

Matrix H2::get_Vbig(int64_t node, int64_t level) {
  if (level == height) {
    return V(node, height);
  }

  int64_t child1 = node * 2;
  int64_t child2 = node * 2 + 1;
  Matrix Vbig_child1 = get_Vbig(child1, level+1);
  Matrix Vbig_child2 = get_Vbig(child2, level+1);

  int64_t block_size = Vbig_child1.rows + Vbig_child2.rows;
  Matrix Vbig(block_size, V(node, level).cols);
  auto Vbig_splits = Vbig.split(vec{Vbig_child1.rows}, vec{});
  auto V_splits = V(node, level).split(vec{Vbig_child1.cols}, vec{});

  matmul(Vbig_child1, V_splits[0], Vbig_splits[0]);
  matmul(Vbig_child2, V_splits[1], Vbig_splits[1]);
  return Vbig;
}

H2::H2(const Domain& domain, const int64_t N, const int64_t nleaf,
       const int64_t lr_max_rank, const double accuracy, const double admis,
       const std::string& admis_kind, const int64_t matrix_type)
    : N(N), nleaf(nleaf), lr_max_rank(lr_max_rank), accuracy(accuracy),
      admis(admis), admis_kind(admis_kind), matrix_type(matrix_type),
      min_rank(lr_max_rank), max_rank(-lr_max_rank) {
  if (admis_kind == "geometry_admis") {
    // TODO: use dual tree traversal for this.
    height = calc_geometry_based_admissibility(domain);
    // reverse the levels stored in the admis blocks.
    RowColLevelMap<bool> temp_is_admissible;

    for (int64_t level = 0; level < height; ++level) {
      int64_t nblocks = level_blocks[level];

      for (int64_t i = 0; i < nblocks; ++i) {
        for (int64_t j = 0; j < nblocks; ++j) {
          if (is_admissible.exists(i, j, level)) {
            bool value = is_admissible(i, j, level);
            temp_is_admissible.insert(i, j, height - level,
                                      std::move(value));
          }
        }
      }
    }

    is_admissible = temp_is_admissible;
    std::reverse(std::begin(level_blocks), std::end(level_blocks));
  }
  else if (admis_kind == "diagonal_admis") {
    if (matrix_type == BLR2_MATRIX) {
      height = 1;
      int64_t nblocks = domain.boxes.size();
      for (int64_t i = 0; i < nblocks; ++i) {
        for (int64_t j = 0; j < nblocks; ++j) {
          is_admissible.insert(i, j, height, std::abs(i - j) > admis);
        }
      }
      level_blocks.push_back(1);
      level_blocks.push_back(nblocks);
    }
    else if (matrix_type == H2_MATRIX) {
      height = int64_t(log2(N / nleaf));
      calc_diagonal_based_admissibility(height);
      std::reverse(std::begin(level_blocks), std::end(level_blocks));
    }
  }
  else {
    std::cout << "wrong admis condition: " << admis_kind << std::endl;
    abort();
  }

  is_admissible.insert(0, 0, 0, false);
  PV =  (1 / double(N)) * 1e-3;

  int64_t all_dense_row = find_all_dense_row();
  if (all_dense_row != -1) {
    std::cout << "found all dense row at " << all_dense_row << ". Aborting.\n";
    abort();
  }

  Matrix rand = generate_random_matrix(N, lr_max_rank);
  generate_leaf_nodes(domain, rand);
  RowLevelMap Uchild = U;
  ColLevelMap Vchild = V;

  for (int64_t level = height-1; level > 0; --level) {
    std::tie(Uchild, Vchild) = generate_transfer_matrices(domain, level, rand, Uchild, Vchild);
  }
}

double H2::construction_absolute_error(const Domain& domain) {
  double error = 0;
  double dense_norm = 0;
  int64_t nblocks = level_blocks[height];

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < nblocks; ++j) {
      if (is_admissible.exists(i, j, height) && !is_admissible(i, j, height)) {
        Matrix actual = Hatrix::generate_p2p_interactions(domain, i, j);
        Matrix expected = D(i, j, height);
        error += pow(norm(actual - expected), 2);
        dense_norm += pow(norm(actual), 2);
      }
    }
  }

  for (int64_t level = height; level > 0; --level) {
    int64_t nblocks = level_blocks[level];

    for (int64_t row = 0; row < nblocks; ++row) {
      for (int64_t col = 0; col < nblocks; ++col) {
        if (is_admissible.exists(row, col, level) && is_admissible(row, col, level)) {
          Matrix Ubig = get_Ubig(row, level);
          Matrix Vbig = get_Vbig(col, level);

          Matrix expected_matrix = matmul(matmul(Ubig, S(row, col, level)), Vbig, false, true);
          Matrix actual_matrix =
              Hatrix::generate_p2p_interactions(domain, row, col, level, height);

          dense_norm += pow(norm(actual_matrix), 2);
          error += pow(norm(expected_matrix - actual_matrix), 2);
        }
      }
    }
  }

  return std::sqrt(error);
}

void H2::actually_print_structure(int64_t level) {
  if (level == 0) { return; }
  int64_t nblocks = level_blocks[level];
  std::cout << "LEVEL: " << level << " NBLOCKS: " << nblocks << std::endl;
  for (int64_t i = 0; i < nblocks; ++i) {
    if (level == height && D.exists(i, i, height)) {
      std::cout << D(i, i, height).rows << " ";
    }
    std::cout << "| ";
    for (int64_t j = 0; j < nblocks; ++j) {
      if (is_admissible.exists(i, j, level)) {
        std::cout << is_admissible(i, j, level) << " | " ;
      }
      else {
        std::cout << "  | ";
      }
    }
    std::cout << std::endl;
  }

  std::cout << std::endl;

  actually_print_structure(level-1);
}

void H2::print_structure() {
  actually_print_structure(height);
}

double H2::low_rank_block_ratio() {
  double total = 0, low_rank = 0;

  int64_t nblocks = level_blocks[height];
  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < nblocks; ++j) {
      if ((is_admissible.exists(i, j, height) && is_admissible(i, j, height)) ||
          !is_admissible.exists(i, j, height)) {
        low_rank += 1;
      }

      total += 1;
    }
  }

  return low_rank / total;
}

} // namespace Hatrix

int main(int argc, char ** argv) {
  const int64_t N = argc > 1 ? atoi(argv[1]) : 256;
  const int64_t nleaf = argc > 2 ? atoi(argv[2]) : 32;
  const int64_t lr_max_rank = argc > 3 ? atoi(argv[3]) : nleaf/2;
  const double accuracy = argc > 4 ? atof(argv[4]) : 1e-5;
  const double admis = argc > 5 ? atof(argv[5]) : 1.0;
  // diagonal_admis or geometry_admis
  const std::string admis_kind = argc > 6 ? std::string(argv[6]) : "diagonal_admis";
  // 0: BLR2
  // 1: H2
  const int64_t matrix_type = argc > 7 ? atoi(argv[7]) : 1;

  Hatrix::Context::init();

  const auto start_particles = std::chrono::system_clock::now();
  constexpr int64_t ndim = 2;
  Hatrix::Domain domain(N, ndim);
  // Laplace kernel
  domain.generate_particles(0, N);
  Hatrix::kernel_function = Hatrix::laplace_kernel;
  domain.divide_domain_and_create_particle_boxes(nleaf);
  const auto stop_particles = std::chrono::system_clock::now();
  const double particle_construct_time = std::chrono::duration_cast<std::chrono::milliseconds>
                                         (stop_particles - start_particles).count();

  const auto start_construct = std::chrono::system_clock::now();
  Hatrix::H2 A(domain, N, nleaf, lr_max_rank, accuracy, admis, admis_kind, matrix_type);
  const auto stop_construct = std::chrono::system_clock::now();
  const double construct_time = std::chrono::duration_cast<std::chrono::milliseconds>
                                (stop_construct - start_construct).count();
  double construct_error, lr_ratio, solve_error;
  construct_error = A.construction_absolute_error(domain);
  lr_ratio = A.low_rank_block_ratio();

  Hatrix::Context::finalize();

  std::cout << "N=" << N
            << " nleaf=" << nleaf
            << " lr_max_rank=" << lr_max_rank
            << " accuracy=" << accuracy
            << " admis=" << admis << std::setw(3)
            << " ndim=" << ndim
            << " height=" << A.height
            << " admis_kind=" << admis_kind
            << " matrix_type=" << (matrix_type == BLR2_MATRIX ? "BLR2" : "H2")
            << " LR%=" << std::setprecision(5) << lr_ratio * 100 << "%"
            << " min_rank=" << A.min_rank
            << " max_rank=" << A.max_rank
            << " construct_error=" << std::setprecision(5) << construct_error
            << " construct_time=" << construct_time
            << " particle_time=" << particle_construct_time
            << std::endl;

  return 0;
}
