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
#include <stdexcept>

#include "Hatrix/Hatrix.h"

#define USE_QR_COMPRESSION

using vec = std::vector<int64_t>;

constexpr double EPS = 1e-14;
enum MATRIX_TYPES {BLR2_MATRIX=0, H2_MATRIX=1};
double PV = 1e-3;

Hatrix::Matrix prepend_complement(const Hatrix::Matrix &Q) {
  Hatrix::Matrix Q_F(Q.rows, Q.rows);
  Hatrix::Matrix Q_full, R;
  std::tie(Q_full, R) = Hatrix::qr(Q, Hatrix::Lapack::QR_mode::Full, Hatrix::Lapack::QR_ret::OnlyQ);

  for (int64_t i = 0; i < Q_F.rows; ++i) {
    for (int64_t j = 0; j < Q_F.cols - Q.cols; ++j) {
      Q_F(i, j) = Q_full(i, j + Q.cols);
    }
  }

  for (int64_t i = 0; i < Q_F.rows; ++i) {
    for (int64_t j = 0; j < Q.cols; ++j) {
      Q_F(i, j + (Q_F.cols - Q.cols)) = Q(i, j);
    }
  }
  return Q_F;
}

Hatrix::Matrix diag(const Hatrix::Matrix& A) {
  Hatrix::Matrix diag(A.min_dim(), 1);
  for(int64_t i = 0; i < A.min_dim(); i++) {
    diag(i, 0) = A(i, i);
  }
  return diag;
}

void shift_diag(Hatrix::Matrix& A, const double shift) {
  for(int64_t i = 0; i < A.min_dim(); i++) {
    A(i, i) += shift;
  }
}

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

class SymmetricH2 {
 public:
  int64_t N, nleaf, n_blocks;
  double accuracy;
  int64_t rank;
  double admis;
  std::string admis_kind;
  int64_t matrix_type;
  int64_t height;
  RowLevelMap U;
  RowColLevelMap<Matrix> D, S;
  RowColLevelMap<bool> is_admissible;
  RowLevelMap Srow;
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
  std::tuple<Matrix, Matrix, Matrix> svd_like_compression(Matrix& A);
  Matrix generate_block_row(int64_t block, int64_t block_size,
                            const Domain& domain, int64_t level,
                            const Matrix& rand);
  std::tuple<Matrix, Matrix>
  generate_row_cluster_bases(int64_t block, int64_t block_size,
                             const Domain& domain, int64_t level,
                             const Matrix& rand);
  void generate_leaf_nodes(const Domain& domain, const Matrix& rand);

  std::tuple<Matrix, Matrix>
  generate_U_transfer_matrix(Matrix& Ubig_child1, Matrix& Ubig_child2, int64_t node,
                             int64_t block_size, const Domain& domain, int64_t level,
                             const Matrix& rand);
  RowLevelMap generate_transfer_matrices(const Domain& domain,
                                         int64_t level, const Matrix& rand,
                                         RowLevelMap& Uchild);
  Matrix get_Ubig(int64_t node, int64_t level);
  void actually_print_structure(int64_t level);

  Matrix compute_Srow(int64_t row, int64_t row_level); // TODO implement
  void update_row_cluster_bases(int64_t row, int64_t level,
                                RowColLevelMap<Matrix>& F, RowMap<Matrix>& r);
  void factorize_level(const Domain& domain,
                       int64_t level, int64_t nblocks,
                       RowColLevelMap<Matrix>& F, RowMap<Matrix>& r);

 public:
  SymmetricH2(const Domain& domain, const int64_t N, const int64_t nleaf,
              const double accuracy, const int64_t rank, const double admis,
              const std::string& admis_kind, const int64_t matrix_type,
              const Matrix& rand);
  double construction_absolute_error(const Domain& domain);
  void print_structure();
  double low_rank_block_ratio();
  void factorize(const Domain& domain);
  void print_ranks();
  std::tuple<int64_t, int64_t> inertia(const Domain& domain,
                                       const double lambda, bool &singular);
  std::tuple<double, int64_t, double>
  get_mth_eigenvalue(const Domain& domain, const int64_t m,
                     const double ev_tol,
                     double left, double right);
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
      std::tie(Utemp, Stemp, Vtemp) = error_svd(block, 1e-9, false);
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

int64_t SymmetricH2::find_all_dense_row() {
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

void SymmetricH2::coarsen_blocks(int64_t level) {
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

int64_t SymmetricH2::geometry_admis_non_leaf(int64_t nblocks, int64_t level) {
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

int64_t SymmetricH2::calc_geometry_based_admissibility(const Domain& domain) {
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

void SymmetricH2::calc_diagonal_based_admissibility(int64_t level) {
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

int64_t SymmetricH2::get_block_size_row(const Domain& domain, int64_t parent, int64_t level) {
  if (level == height) {
    return domain.boxes[parent].num_particles;
  }
  int64_t child_level = level + 1;
  int64_t child1 = parent * 2;
  int64_t child2 = parent * 2 + 1;

  return get_block_size_row(domain, child1, child_level) +
      get_block_size_row(domain, child2, child_level);
}

int64_t SymmetricH2::get_block_size_col(const Domain& domain, int64_t parent, int64_t level) {
  if (level == height) {
    return domain.boxes[parent].num_particles;
  }
  int64_t child_level = level + 1;
  int64_t child1 = parent * 2;
  int64_t child2 = parent * 2 + 1;

  return get_block_size_col(domain, child1, child_level) +
      get_block_size_col(domain, child2, child_level);
}

bool SymmetricH2::row_has_admissible_blocks(int64_t row, int64_t level) {
  bool has_admis = false;
  for (int64_t j = 0; j < level_blocks[level]; ++j) {
    if ((!is_admissible.exists(row, j, level)) || // part of upper level admissible block
        (is_admissible.exists(row, j, level) && is_admissible(row, j, level))) {
      has_admis = true;
      break;
    }
  }
  return has_admis;
}

bool SymmetricH2::col_has_admissible_blocks(int64_t col, int64_t level) {
  bool has_admis = false;
  for (int64_t i = 0; i < level_blocks[level]; ++i) {
    if ((!is_admissible.exists(i, col, level)) || // part of upper level admissible block
        (is_admissible.exists(i, col, level) && is_admissible(i, col, level))) {
      has_admis = true;
      break;
    }
  }
  return has_admis;
}

std::tuple<Matrix, Matrix, Matrix> SymmetricH2::svd_like_compression(Matrix& A) {
  Matrix Ui, Si, Vi;
  if (accuracy == 0.) {  // Fixed rank
    double error;
    std::tie(Ui, Si, Vi, error) = truncated_svd(A, std::min(rank, A.min_dim()));
  }
  else {  // Fixed accuracy
#ifdef USE_QR_COMPRESSION
    Matrix R;
    std::tie(Ui, R) = truncated_pivoted_qr(A, accuracy, false);
    Si = Matrix(R.rows, R.rows);
    Vi = Matrix(R.rows, R.cols);
    rq(R, Si, Vi);
#else
    std::tie(Ui, Si, Vi) = error_svd(A, accuracy, false);
#endif
  }
  return {std::move(Ui), std::move(Si), std::move(Vi)};
}

Matrix SymmetricH2::generate_block_row(int64_t block, int64_t block_size,
                                       const Domain& domain, int64_t level,
                                       const Matrix& rand) {
  int64_t nblocks = level_blocks[level];
  std::vector<Matrix> rand_splits;
  bool sample = (rank > 0);
  if (sample) {
    rand_splits = rand.split(nblocks, 1);
  }

  Matrix block_row(block_size, sample ? rand.cols : 0);
  for (int64_t j = 0; j < nblocks; ++j) {
    if ((!is_admissible.exists(block, j, level)) || // part of upper level admissible block
        (is_admissible.exists(block, j, level) && is_admissible(block, j, level))) {
      if (sample) {
        matmul(generate_p2p_interactions(domain, block, j, level, height), rand_splits[j],
               block_row, false, false, 1.0, 1.0);
      }
      else {
        block_row =
            concat(block_row, generate_p2p_interactions(domain, block, j, level, height), 1);
      }
    }
  }
  return block_row;
}

std::tuple<Matrix, Matrix>
SymmetricH2::generate_row_cluster_bases(int64_t block, int64_t block_size,
                                        const Domain& domain, int64_t level,
                                        const Matrix& rand) {
  Matrix block_row = generate_block_row(block, block_size, domain, level, rand);
  Matrix Ui, Si, Vi;
  std::tie(Ui, Si, Vi) = svd_like_compression(block_row);
  min_rank = std::min(min_rank, Ui.cols);
  max_rank = std::max(max_rank, Ui.cols);

  return {std::move(Ui), std::move(Si)};
}

void SymmetricH2::generate_leaf_nodes(const Domain& domain, const Matrix& rand) {
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
    Matrix Utemp, Stemp;
    std::tie(Utemp, Stemp) =
        generate_row_cluster_bases(i, domain.boxes[i].num_particles, domain, height, rand);
    U.insert(i, height, std::move(Utemp));
    Srow.insert(i, height, std::move(Stemp));
  }
  // Generate S coupling matrices
  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < nblocks; ++j) {
      if (is_admissible.exists(i, j, height) && is_admissible(i, j, height)) {
        Matrix dense = generate_p2p_interactions(domain, i, j);

        S.insert(i, j, height,
                 matmul(matmul(U(i, height), dense, true, false),
                        U(j, height)));
      }
    }
  }
}

std::tuple<Matrix, Matrix>
SymmetricH2::generate_U_transfer_matrix(Matrix& Ubig_child1, Matrix& Ubig_child2, int64_t node,
                                        int64_t block_size, const Domain& domain, int64_t level,
                                        const Matrix& rand) {
  Matrix block_row = generate_block_row(node, block_size, domain, level, rand);
  auto block_row_splits = block_row.split(2, 1);

  Matrix temp(Ubig_child1.cols + Ubig_child2.cols, block_row.cols);
  auto temp_splits = temp.split(vec{Ubig_child1.cols}, vec{});

  matmul(Ubig_child1, block_row_splits[0], temp_splits[0], true, false, 1, 0);
  matmul(Ubig_child2, block_row_splits[1], temp_splits[1], true, false, 1, 0);

  Matrix Ui, Si, Vi;
  std::tie(Ui, Si, Vi) = svd_like_compression(temp);
  min_rank = std::min(min_rank, Ui.cols);
  max_rank = std::max(max_rank, Ui.cols);

  return {std::move(Ui), std::move(Si)};
}

RowLevelMap SymmetricH2::generate_transfer_matrices(const Domain& domain,
                                                    int64_t level, const Matrix& rand,
                                                    RowLevelMap& Uchild) {
  // Generate the actual bases for the upper level and pass it to this
  // function again for generating transfer matrices at successive levels.
  RowLevelMap Ubig_parent;

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
      Matrix Utransfer, Stemp;
      std::tie(Utransfer, Stemp) =
          generate_U_transfer_matrix(Ubig_child1, Ubig_child2,
                                     node, block_size, domain, level, rand);
      U.insert(node, level, std::move(Utransfer));
      Srow.insert(node, level, std::move(Stemp));

      // Generate the full bases to pass onto the parent.
      auto Utransfer_splits = U(node, level).split(vec{Ubig_child1.cols}, vec{});
      Matrix Ubig(block_size, U(node, level).cols);
      auto Ubig_splits = Ubig.split(vec{Ubig_child1.rows}, vec{});

      matmul(Ubig_child1, Utransfer_splits[0], Ubig_splits[0]);
      matmul(Ubig_child2, Utransfer_splits[1], Ubig_splits[1]);
      Ubig_parent.insert(node, level, std::move(Ubig));
    }
  }

  for (int64_t row = 0; row < nblocks; ++row) {
    for (int64_t col = 0; col < nblocks; ++col) {
      if (is_admissible.exists(row, col, level) && is_admissible(row, col, level)) {
        Matrix D = generate_p2p_interactions(domain, row, col, level, height);

        S.insert(row, col, level, matmul(matmul(Ubig_parent(row, level), D, true, false),
                                         Ubig_parent(col, level)));
      }
    }
  }
  return Ubig_parent;
}

Matrix SymmetricH2::get_Ubig(int64_t node, int64_t level) {
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

SymmetricH2::SymmetricH2(const Domain& domain, const int64_t N, const int64_t nleaf,
                         const double accuracy, const int64_t rank, const double admis,
                         const std::string& admis_kind, const int64_t matrix_type,
                         const Matrix& rand)
    : N(N), nleaf(nleaf), accuracy(accuracy), rank(rank),
      admis(admis), admis_kind(admis_kind), matrix_type(matrix_type),
      min_rank(N), max_rank(-N) {
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

  int64_t all_dense_row = find_all_dense_row();
  if (all_dense_row != -1) {
    std::cout << "found all dense row at " << all_dense_row << ". Aborting.\n";
    abort();
  }

  generate_leaf_nodes(domain, rand);
  RowLevelMap Uchild = U;

  for (int64_t level = height-1; level > 0; --level) {
    Uchild = generate_transfer_matrices(domain, level, rand, Uchild);
  }
}

double SymmetricH2::construction_absolute_error(const Domain& domain) {
  double error = 0;
  int64_t nblocks = level_blocks[height];

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < nblocks; ++j) {
      if (is_admissible.exists(i, j, height) && !is_admissible(i, j, height)) {
        Matrix actual = Hatrix::generate_p2p_interactions(domain, i, j);
        Matrix expected = D(i, j, height);
        error += pow(norm(actual - expected), 2);
      }
    }
  }

  for (int64_t level = height; level > 0; --level) {
    int64_t nblocks = level_blocks[level];

    for (int64_t row = 0; row < nblocks; ++row) {
      for (int64_t col = 0; col < nblocks; ++col) {
        if (is_admissible.exists(row, col, level) && is_admissible(row, col, level)) {
          Matrix Ubig = get_Ubig(row, level);
          Matrix Vbig = get_Ubig(col, level);

          Matrix expected_matrix = matmul(matmul(Ubig, S(row, col, level)), Vbig, false, true);
          Matrix actual_matrix =
              Hatrix::generate_p2p_interactions(domain, row, col, level, height);

          error += pow(norm(expected_matrix - actual_matrix), 2);
        }
      }
    }
  }

  return std::sqrt(error);
}

void SymmetricH2::actually_print_structure(int64_t level) {
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

void SymmetricH2::print_structure() {
  actually_print_structure(height);
}

double SymmetricH2::low_rank_block_ratio() {
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

void SymmetricH2::update_row_cluster_bases(int64_t row, int64_t level,
                                           RowColLevelMap<Matrix>& F, RowMap<Matrix>& r) {
  int64_t nblocks = level_blocks[level];
  int64_t block_size = D(row, row, level).rows;
  Matrix block_row(block_size, 0);

  // Concatenation approach
  block_row = concat(block_row, matmul(U(row, level), Srow(row, level)), 1);
  for (int64_t j = 0; j < nblocks; ++j) {
    if (F.exists(row, j, level)) {
      block_row = concat(block_row, F(row, j, level), 1);
    }
  }

  Matrix Ui, Si, Vi;
  std::tie(Ui, Si, Vi) = svd_like_compression(block_row);
  min_rank = std::min(min_rank, Ui.cols);
  max_rank = std::max(max_rank, Ui.cols);

  Matrix r_row = matmul(Ui, U(row, level), true, false);
  if (r.exists(row)) {
    r.erase(row);
  }
  r.insert(row, std::move(r_row));

  U.erase(row, level);
  U.insert(row, level, std::move(Ui));

  Srow.erase(row, level);
  Srow.insert(row, level, std::move(Si));
}

void SymmetricH2::factorize_level(const Domain& domain,
                                  int64_t level, int64_t nblocks,
                                  RowColLevelMap<Matrix>& F, RowMap<Matrix>& r) {
  int64_t parent_level = level - 1;
  for (int64_t block = 0; block < nblocks; ++block) {
    int64_t parent_node = block / 2;
    // Check for fill-ins along row
    bool found_row_fill_in = false;
    for (int64_t j = 0; j < nblocks; ++j) {
      if (F.exists(block, j, level)) {
        found_row_fill_in = true;
        break;
      }
    }
    // Update cluster bases if necessary
    if (found_row_fill_in) {
      update_row_cluster_bases(block, level, F, r);
      // Project admissible blocks accordingly
      // Current level: update coupling matrix
      for (int j = 0; j < nblocks; ++j) {
        if (is_admissible.exists(block, j, level) && is_admissible(block, j, level)) {
          S(block, j, level) = matmul(r(block), S(block, j, level));
        }
      }
      for (int i = 0; i < nblocks; ++i) {
        if (is_admissible.exists(i, block, level) && is_admissible(i, block, level)) {
          S(i, block, level) = matmul(S(i, block, level), r(block), false, true);
        }
      }
      // Upper levels: update transfer matrix one level higher
      if (row_has_admissible_blocks(parent_node, parent_level)) {
        int64_t c1 = parent_node * 2;
        int64_t c2 = parent_node * 2 + 1;
        Matrix& Utransfer = U(parent_node, parent_level);
        Matrix Utransfer_new(U(c1, level).cols + U(c2, level).cols, Utransfer.cols);
        auto Utransfer_new_splits = Utransfer_new.split(vec{U(c1, level).cols}, vec{});
        if (block == c1) {
          auto Utransfer_splits = Utransfer.split(vec{r(c1).cols}, vec{});
          matmul(r(c1), Utransfer_splits[0], Utransfer_new_splits[0], false, false, 1, 0);
          Utransfer_new_splits[1] = Utransfer_splits[1];
          r.erase(c1);
        }
        else { // block == c2
          auto Utransfer_splits = Utransfer.split(vec{U(c1, level).cols}, vec{});
          Utransfer_new_splits[0] = Utransfer_splits[0];
          matmul(r(c2), Utransfer_splits[1], Utransfer_new_splits[1], false, false, 1, 0);
          r.erase(c2);
        }
        U.erase(parent_node, parent_level);
        U.insert(parent_node, parent_level, std::move(Utransfer_new));
      }
    }

    // Multiplication with U_F
    Matrix U_F = prepend_complement(U(block, level));
    // Multiply to dense blocks along the row in current level
    for (int j = 0; j < nblocks; ++j) {
      if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) {
        if (j < block) {
          // Do not touch the eliminated part (cc and oc)
          int64_t left_col_split = D(block, j, level).cols - U(j, level).cols;
          auto D_splits = D(block, j, level).split(vec{}, vec{left_col_split});
          D_splits[1] = matmul(U_F, D_splits[1], true);
        }
        else {
          D(block, j, level) = matmul(U_F, D(block, j, level), true);
        }
      }
    }
    // Multiply to dense blocks along the column in current level
    for (int i = 0; i < nblocks; ++i) {
      if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) {
        if (i < block) {
          // Do not touch the eliminated part (cc and co)
          int64_t up_row_split = D(i, block, level).rows - U(i, level).cols;
          auto D_splits = D(i, block, level).split(vec{up_row_split}, vec{});
          D_splits[1] = matmul(D_splits[1], U_F);
        }
        else {
          D(i, block, level) = matmul(D(i, block, level), U_F);
        }
      }
    }
    // At non-leaf level, U(block, level) may not have orthonormal columns
    // So multiplication with U_F may update coupling or transfer matrices
    if (level < height && !found_row_fill_in) {
      Matrix UTxU = matmul(U(block, level), U(block, level), true, false);
      // Apply to S along the row in current level
      for (int j = 0; j < nblocks; ++j) {
        if (is_admissible.exists(block, j, level) && is_admissible(block, j, level)) {
          S(block, j, level) = matmul(UTxU, S(block, j, level));
        }
      }
      // Apply to S along the column in current level
      for (int i = 0; i < nblocks; ++i) {
        if (is_admissible.exists(i, block, level) && is_admissible(i, block, level)) {
          S(i, block, level) = matmul(S(i, block, level), UTxU);
        }
      }
      // Update transfer matrix
      if (row_has_admissible_blocks(parent_node, parent_level)) {
        int64_t c1 = parent_node * 2;
        int64_t c2 = parent_node * 2 + 1;
        Matrix& Utransfer = U(parent_node, parent_level);
        auto Utransfer_splits = Utransfer.split(vec{U(c1, level).cols}, vec{});
        Matrix temp(Utransfer);
        auto temp_splits = temp.split(vec{U(c1, level).cols}, vec{});
        if (block == c1) {
          matmul(UTxU, temp_splits[0], Utransfer_splits[0], false, false, 1, 0);
        }
        else { // block == c2
          matmul(UTxU, temp_splits[1], Utransfer_splits[1], false, false, 1, 0);
        }
      }
    }

    // The diagonal block is split along the row and column.
    int64_t diag_row_split = D(block, block, level).rows - U(block, level).cols;
    int64_t diag_col_split = D(block, block, level).cols - U(block, level).cols;
    auto diagonal_splits = D(block, block, level).split(vec{diag_row_split}, vec{diag_col_split});
    Matrix& Dcc = diagonal_splits[0];
    ldl(Dcc);

    // TRSM with cc blocks on the column
    for (int64_t i = block+1; i < nblocks; ++i) {
      if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) {
        int64_t lower_row_split = D(i, block, level).rows -
                                  (level == height ? U(i, level).cols : U(i * 2, level + 1).cols);
        auto D_splits = D(i, block, level).split(vec{lower_row_split}, vec{diag_col_split});
        solve_triangular(Dcc, D_splits[0], Hatrix::Right, Hatrix::Lower, true, true);
        solve_diagonal(Dcc, D_splits[0], Hatrix::Right);
      }
    }
    // TRSM with oc blocks on the column
    for (int64_t i = 0; i < nblocks; ++i) {
      if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) {
        int64_t lower_row_split = D(i, block, level).rows -
                                  (i <= block || level == height ?
                                   U(i, level).cols :
                                   U(i * 2, level + 1).cols);
        auto D_splits = D(i, block, level).split(vec{lower_row_split}, vec{diag_col_split});
        solve_triangular(Dcc, D_splits[2], Hatrix::Right, Hatrix::Lower, true, true);
        solve_diagonal(Dcc, D_splits[2], Hatrix::Right);
      }
    }

    // TRSM with cc blocks on the row
    for (int64_t j = block + 1; j < nblocks; ++j) {
      if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) {
        int64_t right_col_split = D(block, j, level).cols -
                                  (level == height ? U(j, level).cols : U(j * 2, level + 1).cols);
        auto D_splits = D(block, j, level).split(vec{diag_row_split}, vec{right_col_split});
        solve_triangular(Dcc, D_splits[0], Hatrix::Left, Hatrix::Lower, true, false);
        solve_diagonal(Dcc, D_splits[0], Hatrix::Left);
      }
    }
    // TRSM with co blocks on this row
    for (int64_t j = 0; j < nblocks; ++j) {
      if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) {
        int64_t right_col_split = D(block, j, level).cols -
                                  (j <= block || level == height ?
                                   U(j, level).cols :
                                   U(j * 2, level + 1).cols);
        auto D_splits = D(block, j, level).split(vec{diag_row_split}, vec{right_col_split});
        solve_triangular(Dcc, D_splits[1], Hatrix::Left, Hatrix::Lower, true, false);
        solve_diagonal(Dcc, D_splits[1], Hatrix::Left);
      }
    }

    // Schur's complement into dense block
    // cc x cc -> cc
    for (int64_t i = block+1; i < nblocks; ++i) {
      for (int64_t j = block+1; j < nblocks; ++j) {
        if ((is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) &&
            (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) &&
            (is_admissible.exists(i, j, level) && !is_admissible(i, j, level))) {
          int64_t lower_row_rank =
              level == height ? U(i, level).cols : U(i * 2, level + 1).cols;
          int64_t right_col_rank =
              level == height ? U(j, level).cols : U(j * 2, level + 1).cols;
          auto lower_splits = D(i, block, level).split(
              vec{D(i, block, level).rows - lower_row_rank}, vec{diag_col_split});
          auto right_splits = D(block, j, level).split(
              vec{diag_row_split}, vec{D(block, j, level).cols - right_col_rank});
          auto reduce_splits = D(i, j, level).split(
              vec{D(i, j, level).rows - lower_row_rank},
              vec{D(i, j, level).cols - right_col_rank});

          Matrix lower0_scaled(lower_splits[0], true);
          column_scale(lower0_scaled, Dcc);
          // Update cc part
          matmul(lower0_scaled, right_splits[0], reduce_splits[0],
                 false, false, -1.0, 1.0);
        }
      }
    }
    // Schur's complement into dense block
    // cc x co -> co
    for (int64_t i = block+1; i < nblocks; ++i) {
      for (int64_t j = 0; j < nblocks; ++j) {
        if ((is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) &&
            (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) &&
            (is_admissible.exists(i, j, level) && !is_admissible(i, j, level))) {
          int64_t lower_row_rank =
              level == height ? U(i, level).cols : U(i * 2, level + 1).cols;
          int64_t right_col_rank =
              (j <= block || level == height) ? U(j, level).cols : U(j * 2, level + 1).cols;
          auto lower_splits = D(i, block, level).split(
              vec{D(i, block, level).rows - lower_row_rank}, vec{diag_col_split});
          auto right_splits = D(block, j, level).split(
              vec{diag_row_split}, vec{D(block, j, level).cols - right_col_rank});
          auto reduce_splits = D(i, j, level).split(
              vec{D(i, j, level).rows - lower_row_rank},
              vec{D(i, j, level).cols - right_col_rank});

          Matrix lower0_scaled(lower_splits[0], true);
          column_scale(lower0_scaled, Dcc);
          // Update co part
          matmul(lower0_scaled, right_splits[1], reduce_splits[1],
                 false, false, -1.0, 1.0);
        }
      }
    }
    // Schur's complement into dense block
    // oc x cc -> oc
    for (int64_t i = 0; i < nblocks; ++i) {
      for (int64_t j = block+1; j < nblocks; ++j) {
        if ((is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) &&
            (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) &&
            (is_admissible.exists(i, j, level) && !is_admissible(i, j, level))) {
          int64_t lower_row_rank =
              (i <= block || level == height) ? U(i, level).cols : U(i * 2, level + 1).cols;
          int64_t right_col_rank =
              level == height ? U(j, level).cols : U(j * 2, level + 1).cols;
          auto lower_splits = D(i, block, level).split(
              vec{D(i, block, level).rows - lower_row_rank}, vec{diag_col_split});
          auto right_splits = D(block, j, level).split(
              vec{diag_row_split}, vec{D(block, j, level).cols - right_col_rank});
          auto reduce_splits = D(i, j, level).split(
              vec{D(i, j, level).rows - lower_row_rank},
              vec{D(i, j, level).cols - right_col_rank});

          Matrix lower2_scaled(lower_splits[2], true);
          column_scale(lower2_scaled, Dcc);
          // Update oc part
          matmul(lower2_scaled, right_splits[0], reduce_splits[2],
                 false, false, -1.0, 1.0);
        }
      }
    }
    // Schur's complement into dense block
    // oc x co -> oo
    for (int64_t i = 0; i < nblocks; ++i) {
      for (int64_t j = 0; j < nblocks; ++j) {
        if ((is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) &&
            (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) &&
            (is_admissible.exists(i, j, level) && !is_admissible(i, j, level))) {
          int64_t lower_row_rank =
              (i <= block || level == height) ? U(i, level).cols : U(i * 2, level + 1).cols;
          int64_t right_col_rank =
              (j <= block || level == height) ? U(j, level).cols : U(j * 2, level + 1).cols;
          auto lower_splits = D(i, block, level).split(
              vec{D(i, block, level).rows - lower_row_rank}, vec{diag_col_split});
          auto right_splits = D(block, j, level).split(
              vec{diag_row_split}, vec{D(block, j, level).cols - right_col_rank});
          auto reduce_splits = D(i, j, level).split(
              vec{D(i, j, level).rows - lower_row_rank},
              vec{D(i, j, level).cols - right_col_rank});

          Matrix lower2_scaled(lower_splits[2], true);
          column_scale(lower2_scaled, Dcc);
          // Update oo part
          matmul(lower2_scaled, right_splits[1], reduce_splits[3],
                 false, false, -1.0, 1.0);
        }
      }
    }

    // Schur's complement into low-rank block (fill-in)
    // Produces b*b fill-in
    for (int64_t i = block+1; i < nblocks; ++i) {
      for (int64_t j = block+1; j < nblocks; ++j) {
        if ((is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) &&
            (is_admissible.exists(block, j, level) && !is_admissible(block, j, level))) {
          int64_t lower_row_rank =
              level == height ? U(i, level).cols : U(i * 2, level + 1).cols;
          int64_t right_col_rank =
              level == height ? U(j, level).cols : U(j * 2, level + 1).cols;
          auto lower_splits = D(i, block, level).split(
              vec{D(i, block, level).rows - lower_row_rank}, vec{diag_col_split});
          auto right_splits = D(block, j, level).split(
              vec{diag_row_split}, vec{D(block, j, level).cols - right_col_rank});
          Matrix lower0_scaled(lower_splits[0], true);
          Matrix lower2_scaled(lower_splits[2], true);
          column_scale(lower0_scaled, Dcc);
          column_scale(lower2_scaled, Dcc);

          if ((!is_admissible.exists(i, j, level)) ||
              (is_admissible.exists(i, j, level) && is_admissible(i, j, level))) {
            // Create b*b fill-in block
            int64_t nrows = D(i, block, level).rows;
            int64_t ncols = D(block, j, level).cols;
            Matrix fill_in(nrows, ncols);
            auto fill_in_splits = fill_in.split(vec{nrows - lower_row_rank},
                                                vec{ncols - right_col_rank});
            // Fill cc part
            matmul(lower0_scaled, right_splits[0], fill_in_splits[0],
                   false, false, -1.0, 1.0);
            // Fill co part
            matmul(lower0_scaled, right_splits[1], fill_in_splits[1],
                   false, false, -1.0, 1.0);
            // Fill oc part
            matmul(lower2_scaled, right_splits[0], fill_in_splits[2],
                   false, false, -1.0, 1.0);
            // Fill oo part
            matmul(lower2_scaled, right_splits[1], fill_in_splits[3],
                   false, false, -1.0, 1.0);

            // Save or accumulate with existing fill-in
            if (!F.exists(i, j, level)) {
              F.insert(i, j, level, std::move(fill_in));
            }
            else {
              assert(F(i, j, level).rows == D(i, block, level).rows);
              assert(F(i, j, level).cols == D(block, j, level).cols);
              F(i, j, level) += fill_in;
            }
          }
        }
      }
    }
    // Schur's complement into low-rank block (fill-in)
    // Produces b*rank fill-in
    for (int64_t i = block+1; i < nblocks; ++i) {
      for (int64_t j = 0; j < block; ++j) {
        if ((is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) &&
            (is_admissible.exists(block, j, level) && !is_admissible(block, j, level))) {
          int64_t lower_row_rank =
              level == height ? U(i, level).cols : U(i * 2, level + 1).cols;
          int64_t right_col_rank = U(j, level).cols;
          auto lower_splits = D(i, block, level).split(
              vec{D(i, block, level).rows - lower_row_rank}, vec{diag_col_split});
          auto right_splits = D(block, j, level).split(
              vec{diag_row_split}, vec{D(block, j, level).cols - right_col_rank});
          Matrix lower0_scaled(lower_splits[0], true);
          Matrix lower2_scaled(lower_splits[2], true);
          column_scale(lower0_scaled, Dcc);
          column_scale(lower2_scaled, Dcc);

          if ((!is_admissible.exists(i, j, level)) ||
              (is_admissible.exists(i, j, level) && is_admissible(i, j, level))) {
            // Create b*rank fill-in block
            int64_t nrows = D(i, block, level).rows;
            int64_t ncols = right_col_rank;
            Matrix fill_in(nrows, ncols);
            auto fill_in_splits = fill_in.split(vec{nrows - lower_row_rank},
                                                vec{});
            // Fill co part
            matmul(lower0_scaled, right_splits[1], fill_in_splits[0],
                   false, false, -1.0, 1.0);
            // Fill oo part
            matmul(lower2_scaled, right_splits[1], fill_in_splits[1],
                   false, false, -1.0, 1.0);

            // b*rank fill-in always has a form of Aik*Vk_c * inv(Akk_cc) x (Uk_c)^T*Akj*Vj_o
            // Convert to b*b block by applying (Vj_o)^T from right
            // Which is safe from bases update since j has been eliminated before (j < k)
            Matrix projected_fill_in = matmul(fill_in, U(j, level), false, true);

            // Save or accumulate with existing fill-in
            if (!F.exists(i, j, level)) {
              F.insert(i, j, level, std::move(projected_fill_in));
            }
            else {
              assert(F(i, j, level).rows == D(i, block, level).rows);
              assert(F(i, j, level).cols == D(block, j, level).cols);
              F(i, j, level) += projected_fill_in;
            }
          }
        }
      }
    }
    // Schur's complement into low-rank block (fill-in)
    // Produces rank*b fill-in
    for (int64_t i = 0; i < block; ++i) {
      for (int64_t j = block+1; j < nblocks; ++j) {
        if ((is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) &&
            (is_admissible.exists(block, j, level) && !is_admissible(block, j, level))) {
          int64_t lower_row_rank = U(i, level).cols;
          int64_t right_col_rank =
              level == height ? U(j, level).cols : U(j * 2, level + 1).cols;
          auto lower_splits = D(i, block, level).split(
              vec{D(i, block, level).rows - lower_row_rank}, vec{diag_col_split});
          auto right_splits = D(block, j, level).split(
              vec{diag_row_split}, vec{D(block, j, level).cols - right_col_rank});
          Matrix lower2_scaled(lower_splits[2], true);
          column_scale(lower2_scaled, Dcc);

          if ((!is_admissible.exists(i, j, level)) ||
              (is_admissible.exists(i, j, level) && is_admissible(i, j, level))) {
            // Create rank*b fill-in block
            int64_t nrows = lower_row_rank;
            int64_t ncols = D(block, j, level).cols;
            Matrix fill_in(nrows, ncols);
            auto fill_in_splits = fill_in.split(vec{},
                                                vec{ncols - right_col_rank});
            // Fill oc part
            matmul(lower2_scaled, right_splits[0], fill_in_splits[0],
                   false, false, -1.0, 1.0);
            // Fill oo part
            matmul(lower2_scaled, right_splits[1], fill_in_splits[1],
                   false, false, -1.0, 1.0);

            // rank*b fill-in always has a form of (Ui_o)^T*Aik*Vk_c * inv(Akk_cc) * (Uk_c)^T*A_kj
            // Convert to b*b block by applying Ui_o from left
            // Which is safe from bases update since i has been eliminated before (i < k)
            Matrix projected_fill_in = matmul(U(i, level), fill_in);

            // Save or accumulate with existing fill-in
            if (!F.exists(i, j, level)) {
              F.insert(i, j, level, std::move(projected_fill_in));
            }
            else {
              assert(F(i, j, level).rows == D(i, block, level).rows);
              assert(F(i, j, level).cols == D(block, j, level).cols);
              F(i, j, level) += projected_fill_in;
            }
          }
        }
      }
    }
    // Schur's complement into low-rank block (fill-in)
    // Produces rank*rank fill-in
    for (int64_t i = 0; i < block; ++i) {
      for (int64_t j = 0; j < block; ++j) {
        if ((is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) &&
            (is_admissible.exists(block, j, level) && !is_admissible(block, j, level))) {
          int64_t lower_row_rank = U(i, level).cols;
          int64_t right_col_rank = U(j, level).cols;
          auto lower_splits = D(i, block, level).split(
              vec{D(i, block, level).rows - lower_row_rank}, vec{diag_col_split});
          auto right_splits = D(block, j, level).split(
              vec{diag_row_split}, vec{D(block, j, level).cols - right_col_rank});
          Matrix lower2_scaled(lower_splits[2], true);
          column_scale(lower2_scaled, Dcc);

          if ((!is_admissible.exists(i, j, level)) ||
              (is_admissible.exists(i, j, level) && is_admissible(i, j, level))) {
            // Create rank*rank fill-in block
            int64_t nrows = lower_row_rank;
            int64_t ncols = right_col_rank;
            Matrix fill_in(nrows, ncols);
            // Fill oo part
            matmul(lower2_scaled, right_splits[1], fill_in,
                   false, false, -1.0, 1.0);

            // rank*rank fill-in always has a form of (Ui_o)^T*Aik*Vk_c * inv(Akk_cc) * (Uk_c)^T*A_kj*Vj_o
            // Convert to b*b block by applying Ui_o from left and (Vj_o)^T from right
            // Which is safe from bases update since i and j have been eliminated before (i,j < k)
            Matrix projected_fill_in = matmul(matmul(U(i, level), fill_in),
                                              U(j, level), false, true);

            // Save or accumulate with existing fill-in
            if (!F.exists(i, j, level)) {
              F.insert(i, j, level, std::move(projected_fill_in));
            }
            else {
              assert(F(i, j, level).rows == D(i, block, level).rows);
              assert(F(i, j, level).cols == D(block, j, level).cols);
              F(i, j, level) += projected_fill_in;
            }
          }
        }
      }
    }
  } // for (int block = 0; block < nblocks; ++block)
}

void SymmetricH2::factorize(const Domain& domain) {
  int64_t level = height;
  RowColLevelMap<Matrix> F;
  RowMap<Matrix> r;

  for (; level > 0; --level) {
    int64_t nblocks = level_blocks[level];
    // Make sure all cluster bases exist and none of them is full-rank
    for (int64_t i = 0; i < nblocks; ++i) {
      if (!U.exists(i, level)) {
        throw std::logic_error("Cluster bases not found at U(" + std::to_string(i) +
                               "," + std::to_string(level) + ")");
      }
      if (U(i, level).rows <= U(i, level).cols) {
        throw std::domain_error("Full rank cluster bases found at U(" + std::to_string(i) +
                                "," + std::to_string(level) + ")");
      }
    }

    factorize_level(domain, level, nblocks, F, r);

    // Update coupling matrices of admissible blocks in the current level
    // To ad fill-in contributions
    for (int64_t i = 0; i < nblocks; ++i) {
      for (int64_t j = 0; j < nblocks; ++j) {
        if (is_admissible.exists(i, j, level) && is_admissible(i, j, level)) {
          if (F.exists(i, j, level)) {
            Matrix projected_fill_in = matmul(matmul(U(i, level), F(i, j, level), true),
                                              U(j, level));
            S(i, j, level) += projected_fill_in;
          }
        }
      }
    }

    int64_t parent_level = level - 1;
    int64_t parent_nblocks = level_blocks[parent_level];
    // Propagate fill-in to upper level admissible blocks (if any)
    if (parent_level > 0) {
      for (int64_t i = 0; i < parent_nblocks; ++i) {
        for (int64_t j = 0; j < parent_nblocks; ++j) {
          if ((!is_admissible.exists(i, j, parent_level)) ||
              (is_admissible.exists(i, j, parent_level) && is_admissible(i, j, parent_level))) {
            int64_t i1 = i * 2;
            int64_t i2 = i * 2 + 1;
            int64_t j1 = j * 2;
            int64_t j2 = j * 2 + 1;
            if (F.exists(i1, j1, level) || F.exists(i1, j2, level) ||
                F.exists(i2, j1, level) || F.exists(i2, j2, level)) {
              int64_t nrows = U(i1, level).cols + U(i2, level).cols;
              int64_t ncols = U(j1, level).cols + U(j2, level).cols;
              Matrix fill_in(nrows, ncols);
              auto fill_in_splits = fill_in.split(vec{U(i1, level).cols},
                                                  vec{U(j1, level).cols});
              if (F.exists(i1, j1, level)) {
                matmul(matmul(U(i1, level), F(i1, j1, level), true, false),
                       U(j1, level), fill_in_splits[0], false, false, 1, 0);
              }
              if (F.exists(i1, j2, level)) {
                matmul(matmul(U(i1, level), F(i1, j2, level), true, false),
                       U(j2, level), fill_in_splits[1], false, false, 1, 0);
              }
              if (F.exists(i2, j1, level)) {
                matmul(matmul(U(i2, level), F(i2, j1, level), true, false),
                       U(j1, level), fill_in_splits[2], false, false, 1, 0);
              }
              if (F.exists(i2, j2, level)) {
                matmul(matmul(U(i2, level), F(i2, j2, level), true, false),
                       U(j2, level), fill_in_splits[3], false, false, 1, 0);
              }
              F.insert(i, j, parent_level, std::move(fill_in));
            }
          }
        }
      }
      // Put identity bases when all dense is encountered in parent level
      for (int64_t block = 0; block < nblocks; block += 2) {
        int64_t parent_node = block / 2;
        if (!U.exists(parent_node, parent_level)) {
          // Use identity matrix as U bases whenever all dense row is encountered
          int64_t c1 = block;
          int64_t c2 = block + 1;
          int64_t rank_c1 = U(c1, level).cols;
          int64_t rank_c2 = U(c2, level).cols;
          int64_t rank_parent = std::max(rank_c1, rank_c2);
          Matrix Utransfer =
              generate_identity_matrix(rank_c1 + rank_c2, rank_parent);
        
          U.insert(parent_node, parent_level, std::move(Utransfer));
        }
      }
    }

    // Merge the unfactorized parts.
    for (int64_t i = 0; i < parent_nblocks; ++i) {
      for (int64_t j = 0; j < parent_nblocks; ++j) {
        if (is_admissible.exists(i, j, parent_level) && !is_admissible(i, j, parent_level)) {
          // TODO: need to switch to morton indexing so finding the parent is straightforward.
          std::vector<int64_t> i_children, j_children;
          std::vector<int64_t> row_split, col_split;
          int64_t nrows=0, ncols=0;
          if (matrix_type == BLR2_MATRIX) {
            for (int64_t n = 0; n < level_blocks[level]; ++n) {
              i_children.push_back(n);
              j_children.push_back(n);

              nrows += U(n, level).cols;
              ncols += U(n, level).cols;
              if(n < (level_blocks[level] - 1)) {
                row_split.push_back(nrows);
                col_split.push_back(ncols);
              }
            }
          }
          else if (matrix_type == H2_MATRIX) {
            for (int64_t n = 0; n < 2; ++n) {
              int64_t ic = i * 2 + n;
              int64_t jc = j * 2 + n;
              i_children.push_back(ic);
              j_children.push_back(jc);

              nrows += U(ic, level).cols;
              ncols += U(jc, level).cols;
              if(n < 1) {
                row_split.push_back(nrows);
                col_split.push_back(ncols);
              }
            }
          }
          Matrix D_unelim(nrows, ncols);
          auto D_unelim_splits = D_unelim.split(row_split, col_split);

          for (int64_t ic1 = 0; ic1 < i_children.size(); ++ic1) {
            for (int64_t jc2 = 0; jc2 < j_children.size(); ++jc2) {
              int64_t c1 = i_children[ic1], c2 = j_children[jc2];
              if (!U.exists(c1, level)) { continue; }

              if (is_admissible.exists(c1, c2, level) && !is_admissible(c1, c2, level)) {
                auto D_splits = D(c1, c2, level).split(
                    vec{D(c1, c2, level).rows - U(c1, level).cols},
                    vec{D(c1, c2, level).cols - U(c2, level).cols});
                D_unelim_splits[ic1 * j_children.size() + jc2] = D_splits[3];
              }
              else {
                D_unelim_splits[ic1 * j_children.size() + jc2] = S(c1, c2, level);
              }
            }
          }

          D.insert(i, j, parent_level, std::move(D_unelim));
        }
      }
    }
  } // for (; level > 0; --level)

  // Factorize remaining root level
  ldl(D(0, 0, level));
}

void SymmetricH2::print_ranks() {
  for(int64_t level = height; level > 0; level--) {
    int64_t nblocks = level_blocks[level];
    for(int64_t block = 0; block < nblocks; block++) {
      std::cout << "block=" << block << "," << "level=" << level << ":\t"
                << "diag= ";
      if(D.exists(block, block, level)) {
        std::cout << D(block, block, level).rows << "x" << D(block, block, level).cols;
      }
      else {
        std::cout << "empty";
      }
      std::cout << ", row_rank=" << (U.exists(block, level) ?
                                     U(block, level).cols : -1)
                << ", col_rank=" << (U.exists(block, level) ?
                                     U(block, level).cols : -1)
                << std::endl;
    }
  }
}

std::tuple<int64_t, int64_t>
SymmetricH2::inertia(const Domain& domain, const double lambda, bool &singular) {
  SymmetricH2 A_shifted(*this);
  // Shift leaf level diagonal blocks
  int64_t leaf_nblocks = level_blocks[height];
  for(int64_t block = 0; block < leaf_nblocks; block++) {
    shift_diag(A_shifted.D(block, block, height), -lambda);
  }
  // LDL Factorize
  A_shifted.factorize(domain);
  // // Gather values in D
  Matrix D_lambda(0, 0);
  for(int64_t level = height; level >= 0; level--) {
    int64_t nblocks = level_blocks[level];
    for(int64_t block = 0; block < nblocks; block++) {
      if(level == 0) {
        D_lambda = concat(D_lambda, diag(A_shifted.D(block, block, level)), 0);
      }
      else {
        int64_t rank = A_shifted.U(block, level).cols;
        int64_t row_split = A_shifted.D(block, block, level).rows - rank;
        int64_t col_split = A_shifted.D(block, block, level).cols - rank;
        auto D_splits = A_shifted.D(block, block, level).split(vec{row_split},
                                                               vec{col_split});
        D_lambda = concat(D_lambda, diag(D_splits[0]), 0);
      }
    }
  }
  int64_t negative_elements_count = 0;
  for(int64_t i = 0; i < D_lambda.rows; i++) {
    negative_elements_count += (D_lambda(i, 0) < 0 ? 1 : 0);
    if(std::isnan(D_lambda(i, 0)) || std::abs(D_lambda(i, 0)) < EPS) singular = true;
  }
  return {negative_elements_count, A_shifted.max_rank};
}

std::tuple<double, int64_t, double>
SymmetricH2::get_mth_eigenvalue(const Domain& domain, const int64_t m,
                                const double ev_tol,
                                double left, double right) {
  int64_t shift_max_rank = max_rank;
  double max_rank_shift = -1;
  bool singular = false;
  while((right - left) >= ev_tol) {
    const auto mid = (left + right) / 2;
    int64_t value, factor_max_rank;
    std::tie(value, factor_max_rank) = (*this).inertia(domain, mid, singular);
    if(singular) {
      std::cout << "Shifted matrix became singular (shift=" << mid << ")" << std::endl;
      break;
    }
    if(factor_max_rank >= shift_max_rank) {
      shift_max_rank = factor_max_rank;
      max_rank_shift = mid;
    }
    if(value >= m) right = mid;
    else left = mid;
  }
  return {(left + right) / 2, shift_max_rank, max_rank_shift};
}

} // namespace Hatrix

int64_t inertia(const Hatrix::Matrix& A, const double lambda, bool& singular) {
  Hatrix::Matrix Ac(A);
  shift_diag(Ac, -lambda);
  Hatrix::ldl(Ac);
  int64_t negative_elements_count = 0;
  bool zero_found = false;
  for(int64_t i = 0; i < Ac.min_dim(); i++) {
    negative_elements_count += (Ac(i, i) < 0 ? 1 : 0);
    if(std::isnan(Ac(i, i)) || std::abs(Ac(i, i)) < EPS) singular = true;
  }
  return negative_elements_count;
}

double get_mth_eigenvalue(const Hatrix::Matrix& A, const int64_t m,
                          const double ev_tol,
                          double left, double right) {
  bool singular = false;
  while((right - left) >= ev_tol) {
    const auto mid = (left + right) / 2;
    const auto value = inertia(A, mid, singular);
    if(singular) {
      std::cout << "Shifted matrix became singular (at shift=" << mid << ")" << std::endl;
      break;
    }
    if(value >= m) right = mid;
    else left = mid;
  }
  return (left + right) / 2;
}

int main(int argc, char ** argv) {
  const int64_t N = argc > 1 ? atoi(argv[1]) : 256;
  const int64_t nleaf = argc > 2 ? atoi(argv[2]) : 32;
  const double accuracy = argc > 3 ? atof(argv[3]) : 1e-5;
  const int64_t rank = argc > 4 ? atoi(argv[4]) : 0;
  const double admis = argc > 5 ? atof(argv[5]) : 1.0;
  // diagonal_admis or geometry_admis
  const std::string admis_kind = argc > 6 ? std::string(argv[6]) : "diagonal_admis";
  const int64_t ndim  = argc > 7 ? atoi(argv[7]) : 2;
  // 0: BLR2
  // 1: H2
  const int64_t matrix_type = argc > 8 ? atoi(argv[8]) : 1;
  const double ev_tol = argc > 9 ? atof(argv[9]) : 1e-5;
  int64_t m  = argc > 10 ? atoi(argv[10]) : 0;
  PV = (1/(double)N) * 1e-2;

  Hatrix::Context::init();

  const auto start_particles = std::chrono::system_clock::now();
  Hatrix::Domain domain(N, ndim);
  // Laplace kernel
  domain.generate_particles(0, N);
  // domain.generate_starsh_grid_particles();
  Hatrix::kernel_function = Hatrix::laplace_kernel;
  domain.divide_domain_and_create_particle_boxes(nleaf);
  const auto stop_particles = std::chrono::system_clock::now();
  const double particle_construct_time = std::chrono::duration_cast<std::chrono::milliseconds>
                                         (stop_particles - start_particles).count();

  const int64_t oversampling = 5;
  const int64_t sample_size = rank > 0 ? rank + oversampling : 0;
  Hatrix::Matrix rand = Hatrix::generate_random_matrix(N, sample_size);
  const auto start_construct = std::chrono::system_clock::now();
  Hatrix::SymmetricH2 A(domain, N, nleaf, accuracy, rank, admis, admis_kind, matrix_type, rand);
  const auto stop_construct = std::chrono::system_clock::now();
  const double construct_time = std::chrono::duration_cast<std::chrono::milliseconds>
                                (stop_construct - start_construct).count();  
  double construct_error = A.construction_absolute_error(domain);
  double lr_ratio = A.low_rank_block_ratio();

  std::cout << "N=" << N
            << " nleaf=" << nleaf
            << " accuracy=" << accuracy
            << " rank=" << rank
            << " admis=" << admis << std::setw(3)
            << " ndim=" << ndim
            << " height=" << A.height
            << " admis_kind=" << admis_kind
            << " matrix_type=" << (matrix_type == BLR2_MATRIX ? "BLR2" : "H2")
            << " LR%=" << lr_ratio * 100 << "%"
            << " construct_min_rank=" << A.min_rank
            << " construct_max_rank=" << A.max_rank
            << " construct_time=" << construct_time
            << std::scientific
            << " construct_error=" << construct_error
            << std::fixed
            << std::endl;

  Hatrix::Matrix Adense = Hatrix::generate_p2p_matrix(domain);
  auto lapack_eigv = Hatrix::get_eigenvalues(Adense);

  bool s = false;
  const auto b = 10 * (1 / PV);
  const auto a = -b;
  int64_t v_a, v_b, _;
  std::tie(v_a, _) = A.inertia(domain, a, s);
  std::tie(v_b, _) = A.inertia(domain, b, s);
  if(v_a != 0 || v_b != N) {
    std::cerr << "Warning: starting interval does not contain the whole spectrum" << std::endl
              << "at N=" << N << ",nleaf=" << nleaf << ",accuracy=" << accuracy
              << ",admis=" << admis << ",b=" << b << std::endl;
  }

  if (m == 0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int64_t> dist(1, N);
    bool success = true;
    for (int64_t k = 1; k <= 10; k++) {
      m = dist(gen);
      // m = k;
      double h2_mth_eigv, max_rank_shift;
      int64_t factor_max_rank;
      const auto eig_start = std::chrono::system_clock::now();
      std::tie(h2_mth_eigv, factor_max_rank, max_rank_shift) =
          A.get_mth_eigenvalue(domain, m, ev_tol, a, b);
      const auto eig_stop = std::chrono::system_clock::now();
      const double eig_time = std::chrono::duration_cast<std::chrono::milliseconds>
                              (eig_stop - eig_start).count();
      double eig_abs_err = std::abs(h2_mth_eigv - lapack_eigv[m - 1]);
      success = (eig_abs_err < (0.5 * ev_tol));
      std::cout << "m=" << m
                << " ev_tol=" << ev_tol
                << " eig_time=" << eig_time
                << " factor_max_rank=" << factor_max_rank
                << " max_rank_shift=" << max_rank_shift
                << " lapack_eigv=" << lapack_eigv[m - 1]
                << " h2_eigv=" << h2_mth_eigv
                << " eig_abs_err=" << std::scientific << eig_abs_err << std::fixed
                << " success=" << (success ? "TRUE" : "FALSE")
                << std::endl;
      if (!success) break;
    }
    std::cout << (success ? "SUCCESS" : "FAILED") << std::endl;
  }
  else {
    double h2_mth_eigv, max_rank_shift;
    int64_t factor_max_rank;
    const auto eig_start = std::chrono::system_clock::now();
    std::tie(h2_mth_eigv, factor_max_rank, max_rank_shift) =
        A.get_mth_eigenvalue(domain, m, ev_tol, a, b);
    const auto eig_stop = std::chrono::system_clock::now();
    const double eig_time = std::chrono::duration_cast<std::chrono::milliseconds>
                            (eig_stop - eig_start).count();
    double eig_abs_err = std::abs(h2_mth_eigv - lapack_eigv[m - 1]);
    bool success = (eig_abs_err < (0.5 * ev_tol));
    std::cout << "m=" << m
              << " ev_tol=" << ev_tol
              << " eig_time=" << eig_time
              << " factor_max_rank=" << factor_max_rank
              << " max_rank_shift=" << max_rank_shift
              << " lapack_eigv=" << lapack_eigv[m - 1]
              << " h2_eigv=" << h2_mth_eigv
              << " eig_abs_err=" << std::scientific << eig_abs_err << std::fixed
              << " success=" << (success ? "TRUE" : "FALSE")
              << std::endl;
  }

  Hatrix::Context::finalize();
  return 0;
}