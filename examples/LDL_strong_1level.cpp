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
constexpr int64_t oversampling = 5;
double PV = 1e-3;
enum MATRIX_TYPES {BLR2_MATRIX=0, H2_MATRIX=1};

Hatrix::Matrix prepend_complement(const Hatrix::Matrix &Q) {
  Hatrix::Matrix Q_F(Q.rows, Q.rows);
  Hatrix::Matrix Q_full, R;
  std::tie(Q_full, R) = qr(Q, Hatrix::Lapack::QR_mode::Full, Hatrix::Lapack::QR_ret::OnlyQ);

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

class BLR2_SPD {
 public:
  int64_t N, nleaf, n_blocks, rank;
  double admis;
  RowLevelMap U;
  RowLevelMap Scol;
  RowColLevelMap<Matrix> D, S;
  RowColLevelMap<bool> is_admissible;
  std::string admis_kind;
  std::vector<int64_t> level_blocks;
  const int64_t height = 1;
  const int64_t matrix_type = BLR2_MATRIX;

 private:
  bool all_dense_blocks(int64_t level, int64_t nblocks);
  int64_t permute_forward(Matrix& x, const int64_t level, int64_t rank_offset);
  int64_t permute_backward(Matrix& x, const int64_t level, int64_t rank_offset);
  void solve_forward_level(Matrix& x_level, int64_t level);
  void solve_backward_level(Matrix& x_level, int64_t level);

  void generate_leaf_nodes(const Domain& domain);
  void actually_print_structure(int64_t level);
  bool row_has_admissible_blocks(int64_t row, int64_t level);
  bool col_has_admissible_blocks(int64_t col, int64_t level);
  Matrix generate_column_block(int64_t block, int64_t block_size,
                               const Domain& domain, int64_t level);
  std::tuple<Matrix, Matrix>
  generate_column_bases(int64_t block, int64_t block_size, const Domain& domain,
                        std::vector<Matrix>& Y, int64_t level);

  Matrix get_Ubig(int64_t node, int64_t level);

  void calc_geometry_based_admissibility(const Domain& domain);
  int64_t get_block_size_col(const Domain& domain, int64_t parent, int64_t level);
  void factorize_level(int64_t level, int64_t nblocks, const Domain& domain, RowMap& r);
  int64_t find_all_dense_row();
  void update_row_basis(int64_t row, int64_t level, RowColMap<Matrix>& F, RowMap& r);

 public:
  BLR2_SPD(const Domain& domain, const int64_t N, const int64_t rank,
           const int64_t nleaf, const double admis, const std::string& admis_kind);
  double construction_relative_error(const Domain& domain);
  void print_structure();
  double low_rank_block_ratio();
  void factorize(const Domain& domain);
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

bool BLR2_SPD::all_dense_blocks(int64_t level, int64_t nblocks) {
  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < nblocks; ++j) {
      if (!is_admissible.exists(i, j, level) || is_admissible(i, j, level)) {
        return false;
      }
    }
  }
  return true;
}

int64_t BLR2_SPD::find_all_dense_row() {
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

void BLR2_SPD::calc_geometry_based_admissibility(const Domain& domain) {
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
  level_blocks.push_back(1);
}

Matrix BLR2_SPD::generate_column_block(int64_t block, int64_t block_size,
                                       const Domain& domain, int64_t level) {
  int ncols = 0;
  int num_blocks = 0;
  for (int64_t j = 0; j < level_blocks[level]; ++j) {
    if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) { continue; }
    ncols += get_block_size_col(domain, j, level);
    num_blocks++;
  }

  Matrix AY(block_size, ncols);
  auto AY_splits = AY.split(1, num_blocks);

  int index = 0;
  for (int64_t j = 0; j < level_blocks[level]; ++j) {
    if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) { continue; }
    Hatrix::generate_p2p_interactions(domain, block, j, level, height, AY_splits[index++]);
  }

  return AY;
}

std::tuple<Matrix, Matrix> BLR2_SPD::generate_column_bases(int64_t block, int64_t block_size,
                                                           const Domain& domain,
                                                           std::vector<Matrix>& Y, int64_t level) {
  // Row slice since column bases should be cutting across the columns.
  Matrix AY = generate_column_block(block, block_size, domain, level);
  Matrix Ui, Si, Vi; double error;
  std::tie(Ui, Si, Vi, error) = truncated_svd(AY, rank);

  return {std::move(Ui), std::move(Si)};
}

void BLR2_SPD::generate_leaf_nodes(const Domain& domain) {
  int64_t nblocks = level_blocks[height];
  std::vector<Hatrix::Matrix> Y;

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < nblocks; ++j) {
      if (is_admissible.exists(i, j, height) && !is_admissible(i, j, height)) {
        D.insert(i, j, height,
                 generate_p2p_interactions(domain, i, j));
      }
    }
  }

  for (int64_t i = 0; i < nblocks; ++i) {
    Y.push_back(generate_random_matrix(domain.boxes[i].num_particles, rank + oversampling));
  }

  // Generate U leaf blocks
  for (int64_t i = 0; i < nblocks; ++i) {
    Matrix Utemp, Stemp;
    std::tie(Utemp, Stemp) =
        generate_column_bases(i, domain.boxes[i].num_particles, domain, Y, height);
    U.insert(i, height, std::move(Utemp));
    Scol.insert(i, height, std::move(Stemp));
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

Matrix BLR2_SPD::get_Ubig(int64_t node, int64_t level) {
  if (level == height) {
    return U(node, level);
  }

  int64_t child1 = node * 2;
  int64_t child2 = node * 2 + 1;

  Matrix Ubig_child1 = get_Ubig(child1, level+1);
  Matrix Ubig_child2 = get_Ubig(child2, level+1);

  int64_t block_size = Ubig_child1.rows + Ubig_child2.rows;

  Matrix Ubig(block_size, rank);

  std::vector<Matrix> Ubig_splits = Ubig.split(vec{Ubig_child1.rows}, vec{});

  std::vector<Matrix> U_splits = U(node, level).split(2, 1);

  matmul(Ubig_child1, U_splits[0], Ubig_splits[0]);
  matmul(Ubig_child2, U_splits[1], Ubig_splits[1]);

  return Ubig;

}

int64_t BLR2_SPD::get_block_size_col(const Domain& domain, int64_t parent, int64_t level) {
  if (level == height) {
    return domain.boxes[parent].num_particles;
  }
  int64_t child_level = level + 1;
  int64_t child1 = parent * 2;
  int64_t child2 = parent * 2 + 1;

  return get_block_size_col(domain, child1, child_level) +
      get_block_size_col(domain, child2, child_level);
}

bool BLR2_SPD::row_has_admissible_blocks(int64_t row, int64_t level) {
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

bool BLR2_SPD::col_has_admissible_blocks(int64_t col, int64_t level) {
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

BLR2_SPD::BLR2_SPD(const Domain& domain, const int64_t N, const int64_t rank,
                   const int64_t nleaf, const double admis, const std::string& admis_kind)
    : N(N), rank(rank), nleaf(nleaf), admis(admis), admis_kind(admis_kind) {
  if (admis_kind == "geometry_admis") {
    // TODO: use dual tree traversal for this.
    calc_geometry_based_admissibility(domain);
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
    int64_t nblocks = domain.boxes.size();
    for (int64_t i = 0; i < nblocks; ++i) {
      for (int64_t j = 0; j < nblocks; ++j) {
        is_admissible.insert(i, j, height, std::abs(i - j) > admis);
      }
    }
    level_blocks.push_back(1);
    level_blocks.push_back(nblocks);
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

  generate_leaf_nodes(domain);
}

double BLR2_SPD::construction_relative_error(const Domain& domain) {
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
          Matrix Vbig = get_Ubig(col, level);

          Matrix expected_matrix = matmul(matmul(Ubig, S(row, col, level)), Vbig, false, true);
          Matrix actual_matrix =
              Hatrix::generate_p2p_interactions(domain, row, col, level, height);

          dense_norm += pow(norm(actual_matrix), 2);
          error += pow(norm(expected_matrix - actual_matrix), 2);
        }
      }
    }
  }

  return std::sqrt(error / dense_norm);
}

void BLR2_SPD::actually_print_structure(int64_t level) {
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

void BLR2_SPD::print_structure() {
  actually_print_structure(height);
}

double BLR2_SPD::low_rank_block_ratio() {
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

void BLR2_SPD::update_row_basis(int64_t row, int64_t level, RowColMap<Matrix>& F, RowMap& r) {
  int64_t nblocks = level_blocks[level];
  int64_t block_size = D(row, row, level).rows;
  Matrix row_block(block_size, 0);

  row_block = concat(row_block, matmul(U(row, level), Scol(row, level)), 1);
  for (int64_t j = 0; j < nblocks; ++j) {
    if (F.exists(row, j)) {
      if (F(row, j).rows == block_size && F(row, j).cols == rank)  {
        row_block = concat(row_block, matmul(F(row, j), U(j, level), false, true), 1);
      }
      else if (F(row, j).rows == block_size && F(row, j).cols == block_size) {
        row_block = concat(row_block, matmul(F(row, j), U(j, level)), 1);
      }
    }
  }

  Matrix UN_row, SN_row, _VNT_row; double error;
  std::tie(UN_row, SN_row, _VNT_row, error) = truncated_svd(row_block, rank);

  Matrix r_row = matmul(UN_row, U(row, level), true, false);

  U.erase(row, level);
  U.insert(row, level, std::move(UN_row));

  Scol.erase(row, level);
  Scol.insert(row, level, std::move(SN_row));

  if (r.exists(row)) { r.erase(row); }
  r.insert(row, std::move(r_row));
}

void BLR2_SPD::factorize_level(int64_t level, int64_t nblocks,
                               const Domain& domain,
                               RowMap& r) {
  RowColMap<Matrix> F;      // fill-in blocks.

  for (int64_t block = 0; block < nblocks; ++block) {
    if (block > 0) {
      int64_t block_size = D(block, block, level).rows;
      bool found_col_fill_in = false;

      for (int64_t i = 0; i < nblocks; ++i) {
        if (F.exists(i, block)) {
          found_col_fill_in = true;
          break;
        }
      }

      if (found_col_fill_in) {
        update_row_basis(block, level, F, r);
      }

      if (found_col_fill_in) {
        // Update S along the row
        for (int64_t j = 0; j < nblocks; ++j) {
          if (is_admissible.exists(block, j, level) && is_admissible(block, j, level)) {
            Matrix Sbar_block_j = matmul(r(block), S(block, j, level));

            if (F.exists(block, j)) {
              if (F(block, j).rows == block_size && F(block, j).cols == rank) {
                Sbar_block_j += matmul(U(block, level), F(block, j), true, false);
              }
              else if (F(block, j).rows == block_size &&
                       F(block, j).cols == block_size &&
                       j > block) {
                Sbar_block_j +=
                    matmul(matmul(U(block, level), F(block, j), true, false), U(j, level));
              }
            }

            S.erase(block, j, level);
            S.insert(block, j, level, std::move(Sbar_block_j));
          }
        }
        // Update S along the column
        for (int i = 0; i < nblocks; ++i) {
          if (is_admissible.exists(i, block, level) && is_admissible(i, block, level)) {
            Matrix Sbar_i_block = matmul(S(i, block,level), r(block));
            if (F.exists(i, block)) {
              if (F(i, block).rows == rank && F(i, block).cols == block_size) {
                Sbar_i_block += matmul(F(i, block), U(block, level));
              }
              else if (F(i, block).rows == block_size &&
                       F(i, block).cols == block_size &&
                       i > block) {
                Sbar_i_block += matmul(U(i, level),
                                       matmul(F(i, block), U(block, level)), true, false);
              }
            }

            S.erase(i, block, level);
            S.insert(i, block, level, std::move(Sbar_i_block));
          }
        }
      }
    }

    Matrix U_F = prepend_complement(U(block, level));

    for (int j = 0; j < nblocks; ++j) {
      if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) {
        D(block, j, level) = matmul(U_F, D(block, j, level), true);
      }
    }

    for (int i = 0; i < nblocks; ++i) {
      if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) {
        D(i, block, level) = matmul(D(i, block, level), U_F);
      }
    }

    int64_t row_split = D(block, block, level).rows - rank,
            col_split = D(block, block, level).cols - rank;

    // The diagonal block is split along the row and column.
    auto diagonal_splits = D(block, block, level).split(vec{row_split}, vec{col_split});
    Matrix& Dcc = diagonal_splits[0];
    ldl(Dcc);

    // TRSM with cc blocks on the row
    for (int64_t j = block + 1; j < nblocks; ++j) {
      if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) {
        int64_t col_split = D(block, j, level).cols - rank;
        auto D_splits = D(block, j, level).split(vec{row_split}, vec{col_split});
        solve_triangular(Dcc, D_splits[0], Hatrix::Left, Hatrix::Lower, true, false);
        solve_diagonal(Dcc, D_splits[0], Hatrix::Left);
      }
    }

    // TRSM with co blocks on this row
    for (int64_t j = 0; j < nblocks; ++j) {
      if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) {
        int64_t col_split = D(block, j, level).rows - rank;
        auto D_splits = D(block, j, level).split(vec{row_split}, vec{col_split});
        solve_triangular(Dcc, D_splits[1], Hatrix::Left, Hatrix::Lower, true, false);
        solve_diagonal(Dcc, D_splits[1], Hatrix::Left);
      }
    }

    // TRSM with cc blocks on the column
    for (int64_t i = block + 1; i < nblocks; ++i) {
      if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) {
        int64_t row_split = D(i, block, level).rows - rank;
        auto D_splits = D(i, block, level).split(vec{row_split}, vec{col_split});
        solve_triangular(Dcc, D_splits[0], Hatrix::Right, Hatrix::Lower, true, true);
        solve_diagonal(Dcc, D_splits[0], Hatrix::Right);
      }
    }

    // TRSM with oc blocks on the column
    for (int64_t i = 0; i < nblocks; ++i) {
      if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) {
        int64_t row_split = D(i, block, level).rows - rank;
        auto D_splits = D(i, block, level).split(vec{row_split}, vec{col_split});
        solve_triangular(Dcc, D_splits[2], Hatrix::Right, Hatrix::Lower, true, true);
        solve_diagonal(Dcc, D_splits[2], Hatrix::Right);
      }
    }

    // Fill in between cc blocks.
    for (int64_t i = block+1; i < nblocks; ++i) {
      for (int64_t j = block+1; j < nblocks; ++j) {
        if ((is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) &&
            (is_admissible.exists(i, block, level) && !is_admissible(i, block, level))) {
          auto lower_splits = D(i, block, level).split(vec{D(i, block, level).rows - rank},
                                                       vec{D(i, block, level).cols - rank});
          auto right_splits = D(block, j, level).split(vec{D(block, j, level).rows - rank},
                                                       vec{D(block, j, level).cols - rank});

          Matrix lower_scaled(lower_splits[0]);
          column_scale(lower_scaled, Dcc);
          if (is_admissible.exists(i, j, level) && !is_admissible(i, j, level)) {
            auto reduce_splits = D(i, j, level).split(vec{D(i, j, level).rows - rank},
                                                      vec{D(i, j, level).cols - rank});
            matmul(lower_scaled, right_splits[0], reduce_splits[0], false, false, -1.0, 1.0);
          }
          else {
            int64_t rows = D(i, block, level).rows;
            int64_t cols = D(block, j, level).cols;
            if (F.exists(i, j)) {
              Matrix& fill_in = F(i, j);
              auto fill_in_splits = fill_in.split(vec{rows - rank}, vec{cols - rank});
              matmul(lower_scaled, right_splits[0], fill_in_splits[0],
                     false, false, -1.0, 1.0);
            }
            else {
              Matrix fill_in(rows, cols);
              auto fill_in_splits = fill_in.split(vec{rows - rank}, vec{cols - rank});
              matmul(lower_scaled, right_splits[0], fill_in_splits[0],
                     false, false, -1.0, 1.0);

              F.insert(i, j, std::move(fill_in));
            }
          }
        }
      }
    }

    // Schur's compliment between oc and co blocks.
    for (int64_t i = 0; i < nblocks; ++i) {
      for (int64_t j = 0; j < nblocks; ++j) {
        if ((is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) &&
            (is_admissible.exists(i, block, level) && !is_admissible(i, block, level))) {
          auto lower_splits = D(i, block, level).split(vec{D(i, block, level).rows - rank},
                                                       vec{col_split});
          auto right_splits = D(block, j, level).split(vec{row_split},
                                                       vec{D(block, j, level).cols - rank});

          if (is_admissible.exists(i, j, level) && !is_admissible(i, j, level)) {
            Matrix lower_scaled(lower_splits[2]);
            column_scale(lower_scaled, Dcc);
            // no fill-in in the oo portion. SC into another dense block.
            auto reduce_splits = D(i, j, level).split(vec{D(i, j, level).rows - rank},
                                                      vec{D(i, j, level).cols - rank});
            matmul(lower_scaled, right_splits[1], reduce_splits[3], false, false, -1.0, 1.0);
          }
        }
      }
    }

    // Schur's compliment between cc and co blocks where the result exists
    // before the diagonal block.
    for (int64_t i = block+1; i < nblocks; ++i) {
      for (int64_t j = 0; j < nblocks; ++j) {
        if ((is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) &&
            (is_admissible.exists(i, block, level) && !is_admissible(i, block, level))) {
          auto lower_splits = D(i, block, level).split(vec{D(i, block, level).rows - rank},
                                                       vec{col_split});
          auto right_splits = D(block, j, level).split(vec{row_split},
                                                       vec{D(block, j, level).cols - rank});

          Matrix lower0_scaled(lower_splits[0]);
          Matrix lower2_scaled(lower_splits[2]);
          column_scale(lower0_scaled, Dcc);
          column_scale(lower2_scaled, Dcc);
          // Schur's compliment between co and cc blocks where product exists as dense.
          if (is_admissible.exists(i, j, level) && !is_admissible(i, j, level)) {
            auto reduce_splits = D(i, j, level).split(vec{D(i, j, level).rows - rank},
                                                      vec{D(i, j, level).cols - rank});
            matmul(lower0_scaled, right_splits[1], reduce_splits[1], false, false, -1.0, 1.0);
          }
          // Schur's compliement between co and cc blocks where a new fill-in is created.
          // The product is a (co; oo)-sized matrix.
          else {
            // The Schur's compliments that are formed before the block index are always
            // a narrow strip of size nb * rank. These blocks are formed only on the right
            // part of the permuted matrix in the co section.
            if (!F.exists(i, j)) {
              Matrix fill_in(D(i, block, level).rows, rank);
              auto fill_splits = fill_in.split(vec{D(i, block, level).rows - rank}, {});
              // Update the co block within the fill-in.
              matmul(lower0_scaled, right_splits[1], fill_splits[0],
                     false, false, -1.0, 1.0);

              // Update the oo block within the fill-in.
              matmul(lower2_scaled, right_splits[1], fill_splits[1],
                     false, false, -1.0, 1.0);
              // fill_in_row_indices.insert(i);
              F.insert(i, j, std::move(fill_in));
            }
            else {
              // Schur's compliment between co and cc blocks where the result exists
              // after the diagonal blocks. The fill-in generated here is always part
              // of a nb*nb dense block. Thus we grab the large fill-in block that was
              // already formed previously in the cc * cc schur's compliment computation,
              // and add the resulting schur's compliment into that previously generated block.
              if (F(i, j).rows == D(i, block, level).rows &&
                  F(i, j).cols == D(block, j, level).cols) {
                Matrix& fill_in = F(i, j);
                auto fill_splits = fill_in.split(vec{D(i, block, level).rows - rank},
                                                 vec{D(block, j, level).cols - rank});
                // Update the co block within the fill-in.
                matmul(lower0_scaled, right_splits[1], fill_splits[1],
                       false, false, -1.0, 1.0);
                // Update the oo block within the fill-in.
                matmul(lower2_scaled, right_splits[1], fill_splits[3],
                       false, false, -1.0, 1.0);
              }
              else {
                Matrix &fill_in = F(i, j);
                auto fill_splits = fill_in.split(vec{D(i, block, level).rows - rank}, {});
                // Update the co block within the fill-in.
                matmul(lower0_scaled, right_splits[1], fill_splits[0],
                       false, false, -1.0, 1.0);
                // Update the oo block within the fill-in.
                matmul(lower2_scaled, right_splits[1], fill_splits[1],
                       false, false, -1.0, 1.0);
              }
            }
          }
        }
      }
    }

    // Schur's compliment between oc and cc blocks where the result exists
    // before the diagonal blocks.
    for (int64_t i = 0; i < nblocks; ++i) {
      for (int64_t j = block+1; j < nblocks; ++j) {
        if ((is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) &&
            (is_admissible.exists(i, block, level) && !is_admissible(i, block, level))) {
          auto lower_splits = D(i, block, level).split(vec{D(i, block, level).rows - rank},
                                                       vec{col_split});
          auto right_splits = D(block, j, level).split(vec{row_split},
                                                       vec{D(block, j, level).cols - rank});

          Matrix lower_scaled(lower_splits[2]);
          column_scale(lower_scaled, Dcc);
          // Schur's compliement between oc and cc blocks where product exists as dense.
          if (is_admissible.exists(i, j, level) && !is_admissible(i, j, level)) {
            auto reduce_splits = D(i, j, level).split(vec{D(i, j, level).rows - rank},
                                                      vec{D(i, j, level).cols - rank});
            matmul(lower_scaled, right_splits[0], reduce_splits[2],
                   false, false, -1.0, 1.0);
          }
          // Schur's compliement between oc and cc blocks where a new fill-in is created.
          // The product is a (oc, oo)-sized block.
          else {
            if (!F.exists(i, j)) {
              Matrix fill_in(rank, D(block, j, level).cols);
              auto fill_splits = fill_in.split(vec{}, vec{D(block, j, level).cols - rank});
              // Update the oc block within the fill-ins.
              matmul(lower_scaled, right_splits[0], fill_splits[0],
                     false, false, -1.0, 1.0);
              // Update the oo block within the fill-ins.
              matmul(lower_scaled, right_splits[1], fill_splits[1],
                     false, false, -1.0, 1.0);
              // fill_in_col_indices.insert(j);
              F.insert(i, j, std::move(fill_in));
            }
            else {
              // Schur's compliment between oc and cc blocks where the result exists
              // after the diagonal blocks. The fill-in generated here is always part
              // of a nb*nb dense block.
              if (F(i, j).rows == D(i, block, level).rows &&
                  F(i, j).cols == D(block, j, level).cols) {
                Matrix& fill_in = F(i, j);
                auto fill_splits = fill_in.split(vec{D(i, block, level).rows - rank},
                                                 vec{D(block, j, level).cols - rank});

                // Update the oc block within the fill-ins.
                matmul(lower_scaled, right_splits[0], fill_splits[2],
                       false, false, -1.0, 1.0);
                // Update the oo block within the fill-ins.
                matmul(lower_scaled, right_splits[1], fill_splits[3],
                       false, false, -1.0, 1.0);
              }
              else {
                Matrix& fill_in = F(i, j);
                auto fill_splits = fill_in.split({}, vec{D(block, j, level).cols - rank});
                // Update the oc block within the fill-ins.
                matmul(lower_scaled, right_splits[0], fill_splits[0],
                       false, false, -1.0, 1.0);
                // Update the oo block within the fill-ins.
                matmul(lower_scaled, right_splits[1], fill_splits[1],
                       false, false, -1.0, 1.0);
              }
            }
          }
        }
      }
    }
  } // for (int block = 0; block < nblocks; ++block)

  F.erase_all();
}

void BLR2_SPD::factorize(const Domain& domain) {
  int64_t level = height;
  RowColLevelMap<Matrix> F;
  RowMap r;

  for (; level > 0; --level) {
    int64_t nblocks = level_blocks[level];
    bool is_all_dense_level = false;
    for (int64_t i = 0; i < nblocks; ++i) {
      if (!U.exists(i, level)) {
        is_all_dense_level = true;
      }
    }

    if (is_all_dense_level) {
      if (level != 1) {
        std::cout << "found an all dense block on level " << level << ". Aborting.\n";
        abort();
      }
      break;
    }

    factorize_level(level, nblocks, domain, r);

    int64_t parent_level = level - 1;

    // Merge the unfactorized parts.
    int64_t parent_nblocks = level_blocks[parent_level];

    for (int64_t i = 0; i < parent_nblocks; ++i) {
      for (int64_t j = 0; j < parent_nblocks; ++j) {
        if (is_admissible.exists(i, j, parent_level) && !is_admissible(i, j, parent_level)) {
          // TODO: need to switch to morton indexing so finding the parent is straightforward.
          std::vector<int64_t> i_children, j_children;
          int64_t nrows=0, ncols=0;
          for (int n = 0; n < level_blocks[level]; ++n) {
            i_children.push_back(n);
            j_children.push_back(n);

            nrows += rank;
            ncols += rank;
          }
          Matrix D_unelim(nrows, ncols);
          auto D_unelim_splits = D_unelim.split(i_children.size(), j_children.size());

          for (int64_t ic1 = 0; ic1 < i_children.size(); ++ic1) {
            for (int64_t jc2 = 0; jc2 < j_children.size(); ++jc2) {
              int64_t c1 = i_children[ic1], c2 = j_children[jc2];
              if (!U.exists(c1, level)) { continue; }

              if (is_admissible.exists(c1, c2, level) && !is_admissible(c1, c2, level)) {

                auto D_splits = D(c1, c2, level).split(vec{D(c1, c2, level).rows - rank},
                                                       vec{D(c1, c2, level).cols - rank});
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

  // Factorize remaining part
  int64_t last_nodes = level_blocks[level];
  for (int64_t d = 0; d < last_nodes; ++d) {
    ldl(D(d, d, level));
    for (int64_t j = d+1; j < last_nodes; ++j) {
      solve_triangular(D(d, d, level), D(d, j, level), Hatrix::Left, Hatrix::Lower, true, false);
      solve_diagonal(D(d, d, level), D(d, j, level), Hatrix::Left);
    }
    for (int64_t i = d+1; i < last_nodes; ++i) {
      solve_triangular(D(d, d, level), D(i, d, level), Hatrix::Right, Hatrix::Lower, true, true);
      solve_diagonal(D(d, d, level), D(i, d, level), Hatrix::Right);
    }

    for (int64_t i = d+1; i < last_nodes; ++i) {
      Matrix lower_scaled(D(i, d, level));
      column_scale(lower_scaled, D(d, d, level));
      for (int64_t j = d+1; j < last_nodes; ++j) {
        matmul(lower_scaled, D(d, j, level), D(i, j, level), false, false, -1.0, 1.0);
      }
    }
  }
}

} // namespace Hatrix

int main(int argc, char ** argv) {
  const int64_t N = argc > 1 ? atoi(argv[1]) : 256;
  const int64_t nleaf = argc > 2 ? atoi(argv[2]) : 64;
  const int64_t rank = argc > 3 ? atoi(argv[3]) : 8;
  const double admis = argc > 4 ? atof(argv[4]) : 1.0;
  // diagonal_admis or geometry_admis
  const std::string admis_kind = argc > 5 ? std::string(argv[5]) : "diagonal_admis";
  // 0: BLR2
  constexpr int64_t matrix_type = 0;

  Hatrix::Context::init();

  const auto start_particles = std::chrono::system_clock::now();
  constexpr int64_t ndim = 2;
  Hatrix::Domain domain(N, ndim);
  // Laplace kernel
  domain.generate_particles(0, 1);
  Hatrix::kernel_function = Hatrix::laplace_kernel;
  domain.divide_domain_and_create_particle_boxes(nleaf);
  const auto stop_particles = std::chrono::system_clock::now();
  const double particle_construct_time = std::chrono::duration_cast<std::chrono::milliseconds>
                                         (stop_particles - start_particles).count();

  const auto start_construct = std::chrono::system_clock::now();
  Hatrix::BLR2_SPD A(domain, N, rank, nleaf, admis, admis_kind);
  const auto stop_construct = std::chrono::system_clock::now();
  const double construct_time = std::chrono::duration_cast<std::chrono::milliseconds>
                                (stop_construct - start_construct).count();
  double construct_error, lr_ratio, solve_error;
  construct_error = A.construction_relative_error(domain);
  lr_ratio = A.low_rank_block_ratio();

  A.print_structure();
  const auto start_factor = std::chrono::system_clock::now();
  A.factorize(domain);
  const auto stop_factor = std::chrono::system_clock::now();
  const double factor_time = std::chrono::duration_cast<std::chrono::milliseconds>
                             (stop_factor - start_factor).count();
  /*
  Hatrix::Matrix b = Hatrix::generate_random_matrix(N, 1);
  const auto solve_start = std::chrono::system_clock::now();
  Hatrix::Matrix x = A.solve(b, A.height);
  const auto solve_stop = std::chrono::system_clock::now();
  const double solve_time = std::chrono::duration_cast<std::chrono::milliseconds>
                            (solve_stop - solve_start).count();
  Hatrix::Matrix x_solve = lu_solve(Adense, b);
  solve_error = Hatrix::norm(x - x_solve) / Hatrix::norm(x_solve);

  */
  Hatrix::Context::finalize();

  std::cout << "N=" << N
            << " admis=" << admis << std::setw(3)
            << " nleaf=" << nleaf
            << " ndim=" << ndim
            << " height=" << A.height << " rank=" << rank
            << " construct_error=" << std::setprecision(5) << construct_error
            // << " solve_error=" << std::setprecision(5) << solve_error
            << " LR%= " << std::setprecision(5) << lr_ratio * 100 << "%"
            << " admis_kind=" << admis_kind
            << " matrix_type=" << (matrix_type == BLR2_MATRIX ? "BLR2" : "H2")
            << " factor_time=" << factor_time
            // << " solve_time=" << solve_time
            << " construct_time=" << construct_time
            << " particle_time=" << particle_construct_time
            << std::endl;

  return 0;
}
