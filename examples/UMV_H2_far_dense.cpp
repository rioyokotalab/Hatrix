#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>
#include <random>
#include <string>
#include <iomanip>
#include <functional>
#include <fstream>
#include <set>

#include "Hatrix/Hatrix.h"

// Construction of BLR2 strong admis matrix based on geometry based admis condition.
double PV = 1e-3;
enum MATRIX_TYPES {BLR2_MATRIX=0, H2_MATRIX=1};

#define SPLIT_DENSE(dense, row_split, col_split)        \
  dense.split(std::vector<int64_t>(1, row_split),       \
              std::vector<int64_t>(1, col_split));


// statistics kernels
double beta, nu, noise, sigma;
// electrodynamics kernels
double wave_k, add_diag;


using namespace Hatrix;

Matrix A2_expected, A1_expected;

Hatrix::Matrix make_complement(const Hatrix::Matrix &Q) {
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

  class H2 {
  public:
    int64_t N, rank, nleaf;
    double admis;
    ColLevelMap U;
    RowLevelMap V;
    RowColLevelMap<Matrix> D, S;
    RowColLevelMap<bool> is_admissible;
    RowLevelMap Srow, Scol;
    int64_t oversampling = 5;
    std::string admis_kind;
    int64_t height = -1;
    std::vector<int64_t> level_blocks;
    int64_t matrix_type;

  private:
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
    Matrix generate_row_block(int64_t block, int64_t block_size,
                              const Domain& domain, int64_t level);
    std::tuple<Matrix, Matrix>
    generate_row_bases(int64_t block, int64_t block_size, const Domain& domain,
                       std::vector<Matrix>& Y, int64_t level);
    std::tuple<RowLevelMap, ColLevelMap>
    generate_transfer_matrices(const Domain& domain,
                               int64_t level, RowLevelMap& Uchild,
                               ColLevelMap& Vchild);

    Matrix get_Ubig(int64_t node, int64_t level);
    Matrix get_Vbig(int64_t node, int64_t level);
    void compute_matrix_structure(int64_t level);
    std::tuple<Matrix, Matrix>
    generate_U_transfer_matrix(Matrix& Ubig_child1, Matrix& Ubig_child2, int64_t node,
                               int64_t block_size, const Domain& domain, int64_t level);
    std::tuple<Matrix, Matrix>
    generate_V_transfer_matrix(Matrix& Vbig_child1, Matrix& Vbig_child2, int64_t node,
                               int64_t block_size, const Domain& domain, int64_t level);

    void
    calc_geometry_based_admissibility(int64_t level, const Domain& domain);
    void calc_diagonal_based_admissibility(int64_t level);
    void coarsen_blocks(int64_t level);
    int64_t geometry_admis_non_leaf(int64_t nblocks, int64_t level);
    int64_t calc_geometry_based_admissibility(const Domain& domain);
    int64_t get_block_size_row(const Domain& domain, int64_t i, int64_t level);
    int64_t get_block_size_col(const Domain& domain, int64_t parent, int64_t level);
    void factorize_level(int64_t level, int64_t nblocks, const Domain& domain,
                         RowMap& r, RowMap& t);
    int64_t find_all_dense_row();
    void update_row_basis(int64_t row, int64_t level, RowColMap<Matrix>& F, RowMap& r);
    void update_col_basis(int64_t col, int64_t level, RowColMap<Matrix>& F, RowMap& t);
  public:
    H2(const Domain& domain, int64_t _N, int64_t _rank, int64_t _nleaf, double _admis,
       std::string& admis_kind, int64_t matrix_type);
    double construction_relative_error(const Domain& domain);
    void print_structure();
    double low_rank_block_ratio();
    void factorize(const Domain& domain);
    Matrix solve(const Matrix& b, int64_t _level);
  };
}

namespace Hatrix {
  double
  block_sin(const std::vector<double>& coords_row,
            const std::vector<double>& coords_col) {
    double dist = 0, temp;
    int64_t ndim = coords_row.size();

    for (int64_t k = 0; k < ndim; ++k) {
      dist += pow(coords_row[k] - coords_col[k], 2);
    }
    if (dist == 0) {
      return add_diag;
    }
    else {
      dist = std::sqrt(dist);
      return sin(wave_k * dist) / dist;
    }
  }

  double
  sqrexp_kernel(const std::vector<double>& coords_row,
                const std::vector<double>& coords_col) {
    int64_t ndim = coords_row.size();
    double dist = 0;
    double local_beta = -2 * pow(beta, 2);
    // Copied from kernel_sqrexp.c in stars-H.
    for (int64_t k = 0; k < ndim; ++k) {
      dist += pow(coords_row[k] - coords_col[k], 2);
    }
    dist = dist / local_beta;
    if (std::abs(dist) < 1e-10) {
      return sigma + noise;
    }
    else {
      return sigma * exp(dist);
    }
  }

  double laplace_kernel(const std::vector<double>& coords_row,
                        const std::vector<double>& coords_col) {
    int64_t ndim = coords_row.size();
    double rij = 0;
    for (int64_t k = 0; k < ndim; ++k) {
      rij += pow(coords_row[k] - coords_col[k], 2);
    }
    double out = 1 / (std::sqrt(rij) + PV);

    return out;
  }

  // Generates p2p interactions between the particles of two boxes specified by irow
  // and icol. ndim specifies the dimensionality of the particles present in domain.
  // Uses a laplace kernel for generating the interaction.
  Matrix generate_p2p_interactions(const Domain& domain,
                                   int64_t irow, int64_t icol) {
    Matrix out(domain.boxes[irow].num_particles, domain.boxes[icol].num_particles);

    for (int64_t i = 0; i < domain.boxes[irow].num_particles; ++i) {
      for (int64_t j = 0; j < domain.boxes[icol].num_particles; ++j) {
        int64_t source = domain.boxes[irow].start_index;
        int64_t target = domain.boxes[icol].start_index;

        out(i, j) = kernel_function(domain.particles[source+i].coords,
                                    domain.particles[target+j].coords);
      }
    }

    return out;
  }

  std::vector<int64_t>
  leaf_indices(int64_t node, int64_t level, int64_t height) {
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
  void
  Domain::orthogonal_recursive_bisection_1dim(int64_t start,
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


  // void H2::factorize_level(int64_t level, int64_t nblocks, const Domain& domain,
  //                          RowMap& r, RowMap& t) {
  //   RowColMap<Matrix> F;      // fill-in blocks.
  //   std::vector<int64_t> r_indices, t_indices;

  //   // row indices of row fill-ins, except (oc,oo) type (rank x block_size) sized fill-ins.
  //   std::set<int64_t> fill_in_rows;
  //   // col indices of col fill-ins, except (co;oo) type (block_size x rank) sized fill-ins.
  //   std::set<int64_t> fill_in_cols;

  //   for (int64_t block = 0; block < nblocks; ++block) {
  //     int64_t block_size = D(block, block, level).rows;
  //           if (block > 0) {
  //       {
  //         // Scan for fill-ins in the same row as this diagonal block.
  //         Matrix row_concat(block_size, 0);
  //         bool found_row_fill_in = false;
  //         for (int j = 0; j < nblocks; ++j) {
  //           if (F.exists(block, j)) {
  //             found_row_fill_in = true;
  //             break;
  //           }
  //         }

  //         if (found_row_fill_in) {
  //           row_concat = concat(row_concat, matmul(U(block, level),
  //                                                  Scol(block, level)), 1);
  //           for (int j = 0; j < nblocks; ++j) {
  //             if (F.exists(block, j)) {
  //               Matrix Fp = matmul(F(block, j), V(j, level), false, true);
  //               row_concat = concat(row_concat, Fp, 1);
  //             }
  //           }

  //           Matrix UN1, _SN1, _VN1T; double error;
  //           std::tie(UN1, _SN1, _VN1T, error) = truncated_svd(row_concat, rank);
  //           Scol.erase(block, level);
  //           Scol.insert(block, level, std::move(_SN1));

  //           Matrix r_block = matmul(UN1, U(block, level), true, false);

  //           for (int j = 0; j < nblocks; ++j) {
  //             if (is_admissible.exists(block, j, level) && is_admissible(block, j, level)) {
  //               Matrix Sbar_block_j = matmul(r_block, S(block, j, level));

  //               if (F.exists(block, j)) {
  //                 Matrix SpF = matmul(UN1, F(block, j), true, false);
  //                 Sbar_block_j = Sbar_block_j + SpF;
  //               }

  //               S.erase(block, j, level);
  //               S.insert(block, j, level, std::move(Sbar_block_j));
  //             }
  //           }
  //           U.erase(block, level);
  //           U.insert(block, level, std::move(UN1));
  //           r.insert(block, std::move(r_block));
  //         } // if (found_row_fill_in)
  //       }

  //       {
  //         // Scan for fill-ins in the same col as this diagonal block.
  //         Matrix col_concat(0, block_size);
  //         std::vector<int64_t> UN2_row_splits;
  //         bool found_col_fill_in = false;
  //         for (int i = 0; i < nblocks; ++i) {
  //           if (F.exists(i, block)) {
  //             found_col_fill_in = true;
  //             break;
  //           }
  //         }

  //         if (found_col_fill_in) {
  //           col_concat = concat(col_concat, matmul(Srow(block, level),
  //                                                  transpose(V(block, level))), 0);
  //           for (int i = 0; i < nblocks; ++i) {
  //             if (F.exists(i, block)) {
  //               Matrix Fp = matmul(U(i, level), F(i, block));
  //               col_concat = concat(col_concat, Fp, 0);
  //             }
  //           }

  //           Matrix _UN2, _SN2, VN2T; double error;
  //           std::tie(_UN2, _SN2, VN2T, error) = truncated_svd(col_concat, rank);
  //           Srow.erase(block, level);
  //           Srow.insert(block, level, std::move(_SN2));

  //           Matrix t_block = matmul(V(block, level), VN2T, true, true);

  //           for (int i = 0; i < nblocks; ++i) {
  //             if (is_admissible.exists(i, block, level) && is_admissible(i, block, level)) {
  //               Matrix Sbar_i_block = matmul(S(i, block,level), t_block);
  //               if (F.exists(i, block)) {
  //                 Matrix SpF = matmul(F(i, block), VN2T, false, true);
  //                 Sbar_i_block = Sbar_i_block + SpF;
  //               }

  //               S.erase(i, block, level);
  //               S.insert(i, block, level, std::move(Sbar_i_block));
  //             }
  //           }

  //           V.erase(block, level);
  //           V.insert(block, level, transpose(VN2T));
  //           t.insert(block, std::move(t_block));
  //         }
  //       }
  //     }
  //     if (false) {
  //       int64_t nblocks = level_blocks[level];

  //       // Calculate row fill-ins
  //       for (auto row_fill_itr = fill_in_rows.begin();
  //            row_fill_itr != fill_in_rows.end(); ++row_fill_itr) {
  //         int64_t i = *row_fill_itr;
  //         int64_t block_size = D(i, i, level).rows;

  //         Matrix row_i(block_size, 0);

  //         row_i = concat(row_i, matmul(U(i, level), Scol(i, level)), 1);
  //         for (int64_t j = 0; j < nblocks; ++j) {
  //           if (F.exists(i, j)) {
  //             Matrix Ui, Si, Vi; double error;
  //             Matrix cpy(F(i, j));
  //             std::tie(Ui, Si, Vi) = error_svd(cpy, 1e-9);

  //             if (F(i, j).rows == block_size && F(i, j).cols == rank)  {
  //               row_i = concat(row_i, matmul(F(i, j), V(j, level), false, true), 1);
  //             }
  //             else if (F(i, j).rows == block_size && F(i, j).cols == block_size) {
  //               row_i = concat(row_i, F(i, j), 1);
  //             }
  //           }
  //         }

  //         Matrix UN_i, SN_i, _VNT_i; double error;
  //         std::tie(UN_i, SN_i, _VNT_i, error) = truncated_svd(row_i, rank);

  //         Matrix r_i = matmul(UN_i, U(i, level), true, false);

  //         U.erase(i, level);
  //         U.insert(i, level, std::move(UN_i));

  //         Scol.erase(i, level);
  //         Scol.insert(i, level, std::move(SN_i));


  //         r_indices.push_back(i);
  //         if (r.exists(i)) { r.erase(i); }
  //         r.insert(i, std::move(r_i));
  //       }

  //       // Calculate col fill-ins
  //       for (auto col_fill_itr = fill_in_cols.begin();
  //            col_fill_itr != fill_in_cols.end(); ++col_fill_itr) {
  //         int64_t j = *col_fill_itr;
  //         int64_t block_size = D(j, j, level).rows;
  //         Matrix col_j(0, block_size);
  //         col_j = concat(col_j,
  //                        matmul(Srow(j, level), V(j, level), false, true), 0);
  //         for (int64_t i = 0; i < nblocks; ++i) {
  //           if (F.exists(i, j)) {
  //             Matrix Ui, Si, Vi; double error;
  //             Matrix cpy(F(i, j));
  //             std::tie(Ui, Si, Vi) = error_svd(cpy, 1e-9);

  //             if (F(i, j).rows == rank && F(i, j).cols == block_size) {
  //               col_j = concat(col_j, matmul(U(i, level), F(i, j)), 0);
  //             }
  //             else if (F(i, j).rows == block_size && F(i, j).cols == block_size) {
  //               col_j = concat(col_j, F(i, j), 0);
  //             }
  //           }
  //         }

  //         Matrix _UN_j, SN_j, VNT_j; double error;
  //         std::tie(_UN_j, SN_j, VNT_j, error) = truncated_svd(col_j, rank);

  //         Matrix t_j = matmul(V(j, level), VNT_j, true, true);
  //         V.erase(j, level);
  //         V.insert(j, level, transpose(VNT_j));

  //         Srow.erase(j, level);
  //         Srow.insert(j, level, std::move(SN_j));

  //         t_indices.push_back(j);
  //         if (t.exists(j)) { t.erase(j); }
  //         t.insert(j, std::move(t_j));
  //       }

  //       // Update S blocks in the rows.
  //       for (int64_t i = 0; i < r_indices.size(); ++i) {
  //         int64_t r_index = r_indices[i];
  //         Matrix& r_i = r(r_index);

  //         for (int64_t j = 0; j < nblocks; ++j) {
  //           if (is_admissible.exists(r_index, j, level) && is_admissible(r_index, j, level)) {
  //             int64_t block_size = D(j, j, level).cols;
  //             Matrix Sbar_ij(rank, rank);
  //             if (F.exists(r_index, j)) {
  //               if (F(r_index, j).rows == block_size && F(r_index, j).cols == rank) {
  //                 Sbar_ij = matmul(r_i, S(r_index, j, level)) + matmul(U(r_index, level), F(r_index, j), true, false);
  //               }
  //               else if (F(r_index, j).rows == block_size && F(r_index, j).cols == block_size) {
  //                 Sbar_ij = matmul(matmul(r_i, S(r_index, j, level)), t(j)) +
  //                   matmul(matmul(U(r_index, level), F(r_index, j)), V(j, level));
  //               }
  //             }
  //             else {
  //               Sbar_ij = matmul(r_i, S(r_index, j, level));
  //             }

  //             S.erase(r_index, j, level);
  //             S.insert(r_index, j, level, std::move(Sbar_ij));
  //           }

  //           if (F.exists(r_index, j)) {
  //             // F.erase(r_index, j);
  //           }
  //         }
  //       }

  //       // Update S blocks in the cols
  //       for (int64_t j = 0; j < t_indices.size(); ++j) {
  //         int64_t t_index = t_indices[j];
  //         Matrix& t_j = t(t_index);

  //         for (int64_t i = 0; i < nblocks; ++i) {
  //           int64_t block_size = D(i, i, level).cols;
  //           if (is_admissible.exists(i, t_index, level) && is_admissible(i, t_index, level)) {
  //             // std::cout << "COl update: i-> " << i <<  " t_index-> " << t_index << std::endl;
  //             Matrix Sbar_ij(rank, rank);
  //             if (F.exists(i, t_index)) {
  //               if (F(i, t_index).rows == block_size && F(i, t_index).cols == block_size) {
  //                 continue;
  //               }

  //               if (F(i, t_index).rows == rank && F(i, t_index).cols == block_size) {
  //                 Sbar_ij = matmul(S(i, t_index, level), t_j) +
  //                   matmul(F(i, t_index), V(t_index, level));
  //               }
  //               //F.erase(i, j);
  //             }
  //             else {
  //               Sbar_ij = matmul(S(i, t_index, level), t_j);
  //             }

  //             S.erase(i, t_index, level);
  //             S.insert(i, t_index, level, std::move(Sbar_ij));
  //           }
  //           if (F.exists(i, t_index)) {
  //             // F.erase(i, t_index);
  //           }
  //         }
  //       }
  //     }

  //     r_indices.clear();
  //     t_indices.clear();
  //     fill_in_rows.clear();
  //     fill_in_cols.clear();
  //     F.erase_all();

  //     if (level == 2) {
  //       auto A2_expected_splits = A2_expected.split(4, 4);

  //       for (int64_t i = 0; i < 4; ++i) {
  //         for (int64_t j = 0; j < 4; ++j) {
  //           if (is_admissible(i, j, 2)) {
  //             A2_expected_splits[i * 4 + j] =
  //               matmul(matmul(U(i, 2), S(i, j, 2)), V(j, 2), false, true);
  //           }
  //         }
  //       }
  //     }

  //     if (level == 2) {
  //       std::cout << "Norm<0,1>: " << norm(D(0,1,level)) << std::endl;
  //     }

  //     Matrix U_F = make_complement(U(block, level));
  //     Matrix V_F = make_complement(V(block, level));

  //     for (int64_t j = 0; j < nblocks; ++j) {
  //       if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) {
  //         D(block, j, level) = matmul(U_F, D(block, j, level), true);
  //       }
  //     }

  //     for (int64_t i = 0; i < nblocks; ++i) {
  //       if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) {
  //         D(i, block, level) = matmul(D(i, block, level), V_F);
  //       }
  //     }

  //     int64_t row_split = D(block, block, level).rows - rank,
  //       col_split = D(block, block, level).cols - rank;

  //     // The diagonal block is split along the row and column.
  //     auto diagonal_splits = SPLIT_DENSE(D(block, block, level), row_split, col_split);
  //     Matrix& Dcc = diagonal_splits[0];
  //     lu(Dcc);

  //     // TRSM with CC blocks on the row
  //     for (int64_t j = block + 1; j < nblocks; ++j) {
  //       if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) {
  //         int64_t col_split = D(block, j, level).cols - rank;
  //         auto D_splits = SPLIT_DENSE(D(block, j, level), row_split, col_split);
  //         solve_triangular(Dcc, D_splits[0], Hatrix::Left, Hatrix::Lower, true);
  //       }
  //     }

  //     // TRSM with co blocks on this row
  //     for (int64_t j = 0; j < nblocks; ++j) {
  //       if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) {
  //         int64_t col_split = D(block, j, level).rows - rank;
  //         auto D_splits = SPLIT_DENSE(D(block, j, level), row_split, col_split);
  //         solve_triangular(Dcc, D_splits[1], Hatrix::Left, Hatrix::Lower, true);
  //       }
  //     }

  //     // TRSM with cc blocks on the column
  //     for (int64_t i = block + 1; i < nblocks; ++i) {
  //       if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) {
  //         int64_t row_split = D(i, block, level).rows - rank;
  //         auto D_splits = SPLIT_DENSE(D(i, block, level), row_split, col_split);
  //         solve_triangular(Dcc, D_splits[0], Hatrix::Right, Hatrix::Upper, false);
  //       }
  //     }

  //     // TRSM with oc blocks on the column
  //     for (int64_t i = 0; i < nblocks; ++i) {
  //       if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) {
  //         auto D_splits = SPLIT_DENSE(D(i, block, level),
  //                                     D(i, block, level).rows - rank,
  //                                     col_split);
  //         solve_triangular(Dcc, D_splits[2], Hatrix::Right, Hatrix::Upper, false);
  //       }
  //     }

  //     // Schur's compliment between cc blocks
  //     for (int64_t i = block+1; i < nblocks; ++i) {
  //       for (int64_t j = block+1; j < nblocks; ++j) {
  //         if ((is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) &&
  //             (is_admissible.exists(i, block, level) && !is_admissible(i, block, level))) {
  //           auto lower_splits = SPLIT_DENSE(D(i, block, level),
  //                                           D(i, block, level).rows - rank,
  //                                           col_split);
  //           auto right_splits = SPLIT_DENSE(D(block, j, level),
  //                                           row_split,
  //                                           D(block, j, level).cols - rank);

  //           if (is_admissible.exists(i, j, level) && !is_admissible(i, j, level)) {
  //             auto reduce_splits = SPLIT_DENSE(D(i, j, level),
  //                                              D(i, j, level).rows - rank,
  //                                              D(i, j, level).cols - rank);

  //             matmul(lower_splits[0], right_splits[0], reduce_splits[0], false, false, -1.0, 1.0);
  //           }
  //           // else {
  //           //   // Fill in between cc blocks.
  //           //   int64_t rows = D(i, block, level).rows;
  //           //   int64_t cols = D(block, j, level).cols;
  //           //   if (F.exists(i, j)) {
  //           //     Matrix& fill_in = F(i, j);
  //           //     auto fill_in_splits = SPLIT_DENSE(fill_in, rows - rank, cols - rank);
  //           //     matmul(lower_splits[0], right_splits[0], fill_in_splits[0],
  //           //            false, false, -1.0, 1.0);
  //           //   }
  //           //   else {

  //           //     Matrix fill_in(rows, cols);
  //           //     auto fill_in_splits = SPLIT_DENSE(fill_in, rows - rank, cols - rank);
  //           //     matmul(lower_splits[0], right_splits[0], fill_in_splits[0],
  //           //            false, false, -1.0, 1.0);
  //           //     // fill_in.print();
  //           //     fill_in_rows.insert(i);
  //           //     fill_in_cols.insert(j);
  //           //     F.insert(i, j, std::move(fill_in));
  //           //   }
  //           // }
  //         }
  //       }
  //     }

  //     // Schur's compliment between oc and co blocks.
  //     for (int64_t i = 0; i < nblocks; ++i) {
  //       for (int64_t j = 0; j < nblocks; ++j) {
  //         if ((is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) &&
  //             (is_admissible.exists(i, block, level) && !is_admissible(i, block, level))) {
  //           auto lower_splits = SPLIT_DENSE(D(i, block, level),
  //                                           D(i, block, level).rows - rank,
  //                                           col_split);
  //           auto right_splits = SPLIT_DENSE(D(block, j, level),
  //                                           row_split,
  //                                           D(block, j, level).cols - rank);

  //           if (is_admissible.exists(i, j, level) && !is_admissible(i, j, level)) {
  //             // no fill-in in the oo portion. SC into another dense block.
  //             auto reduce_splits = SPLIT_DENSE(D(i, j, level),
  //                                              D(i, j, level).rows - rank,
  //                                              D(i, j, level).cols - rank);
  //             matmul(lower_splits[2], right_splits[1], reduce_splits[3], false, false, -1.0, 1.0);
  //           }
  //         }
  //       }
  //     }

  //     // Schur's compliment between cc and co blocks where the result exists
  //     // before the diagonal block.
  //     for (int64_t i = block+1; i < nblocks; ++i) {
  //       for (int64_t j = 0; j < nblocks; ++j) {
  //         if ((is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) &&
  //             (is_admissible.exists(i, block, level) && !is_admissible(i, block, level))) {
  //           auto lower_splits = SPLIT_DENSE(D(i, block, level),
  //                                           D(i, block, level).rows - rank,
  //                                           col_split);
  //           auto right_splits = SPLIT_DENSE(D(block, j, level),
  //                                           row_split,
  //                                           D(block, j, level).cols - rank);
  //           // Schur's compliment between co and cc blocks where product exists as dense.
  //           if (is_admissible.exists(i, j, level) && !is_admissible(i, j, level)) {
  //             auto reduce_splits = SPLIT_DENSE(D(i, j, level),
  //                                              D(i, j, level).rows - rank,
  //                                              D(i, j, level).cols - rank);
  //             matmul(lower_splits[0], right_splits[1], reduce_splits[1], false, false, -1.0, 1.0);
  //           }
  //           // Schur's compliement between co and cc blocks where a new fill-in is created.
  //           // The product is a (co; oo)-sized matrix.
  //           else {
  //             // The Schur's compliments that are formed before the block index are always
  //             // a narrow strip of size nb * rank. These blocks are formed only on the right
  //             // part of the permuted matrix in the co section.
  //             // if (j <= block) {
  //               if (!F.exists(i, j)) {
  //                 Matrix fill_in(D(i, block, level).rows, rank);
  //                 auto fill_splits =
  //                   fill_in.split(std::vector<int64_t>(1, D(i, block, level).rows - rank),
  //                                 {});
  //                 // Update the co block within the fill-in.
  //                 matmul(lower_splits[0], right_splits[1], fill_splits[0],
  //                        false, false, -1.0, 1.0);

  //                 // Update the oo block within the fill-in.
  //                 matmul(lower_splits[2], right_splits[1], fill_splits[1],
  //                        false, false, -1.0, 1.0);
  //                 fill_in_rows.insert(i);
  //                 F.insert(i, j, std::move(fill_in));
  //               }

  //               else {
  //                 Matrix &fill_in = F(i, j);
  //                 auto fill_splits =
  //                   fill_in.split(std::vector<int64_t>(1, D(i, block, level).rows - rank),
  //                                 {});
  //                 // Update the co block within the fill-in.
  //                 matmul(lower_splits[0], right_splits[1], fill_splits[0],
  //                        false, false, -1.0, 1.0);
  //                 // Update the oo block within the fill-in.
  //                 matmul(lower_splits[2], right_splits[1], fill_splits[1],
  //                        false, false, -1.0, 1.0);
  //               }
  //             // }
  //             // Schur's compliment between co and cc blocks where the result exists
  //             // after the diagonal blocks. The fill-in generated here is always part
  //             // of a nb*nb dense block. Thus we grab the large fill-in block that was
  //             // already formed previously in the cc * cc schur's compliment computation,
  //             // and add the resulting schur's compliment into that previously generated block.
  //             // else {
  //             //   if (F.exists(i, j)) {
  //             //     Matrix& fill_in = F(i, j);
  //             //     auto fill_splits = SPLIT_DENSE(fill_in,
  //             //                                    D(i, block, level).rows - rank,
  //             //                                    D(block, j, level).cols - rank);
  //             //     // Update the co block within the fill-in.
  //             //     matmul(lower_splits[0], right_splits[1], fill_splits[1],
  //             //            false, false, -1.0, 1.0);
  //             //     // Update the oo block within the fill-in.
  //             //     matmul(lower_splits[2], right_splits[1], fill_splits[3],
  //             //            false, false, -1.0, 1.0);
  //             //   }
  //             //   else {
  //             //     std::cout << "A fill-in block for (co;oo) does not exist where it is supposed to at block-> "
  //             //               << block << " i-> " << i << " j-> " << j << std::endl;
  //             //     abort();
  //             //   }
  //             // }
  //             // }
  //           }
  //         }
  //       }

  //       // Schur's compliment between oc and cc blocks where the result exists
  //       // before the diagonal blocks.
  //       for (int64_t i = 0; i < nblocks; ++i) {
  //         for (int64_t j = block+1; j < nblocks; ++j) {
  //           if ((is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) &&
  //               (is_admissible.exists(i, block, level) && !is_admissible(i, block, level))) {
  //             auto lower_splits = SPLIT_DENSE(D(i, block, level),
  //                                             D(i, block, level).rows - rank,
  //                                             col_split);
  //             auto right_splits = SPLIT_DENSE(D(block, j, level),
  //                                             row_split,
  //                                             D(block, j, level).cols - rank);
  //             // Schur's compliement between oc and cc blocks where product exists as dense.
  //             if (is_admissible.exists(i, j, level) && !is_admissible(i, j, level)) {
  //               auto reduce_splits = SPLIT_DENSE(D(i, j, level),
  //                                                D(i, j, level).rows - rank,
  //                                                D(i, j, level).cols - rank);
  //               matmul(lower_splits[2], right_splits[0], reduce_splits[2],
  //                      false, false, -1.0, 1.0);
  //             }
  //             // Schur's compliement between oc and cc blocks where a new fill-in is created.
  //             // The product is a (oc, oo)-sized block.
  //             else {
  //               // if (i <= block) {
  //                 if (!F.exists(i, j)) {
  //                   Matrix fill_in(rank, D(block, j, level).cols);
  //                   auto fill_splits =
  //                     fill_in.split({},
  //                                   std::vector<int64_t>(1, D(block, j, level).cols - rank));
  //                   // Update the oc block within the fill-ins.
  //                   matmul(lower_splits[2], right_splits[0], fill_splits[0],
  //                          false, false, -1.0, 1.0);
  //                   // Update the oo block within the fill-ins.
  //                   matmul(lower_splits[2], right_splits[1], fill_splits[1],
  //                          false, false, -1.0, 1.0);
  //                   fill_in_cols.insert(j);
  //                   F.insert(i, j, std::move(fill_in));
  //                 }
  //                 else {
  //                   Matrix& fill_in = F(i, j);
  //                   auto fill_splits =
  //                     fill_in.split({},
  //                                   std::vector<int64_t>(1, D(block, j, level).cols - rank));
  //                   // Update the oc block within the fill-ins.
  //                   matmul(lower_splits[2], right_splits[0], fill_splits[0],
  //                          false, false, -1.0, 1.0);
  //                   // Update the oo block within the fill-ins.
  //                   matmul(lower_splits[2], right_splits[1], fill_splits[1],
  //                          false, false, -1.0, 1.0);
  //                 }
  //               // }
  //             }
  //             // Schur's compliment between oc and cc blocks where the result exists
  //             // after the diagonal blocks. The fill-in generated here is always part
  //             // of a nb*nb dense block.
  //             // else {
  //             //   if (F.exists(i, j)) {
  //             //     Matrix& fill_in = F(i, j);
  //             //     auto fill_splits = SPLIT_DENSE(fill_in,
  //             //                                    D(i, block, level).rows - rank,
  //             //                                    D(block, j, level).cols - rank);

  //             //     // Update the oc block within the fill-ins.
  //             //     matmul(lower_splits[2], right_splits[0], fill_splits[2],
  //             //            false, false, -1.0, 1.0);
  //             //     // Update the oo block within the fill-ins.
  //             //     matmul(lower_splits[2], right_splits[1], fill_splits[3],
  //             //            false, false, -1.0, 1.0);
  //             //   }
  //             //   else {
  //             //     std::cout << "A fill-in block for (oc,oo) does not exist where "
  //             //               << " it is supposed to at block-> "
  //             //               << block << " i-> " << i << " j-> " << j << std::endl;
  //             //     abort();
  //             //   }
  //             // }
  //           }
  //         }
  //       }
  //     }
  //   } // for (int64_t block = 0; block < nblocks; ++block)
  // }

  void H2::update_row_basis(int64_t row, int64_t level, RowColMap<Matrix>& F, RowMap& r) {
    int64_t nblocks = level_blocks[level];
    int64_t block_size = D(row, row, level).rows;
    Matrix row_block(block_size, 0);

    row_block = concat(row_block, matmul(U(row, level), Scol(row, level)), 1);
    for (int64_t j = 0; j < nblocks; ++j) {
      if (F.exists(row, j)) {
        if (F(row, j).rows == block_size && F(row, j).cols == rank)  {
          row_block = concat(row_block, matmul(F(row, j), V(j, level), false, true), 1);
        }
        else if (F(row, j).rows == block_size && F(row, j).cols == block_size) {
          row_block = concat(row_block, F(row, j), 1);
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

  void H2::update_col_basis(int64_t col, int64_t level, RowColMap<Matrix>& F, RowMap& t) {
    int64_t block_size = D(col, col, level).rows;
    int64_t nblocks = level_blocks[level];
    Matrix col_block(0, block_size);

    col_block = concat(col_block,
                       matmul(Srow(col, level), V(col, level), false, true), 0);
    for (int64_t i = 0; i < nblocks; ++i) {
      if (F.exists(i, col)) {
        if (F(i, col).rows == rank && F(i, col).cols == block_size) {
          col_block = concat(col_block, matmul(U(i, level), F(i, col)), 0);
        }
        else if (F(i, col).rows == block_size && F(i, col).cols == block_size) {
          col_block = concat(col_block, F(i, col), 0);
        }
      }
    }

    Matrix _UN_col, SN_col, VNT_col; double error;
    std::tie(_UN_col, SN_col, VNT_col, error) = truncated_svd(col_block, rank);

    Matrix t_col = matmul(V(col, level), VNT_col, true, true);
    V.erase(col, level);
    V.insert(col, level, transpose(VNT_col));

    Srow.erase(col, level);
    Srow.insert(col, level, std::move(SN_col));

    if (t.exists(col)) { t.erase(col); }
    t.insert(col, std::move(t_col));
  }

  void H2::factorize_level(int64_t level, int64_t nblocks, const Domain& domain,
                           RowMap& r, RowMap& t) {
    RowColMap<Matrix> F;      // fill-in blocks.

    std::vector<int64_t> r_indices, t_indices;

    // row indices of row fill-ins, except (oc,oo) type (rank x block_size) sized fill-ins.
    std::set<int64_t> fill_in_row_indices;
    // col indices of col fill-ins, except (co;oo) type (block_size x rank) sized fill-ins.
    std::set<int64_t> fill_in_col_indices;

    for (int block = 0; block < nblocks; ++block) {

      if (block > 0) {
        int64_t block_size = U(block, level).rows;
        {
          // Scan for fill-ins in the same row as this diagonal block.
          bool found_row_fill_in = fill_in_row_indices.count(block) != 0;
          bool found_col_fill_in = fill_in_col_indices.count(block) != 0;

          if (found_row_fill_in) {
            update_row_basis(block, level, F, r);

            for (auto col_iter = fill_in_col_indices.begin(); col_iter != fill_in_col_indices.end(); ++col_iter) {
              int64_t j = *col_iter;

              if (F.exists(block, j) && F(block,j).rows == block_size && F(block,j).cols == block_size) {
                update_col_basis(j, level, F, t);
              }
            }
          } // if (found_row_fill_in)


          if (found_col_fill_in) {
            update_col_basis(block, level, F, t);

            for (auto row_iter = fill_in_row_indices.begin(); row_iter != fill_in_row_indices.end(); ++row_iter) {
              int64_t i = *row_iter;

              if (F.exists(i, block) && F(i, block).rows == block_size && F(i, block).cols == block_size) {
                update_row_basis(i,level, F, r);
              }
            }
          }

          if (found_row_fill_in) {
            for (int j = 0; j < nblocks; ++j) {
              if (is_admissible.exists(block, j, level) && is_admissible(block, j, level)) {
                Matrix Sbar_block_j(rank, rank);

                if (F.exists(block, j)) {
                  if (F(block, j).rows == block_size && F(block, j).cols == rank) {
                    Sbar_block_j = matmul(r(block), S(block, j, level)) +
                      matmul(U(block, level), F(block, j), true, false);
                  }
                  else if (F(block, j).rows == block_size && F(block, j).cols == block_size) {
                    Sbar_block_j = matmul(matmul(r(block), S(block, j, level)), t(j)) +
                      matmul(matmul(U(block, level), F(block, j), true, false), V(j, level));

                    // Update S blocks for the column of the block with nb * nb fill-in.
                    for (int64_t i = 0; i < nblocks; ++i) {
                      if (is_admissible.exists(i, j, level) && is_admissible(i, j, level)) {
                        Matrix Sbar_ij(rank, rank);
                        if (i != block) {
                          Sbar_ij = matmul(S(i, j, level), t(j));

                          S.erase(i, j, level);
                          S.insert(i, j, level, std::move(Sbar_ij));
                        }
                      }
                    }
                  }
                  F.erase(block, j);
                }
                else {
                  Sbar_block_j = matmul(r(block), S(block, j, level));
                }

                S.erase(block, j, level);
                S.insert(block, j, level, std::move(Sbar_block_j));
              }
            }
          }

          if (found_col_fill_in) {
            for (int i = 0; i < nblocks; ++i) {
              if (is_admissible.exists(i, block, level) && is_admissible(i, block, level)) {
                Matrix Sbar_i_block(rank, rank);
                if (F.exists(i, block)) {
                  if (F(i, block).rows == rank && F(i, block).cols == block_size) {
                    Sbar_i_block = matmul(S(i, block,level), t(block)) + matmul(F(i, block), V(block, level));
                  }
                  else if (F(i, block).rows == block_size && F(i, block).cols == block_size) {
                    Sbar_i_block = matmul(S(i, block,level), t(block)) +
                      matmul(U(i, level), matmul(F(i, block), V(block, level)), true, false);

                    for (int64_t j = 0; j < nblocks; ++j) {
                      if (is_admissible.exists(i, j, level) && is_admissible(i, j, level)) {
                        Matrix Sbar_ij(rank, rank);
                        if (j != block) {
                          Sbar_ij = matmul(r(i), Sbar_ij);

                          S.erase(i, j, level);
                          S.insert(i, j, level, std::move(Sbar_ij));
                        }
                      }
                    }
                  }
                  F.erase(i, block);
                }
                else {
                  Sbar_i_block = matmul(S(i, block, level), t(block));
                }

                S.erase(i, block, level);
                S.insert(i, block, level, std::move(Sbar_i_block));
              }
            }
          }
        }
      }

      if (false) {
        int64_t nblocks = level_blocks[level];

        // Calculate row fill-ins
        for (auto row_fill_itr = fill_in_row_indices.begin(); row_fill_itr != fill_in_row_indices.end(); ++row_fill_itr) {
          int64_t i = *row_fill_itr;
          int64_t block_size = D(i, i, level).rows;

          Matrix row_i(block_size, 0);

          row_i = concat(row_i, matmul(U(i, level), Scol(i, level)), 1);
          for (int64_t j = 0; j < nblocks; ++j) {
            if (F.exists(i, j)) {
              Matrix Ui, Si, Vi; double error;
              Matrix cpy(F(i, j));
              std::tie(Ui, Si, Vi) = error_svd(cpy, 1e-9);

              if (F(i, j).rows == block_size && F(i, j).cols == rank)  {
                row_i = concat(row_i, matmul(F(i, j), V(j, level), false, true), 1);
              }
              else if (F(i, j).rows == block_size && F(i, j).cols == block_size) {
                row_i = concat(row_i, F(i, j), 1);
              }
            }
          }

          Matrix UN_i, SN_i, _VNT_i; double error;
          std::tie(UN_i, SN_i, _VNT_i, error) = truncated_svd(row_i, rank);

          Matrix r_i = matmul(UN_i, U(i, level), true, false);

          U.erase(i, level);
          U.insert(i, level, std::move(UN_i));

          Scol.erase(i, level);
          Scol.insert(i, level, std::move(SN_i));


          r_indices.push_back(i);
          if (r.exists(i)) { r.erase(i); }
          r.insert(i, std::move(r_i));
        }

        // Calculate col fill-ins
        for (auto col_fill_itr = fill_in_col_indices.begin();
             col_fill_itr != fill_in_col_indices.end(); ++col_fill_itr) {
          int64_t j = *col_fill_itr;
          int64_t block_size = D(j, j, level).rows;
          Matrix col_j(0, block_size);
          col_j = concat(col_j,
                         matmul(Srow(j, level), V(j, level), false, true), 0);
          for (int64_t i = 0; i < nblocks; ++i) {
            if (F.exists(i, j)) {
              Matrix Ui, Si, Vi; double error;
              Matrix cpy(F(i, j));
              std::tie(Ui, Si, Vi) = error_svd(cpy, 1e-9);

              if (F(i, j).rows == rank && F(i, j).cols == block_size) {
                col_j = concat(col_j, matmul(U(i, level), F(i, j)), 0);
              }
              else if (F(i, j).rows == block_size && F(i, j).cols == block_size) {
                col_j = concat(col_j, F(i, j), 0);
              }
            }
          }

          Matrix _UN_j, SN_j, VNT_j; double error;
          std::tie(_UN_j, SN_j, VNT_j, error) = truncated_svd(col_j, rank);

          Matrix t_j = matmul(V(j, level), VNT_j, true, true);
          V.erase(j, level);
          V.insert(j, level, transpose(VNT_j));

          Srow.erase(j, level);
          Srow.insert(j, level, std::move(SN_j));

          t_indices.push_back(j);
          if (t.exists(j)) { t.erase(j); }
          t.insert(j, std::move(t_j));
        }

        // Update S blocks in the rows.
        for (int64_t i = 0; i < r_indices.size(); ++i) {
          int64_t r_index = r_indices[i];
          Matrix& r_i = r(r_index);

          for (int64_t j = 0; j < nblocks; ++j) {
            if (is_admissible.exists(r_index, j, level) && is_admissible(r_index, j, level)) {
              int64_t block_size = D(j, j, level).cols;
              Matrix Sbar_ij(rank, rank);
              if (F.exists(r_index, j)) {
                if (F(r_index, j).rows == block_size && F(r_index, j).cols == rank) {
                  Sbar_ij = matmul(r_i, S(r_index, j, level)) + matmul(U(r_index, level), F(r_index, j), true, false);
                }
                else if (F(r_index, j).rows == block_size && F(r_index, j).cols == block_size) {
                  Sbar_ij = matmul(matmul(r_i, S(r_index, j, level)), t(j)) +
                    matmul(matmul(U(r_index, level), F(r_index, j)), V(j, level));
                }
              }
              else {
                Sbar_ij = matmul(r_i, S(r_index, j, level));
              }

              S.erase(r_index, j, level);
              S.insert(r_index, j, level, std::move(Sbar_ij));
            }

            if (F.exists(r_index, j)) {
              // F.erase(r_index, j);
            }
          }
        }

        // Update S blocks in the cols
        for (int64_t j = 0; j < t_indices.size(); ++j) {
          int64_t t_index = t_indices[j];
          Matrix& t_j = t(t_index);

          for (int64_t i = 0; i < nblocks; ++i) {
            int64_t block_size = D(i, i, level).cols;
            if (is_admissible.exists(i, t_index, level) && is_admissible(i, t_index, level)) {
              // std::cout << "COl update: i-> " << i <<  " t_index-> " << t_index << std::endl;
              Matrix Sbar_ij(rank, rank);
              if (F.exists(i, t_index)) {
                if (F(i, t_index).rows == block_size && F(i, t_index).cols == block_size) {
                  continue;
                }

                if (F(i, t_index).rows == rank && F(i, t_index).cols == block_size) {
                  Sbar_ij = matmul(S(i, t_index, level), t_j) +
                    matmul(F(i, t_index), V(t_index, level));
                }
                //F.erase(i, j);
              }
              else {
                Sbar_ij = matmul(S(i, t_index, level), t_j);
              }

              S.erase(i, t_index, level);
              S.insert(i, t_index, level, std::move(Sbar_ij));
            }
            if (F.exists(i, t_index)) {
              // F.erase(i, t_index);
            }
          }
        }
      }


      r_indices.clear();
      t_indices.clear();
      fill_in_row_indices.clear();
      fill_in_col_indices.clear();
      // F.erase_all();

      if (level == 2) {
        std::cout << "Norm<0,1>: " << norm(D(0,1,level)) << std::endl;
      }

      Matrix U_F = make_complement(U(block, level));
      Matrix V_F = make_complement(V(block, level));

      for (int j = 0; j < nblocks; ++j) {
        if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) {
          D(block, j, level) = matmul(U_F, D(block, j, level), true);
        }
      }

      for (int i = 0; i < nblocks; ++i) {
        if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) {
          D(i, block, level) = matmul(D(i, block, level), V_F);
        }
      }

      int64_t row_split = D(block, block, level).rows - rank,
        col_split = D(block, block, level).cols - rank;

      // The diagonal block is split along the row and column.
      auto diagonal_splits = SPLIT_DENSE(D(block, block, level), row_split, col_split);
      Matrix& Dcc = diagonal_splits[0];
      lu(Dcc);

      // TRSM with CC blocks on the row
      for (int64_t j = block + 1; j < nblocks; ++j) {
        if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) {
          int64_t col_split = D(block, j, level).cols - rank;
          auto D_splits = SPLIT_DENSE(D(block, j, level), row_split, col_split);
          solve_triangular(Dcc, D_splits[0], Hatrix::Left, Hatrix::Lower, true);
        }
      }

      // TRSM with co blocks on this row
      for (int64_t j = 0; j < nblocks; ++j) {
        if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) {
          int64_t col_split = D(block, j, level).rows - rank;
          auto D_splits = SPLIT_DENSE(D(block, j, level), row_split, col_split);
          solve_triangular(Dcc, D_splits[1], Hatrix::Left, Hatrix::Lower, true);
        }
      }

      // TRSM with cc blocks on the column
      for (int64_t i = block + 1; i < nblocks; ++i) {
        if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) {
          int64_t row_split = D(i, block, level).rows - rank;
          auto D_splits = SPLIT_DENSE(D(i, block, level), row_split, col_split);
          solve_triangular(Dcc, D_splits[0], Hatrix::Right, Hatrix::Upper, false);
        }
      }

      // TRSM with oc blocks on the column
      for (int64_t i = 0; i < nblocks; ++i) {
        if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) {
          auto D_splits = SPLIT_DENSE(D(i, block, level),
                                      D(i, block, level).rows - rank,
                                      col_split);
          solve_triangular(Dcc, D_splits[2], Hatrix::Right, Hatrix::Upper, false);
        }
      }

      // Fill in between cc blocks.
      for (int64_t i = block+1; i < nblocks; ++i) {
        for (int64_t j = block+1; j < nblocks; ++j) {
          if ((is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) &&
              (is_admissible.exists(i, block, level) && !is_admissible(i, block, level))) {
            auto lower_splits = SPLIT_DENSE(D(i, block, level),
                                            D(i, block, level).rows - rank,
                                            col_split);
            auto right_splits = SPLIT_DENSE(D(block, j, level),
                                            row_split,
                                            D(block, j, level).cols - rank);

            if (is_admissible.exists(i, j, level) && !is_admissible(i, j, level)) {
              auto reduce_splits = SPLIT_DENSE(D(i, j, level),
                                               D(i, j, level).rows - rank,
                                               D(i, j, level).cols - rank);

              matmul(lower_splits[0], right_splits[0], reduce_splits[0], false, false, -1.0, 1.0);
            }
            else {

              int64_t rows = D(i, block, level).rows;
              int64_t cols = D(block, j, level).cols;
              if (F.exists(i, j)) {
                Matrix& fill_in = F(i, j);
                auto fill_in_splits = SPLIT_DENSE(fill_in, rows - rank, cols - rank);
                matmul(lower_splits[0], right_splits[0], fill_in_splits[0],
                       false, false, -1.0, 1.0);
              }
              else {
                Matrix fill_in(rows, cols);
                auto fill_in_splits = SPLIT_DENSE(fill_in, rows - rank, cols - rank);
                matmul(lower_splits[0], right_splits[0], fill_in_splits[0],
                       false, false, -1.0, 1.0);

                fill_in_row_indices.insert(i);
                fill_in_col_indices.insert(j);
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
            auto lower_splits = SPLIT_DENSE(D(i, block, level),
                                            D(i, block, level).rows - rank,
                                            col_split);
            auto right_splits = SPLIT_DENSE(D(block, j, level),
                                            row_split,
                                            D(block, j, level).cols - rank);

            if (is_admissible.exists(i, j, level) && !is_admissible(i, j, level)) {
              // no fill-in in the oo portion. SC into another dense block.
              auto reduce_splits = SPLIT_DENSE(D(i, j, level),
                                               D(i, j, level).rows - rank,
                                               D(i, j, level).cols - rank);
              matmul(lower_splits[2], right_splits[1], reduce_splits[3], false, false, -1.0, 1.0);
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
            auto lower_splits = SPLIT_DENSE(D(i, block, level),
                                            D(i, block, level).rows - rank,
                                            col_split);
            auto right_splits = SPLIT_DENSE(D(block, j, level),
                                            row_split,
                                            D(block, j, level).cols - rank);
            // Schur's compliment between co and cc blocks where product exists as dense.
            if (is_admissible.exists(i, j, level) && !is_admissible(i, j, level)) {
              auto reduce_splits = SPLIT_DENSE(D(i, j, level),
                                               D(i, j, level).rows - rank,
                                               D(i, j, level).cols - rank);
              matmul(lower_splits[0], right_splits[1], reduce_splits[1], false, false, -1.0, 1.0);
            }
            // Schur's compliement between co and cc blocks where a new fill-in is created.
            // The product is a (co; oo)-sized matrix.
            else {
              // The Schur's compliments that are formed before the block index are always
              // a narrow strip of size nb * rank. These blocks are formed only on the right
              // part of the permuted matrix in the co section.
              if (j <= block) {
                if (!F.exists(i, j)) {
                  Matrix fill_in(D(i, block, level).rows, rank);
                  auto fill_splits =
                    fill_in.split(std::vector<int64_t>(1, D(i, block, level).rows - rank),
                                  {});
                  // Update the co block within the fill-in.
                  matmul(lower_splits[0], right_splits[1], fill_splits[0],
                         false, false, -1.0, 1.0);

                  // Update the oo block within the fill-in.
                  matmul(lower_splits[2], right_splits[1], fill_splits[1],
                         false, false, -1.0, 1.0);
                  fill_in_row_indices.insert(i);
                  F.insert(i, j, std::move(fill_in));
                }

                else {
                  Matrix &fill_in = F(i, j);
                  auto fill_splits =
                    fill_in.split(std::vector<int64_t>(1, D(i, block, level).rows - rank),
                                  {});
                  // Update the co block within the fill-in.
                  matmul(lower_splits[0], right_splits[1], fill_splits[0],
                         false, false, -1.0, 1.0);
                  // Update the oo block within the fill-in.
                  matmul(lower_splits[2], right_splits[1], fill_splits[1],
                         false, false, -1.0, 1.0);
                }
              }
              // Schur's compliment between co and cc blocks where the result exists
              // after the diagonal blocks. The fill-in generated here is always part
              // of a nb*nb dense block. Thus we grab the large fill-in block that was
              // already formed previously in the cc * cc schur's compliment computation,
              // and add the resulting schur's compliment into that previously generated block.
              else {
                if (F.exists(i, j)) {
                  Matrix& fill_in = F(i, j);
                  auto fill_splits = SPLIT_DENSE(fill_in,
                                                 D(i, block, level).rows - rank,
                                                 D(block, j, level).cols - rank);
                  // Update the co block within the fill-in.
                  matmul(lower_splits[0], right_splits[1], fill_splits[1],
                         false, false, -1.0, 1.0);
                  // Update the oo block within the fill-in.
                  matmul(lower_splits[2], right_splits[1], fill_splits[3],
                         false, false, -1.0, 1.0);
                }
                else {
                  std::cout << "A fill-in block for (co;oo) does not exist where it is supposed to at block-> "
                            << block << " i-> " << i << " j-> " << j << std::endl;
                  abort();
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
            auto lower_splits = SPLIT_DENSE(D(i, block, level),
                                            D(i, block, level).rows - rank,
                                            col_split);
            auto right_splits = SPLIT_DENSE(D(block, j, level),
                                            row_split,
                                            D(block, j, level).cols - rank);
            // Schur's compliement between oc and cc blocks where product exists as dense.
            if (is_admissible.exists(i, j, level) && !is_admissible(i, j, level)) {
              auto reduce_splits = SPLIT_DENSE(D(i, j, level),
                                               D(i, j, level).rows - rank,
                                               D(i, j, level).cols - rank);
              matmul(lower_splits[2], right_splits[0], reduce_splits[2],
                     false, false, -1.0, 1.0);
            }
            // Schur's compliement between oc and cc blocks where a new fill-in is created.
            // The product is a (oc, oo)-sized block.
            else {
              if (i <= block) {
                if (!F.exists(i, j)) {
                  Matrix fill_in(rank, D(block, j, level).cols);
                  auto fill_splits =
                    fill_in.split({},
                                  std::vector<int64_t>(1, D(block, j, level).cols - rank));
                  // Update the oc block within the fill-ins.
                  matmul(lower_splits[2], right_splits[0], fill_splits[0],
                         false, false, -1.0, 1.0);
                  // Update the oo block within the fill-ins.
                  matmul(lower_splits[2], right_splits[1], fill_splits[1],
                         false, false, -1.0, 1.0);
                  fill_in_col_indices.insert(j);
                  F.insert(i, j, std::move(fill_in));
                }
                else {
                  Matrix& fill_in = F(i, j);
                  auto fill_splits =
                    fill_in.split({},
                                  std::vector<int64_t>(1, D(block, j, level).cols - rank));
                  // Update the oc block within the fill-ins.
                  matmul(lower_splits[2], right_splits[0], fill_splits[0],
                         false, false, -1.0, 1.0);
                  // Update the oo block within the fill-ins.
                  matmul(lower_splits[2], right_splits[1], fill_splits[1],
                         false, false, -1.0, 1.0);
                }
              }

              // Schur's compliment between oc and cc blocks where the result exists
              // after the diagonal blocks. The fill-in generated here is always part
              // of a nb*nb dense block.
              else {
                if (F.exists(i, j)) {
                  Matrix& fill_in = F(i, j);
                  auto fill_splits = SPLIT_DENSE(fill_in,
                                                 D(i, block, level).rows - rank,
                                                 D(block, j, level).cols - rank);

                  // Update the oc block within the fill-ins.
                  matmul(lower_splits[2], right_splits[0], fill_splits[2],
                         false, false, -1.0, 1.0);
                  // Update the oo block within the fill-ins.
                  matmul(lower_splits[2], right_splits[1], fill_splits[3],
                         false, false, -1.0, 1.0);
                }
                else {
                  std::cout << "A fill-in block for (oc,oo) does not exist where "
                            << " it is supposed to at block-> "
                            << block << " i-> " << i << " j-> " << j << std::endl;
                  abort();
                }
              }
            }
          }
        }
      }
    } // for (int block = 0; block < nblocks; ++block)

    // F.erase_all();
  }


  void
  H2::factorize(const Domain& domain) {
    int64_t level = height;
    RowColLevelMap<Matrix> F;
    RowMap r, t;

    A1_expected = generate_identity_matrix(rank * 8, rank * 8);
    auto A1_expected_splits = A1_expected.split(4, 4);

    A2_expected = generate_identity_matrix(rank * 8, rank * 8);
    auto A2_expected_splits = A2_expected.split(4, 4);

    for (; level > 0; --level) {
      int64_t nblocks = level_blocks[level];
      bool is_all_dense_level = false;
      for (int64_t i = 0; i < nblocks; ++i) {
        if (!U.exists(i, level)) {
          is_all_dense_level = true;
        }
      }

      if (is_all_dense_level) {
        break;
      }

      if (level == 2) {
        for (int64_t i = 0; i < 4; ++i) {
          for (int64_t j = 0; j < 4; ++j) {
            if (D.exists(i, j, level)) {
              A2_expected_splits[i * 4 + j] = D(i, j, level);
            }
          }
        }
      }

      factorize_level(level, nblocks, domain, r, t);

      int64_t parent_level = level - 1;

      // Update transfer matrices on one level higher.
      for (int64_t block = 0; block < nblocks; block += 2) {
        int64_t parent_node = block / 2;
        int64_t c1 = block;
        int64_t c2 = block+1;

        if (row_has_admissible_blocks(parent_node, parent_level) && height != 1) {
          Matrix& Utransfer = U(parent_node, parent_level);
          auto Utransfer_splits = Utransfer.split(2, 1);

          Matrix temp(Utransfer);
          auto temp_splits = temp.split(2, 1);

          if (r.exists(c1)) {
            // std::cout << "C1 update: " << c1 << " norm: " << norm(r(c1)) << std::endl;
            matmul(r(c1), Utransfer_splits[0], temp_splits[0], false, false, 1, 0);
            r.erase(c1);
          }

          if (r.exists(c2)) {
            // std::cout << "C2 update: " << c2 <<  " norm: " << norm(r(c2)) << std::endl;
            matmul(r(c2), Utransfer_splits[1], temp_splits[1], false, false, 1, 0);
            r.erase(c2);
          }

          U.erase(parent_node, parent_level);
          U.insert(parent_node, parent_level, std::move(temp));

          // if (parent_level == 2) {
          //   std::cout << "FAR DENSE parent_node -> " << parent_node
          //             << " parent_level -> " << parent_level
          //             << norm(generate_identity_matrix(rank, rank) -
          //                     matmul(U(parent_node, parent_level),
          //                            U(parent_node, parent_level), true, false))
          //             << std::endl;
          // }

        }

        if (col_has_admissible_blocks(parent_node, parent_level) && height != 1) {
          Matrix& Vtransfer = V(parent_node, parent_level);
          auto Vtransfer_splits = Vtransfer.split(2, 1);

          Matrix temp(Vtransfer);
          auto temp_splits = temp.split(2, 1);

          if (t.exists(c1)) {
            matmul(t(c1), Vtransfer_splits[0], temp_splits[0], false, false, 1, 0);
            t.erase(c1);
          }

          if (t.exists(c2)) {
            matmul(t(c2), Vtransfer_splits[1], temp_splits[1], false, false, 1, 0);
            t.erase(c2);
          }

          V.erase(parent_node, parent_level);
          V.insert(parent_node, parent_level, std::move(temp));
        }
      } // for (block = 0; block < nblocks; block += 2)

      // Merge the unfactorized parts.
      int64_t parent_nblocks = level_blocks[parent_level];
      std::cout << "START MERGE PARENT LEVEL: " << parent_level << " blocks: " << std::endl;

      for (int64_t i = 0; i < parent_nblocks; ++i) {
        for (int64_t j = 0; j < parent_nblocks; ++j) {
          if (is_admissible.exists(i, j, parent_level) && !is_admissible(i, j, parent_level)) {
            // TODO: need to switch to morton indexing so finding the parent is straightforward.
            std::vector<int64_t> i_children, j_children;
            int64_t nrows=0, ncols=0;
            if (matrix_type == BLR2_MATRIX) {
              for (int n = 0; n < level_blocks[level]; ++n) {
                i_children.push_back(n);
                j_children.push_back(n);

                nrows += rank;
                ncols += rank;
              }
            }
            else if (matrix_type == H2_MATRIX) {
              for (int n = 0; n < 2; ++n) {
                i_children.push_back(i * 2 + n);
                j_children.push_back(j * 2 + n);

                nrows += rank;
                ncols += rank;
              }
            }
            Matrix D_unelim(nrows, ncols);
            auto D_unelim_splits = D_unelim.split(i_children.size(), j_children.size());

            for (int64_t ic1 = 0; ic1 < i_children.size(); ++ic1) {
              for (int64_t jc2 = 0; jc2 < j_children.size(); ++jc2) {
                int64_t c1 = i_children[ic1], c2 = j_children[jc2];
                if (!U.exists(c1, level)) { continue; }

                if (is_admissible.exists(c1, c2, level) && !is_admissible(c1, c2, level)) {

                  auto D_splits = SPLIT_DENSE(D(c1, c2, level),
                                              D(c1, c2, level).rows-rank,
                                              D(c1, c2, level).cols-rank);
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

    int64_t last_nodes = level_blocks[level];

    std::cout << "last nodes: " << last_nodes << " level: " << level << std::endl;
    // Capture unfactorized A1 block.
    if (level == 1 && height != 1) {
      for (int64_t i = 0; i < last_nodes; ++i) {
        for (int64_t j = 0; j < last_nodes; ++j) {
          A1_expected_splits[(i + 2) * 4 + j + 2] = D(i, j, level);
        }
      }
    }


    for (int64_t d = 0; d < last_nodes; ++d) {
      lu(D(d, d, level));
      for (int64_t j = d+1; j < last_nodes; ++j) {
        solve_triangular(D(d, d, level), D(d, j, level), Hatrix::Left, Hatrix::Lower, true);
      }
      for (int64_t i = d+1; i < last_nodes; ++i) {
        solve_triangular(D(d, d, level), D(i, d, level), Hatrix::Right, Hatrix::Upper, false);
      }

      for (int64_t i = d+1; i < last_nodes; ++i) {
        for (int64_t j = d+1; j < last_nodes; ++j) {
          matmul(D(i, d, level), D(d, j, level), D(i, j, level), false, false, -1.0, 1.0);
        }
      }
    }
  }


  // permute the vector forward and return the offset at which the new vector begins.
  int64_t H2::permute_forward(Matrix& x, const int64_t level, int64_t rank_offset) {
    Matrix copy(x);
    int64_t num_nodes = level_blocks[level];
    int64_t c_offset = rank_offset;
    for (int64_t block = 0; block < num_nodes; ++block) {
      rank_offset += D(block, block, level).rows - rank;
    }

    int64_t csize_offset = 0, bsize_offset = 0, rsize_offset = 0;
    for (int64_t block = 0; block < num_nodes; ++block) {
      int64_t rows = D(block, block, level).rows;
      int64_t c_size = rows - rank;

      // copy the complement part of the vector into the temporary vector
      for (int64_t i = 0; i < c_size; ++i) {
        copy(c_offset + csize_offset + i, 0) = x(c_offset + bsize_offset + i, 0);
      }
      // copy the rank part of the vector into the temporary vector
      for (int64_t i = 0; i < rank; ++i) {
        copy(rank_offset + rsize_offset + i, 0) = x(c_offset + bsize_offset + c_size + i, 0);
      }

      csize_offset += c_size;
      bsize_offset += rows;
      rsize_offset += rank;
    }

    x = copy;

    return rank_offset;
  }

  int64_t H2::permute_backward(Matrix& x, const int64_t level, int64_t rank_offset) {
    Matrix copy(x);
    int64_t num_nodes = level_blocks[level];
    int64_t c_offset = rank_offset;
    for (int64_t block = 0; block < num_nodes; ++block) {
      c_offset -= D(block, block, level).cols - rank;
    }

    int64_t csize_offset = 0, bsize_offset = 0, rsize_offset = 0;
    for (int64_t block = 0; block < num_nodes; ++block) {
      int64_t cols = D(block, block, level).cols;
      int64_t c_size = cols - rank;

      for (int64_t i = 0; i < c_size; ++i) {
        copy(c_offset + bsize_offset + i, 0) = x(c_offset + csize_offset + i, 0);
      }

      for (int64_t i = 0; i < rank; ++i) {
        copy(c_offset + bsize_offset + c_size + i, 0) = x(rank_offset + rsize_offset + i, 0);
      }

      csize_offset += c_size;
      bsize_offset += cols;
      rsize_offset += rank;
    }

    x = copy;

    return c_offset;
  }

  void H2::solve_forward_level(Matrix& x_level, int64_t level) {
    int64_t nblocks = level_blocks[level];
    std::vector<int64_t> row_offsets;
    int64_t nrows = 0;
    for (int64_t i = 0; i < nblocks; ++i) {
      row_offsets.push_back(nrows + D(i, i, level).rows);
      nrows += D(i, i, level).rows;
    }
    std::vector<Matrix> x_level_split = x_level.split(row_offsets, {});

    for (int64_t block = 0; block < nblocks; ++block) {
      Matrix U_F = make_complement(U(block, level));
      Matrix prod = matmul(U_F, x_level_split[block], true);
      x_level_split[block] = prod;
    }

    // forward substitution with cc blocks
    for (int64_t block = 0; block < nblocks; ++block) {
      int64_t row_split = D(block, block, level).rows - rank;
      int64_t col_split = D(block, block, level).cols - rank;
      auto block_splits = SPLIT_DENSE(D(block, block, level), row_split, col_split);

      Matrix x_block(x_level_split[block]);
      auto x_block_splits = x_block.split(std::vector<int64_t>(1, row_split), {});

      solve_triangular(block_splits[0], x_block_splits[0], Hatrix::Left, Hatrix::Lower, true);
      matmul(block_splits[2], x_block_splits[0], x_block_splits[1], false, false, -1.0, 1.0);
      x_level_split[block] = x_block;

      // Forward with the big c blocks on the lower part.
      for (int64_t irow = block+1; irow < nblocks; ++irow) {
        if (is_admissible.exists(irow, block, level) && !is_admissible(irow, block, level)) {
          int64_t row_split = D(irow, block, level).rows - rank;
          int64_t col_split = D(irow, block, level).cols - rank;
          auto lower_splits = D(irow, block, level).split({}, std::vector<int64_t>(1, row_split));

          Matrix x_block(x_level_split[block]), x_level_irow(x_level_split[irow]);
          auto x_block_splits = x_block.split(std::vector<int64_t>(1, col_split), {});

          matmul(lower_splits[0], x_block_splits[0], x_level_irow, false, false, -1.0, 1.0);
          x_level_split[irow] = x_level_irow;
        }
      }

      // Forward with the oc parts of the block that are actually in the upper part of the matrix.
      for (int64_t irow = 0; irow < block; ++irow) {
        if (is_admissible.exists(irow, block, level) && !is_admissible(irow, block, level)) {
          int64_t row_split = D(irow, block, level).rows - rank;
          int64_t col_split = D(irow, block, level).cols - rank;
          auto top_splits = SPLIT_DENSE(D(irow, block, level), row_split, col_split);

          Matrix x_irow(x_level_split[irow]), x_block(x_level_split[block]);
          auto x_irow_splits = x_irow.split(std::vector<int64_t>(1, row_split), {});
          auto x_block_splits = x_block.split(std::vector<int64_t>(1, col_split), {});

          matmul(top_splits[2], x_block_splits[0], x_irow_splits[1], false, false, -1.0, 1.0);

          x_level_split[irow] = x_irow;
        }
      }
    }
  }

  void H2::solve_backward_level(Matrix& x_level, int64_t level) {
    int64_t nblocks = level_blocks[level];
    std::vector<int64_t> col_offsets;
    int64_t nrows = 0;
    for (int64_t i = 0; i < nblocks; ++i) {
      col_offsets.push_back(nrows + D(i, i, level).cols);
      nrows += D(i, i, level).cols;
    }
    std::vector<Matrix> x_level_split = x_level.split(col_offsets, {});

    // backward substition using cc blocks
    for (int64_t block = nblocks-1; block >= 0; --block) {
      int64_t row_split = D(block, block, level).rows - rank;
      int64_t col_split = D(block, block, level).cols - rank;
      auto block_splits = SPLIT_DENSE(D(block, block, level), row_split, col_split);
      // Apply co block.
      for (int64_t left_col = block-1; left_col >= 0; --left_col) {
        if (is_admissible.exists(block, left_col, level) &&
            !is_admissible(block, left_col, level)) {
          auto left_splits = SPLIT_DENSE(D(block, left_col, level), row_split, col_split);

          Matrix x_block(x_level_split[block]), x_left_col(x_level_split[left_col]);
          auto x_block_splits = x_block.split(std::vector<int64_t>(1, row_split), {});
          auto x_left_col_splits = x_left_col.split(std::vector<int64_t>(1, col_split), {});

          matmul(left_splits[1], x_left_col_splits[1], x_block_splits[0], false, false, -1.0, 1.0);

          x_level_split[block] = x_block;
        }
      }

      // Apply c block present on the right of this diagonal block.
      for (int64_t right_col = nblocks-1; right_col > block; --right_col) {
        if (is_admissible.exists(block, right_col, level) &&
            !is_admissible(block, right_col, level)) {
          int64_t row_split = D(block, right_col, level).rows - rank;
          auto right_splits = D(block, right_col, level).
            split(std::vector<int64_t>(1, row_split), {});

          Matrix x_block(x_level_split[block]);
          auto x_block_splits = x_block.split(std::vector<int64_t>(1, row_split), {});

          matmul(right_splits[0], x_level_split[right_col],
                 x_block_splits[0], false, false, -1.0, 1.0);
          x_level_split[block] = x_block;
        }
      }

      Matrix x_block(x_level_split[block]);
      auto x_block_splits = x_block.split(std::vector<int64_t>(1, row_split), {});
      matmul(block_splits[1], x_block_splits[1], x_block_splits[0], false, false, -1.0, 1.0);
      solve_triangular(block_splits[0], x_block_splits[0], Hatrix::Left, Hatrix::Upper, false);
      x_level_split[block] = x_block;
    }

    for (int64_t block = nblocks-1; block >= 0; --block) {
      auto V_F = make_complement(V(block, level));
      Matrix prod = matmul(V_F, x_level_split[block]);
      x_level_split[block] = prod;
    }
  }


  Matrix
  H2::solve(const Matrix& b, int64_t _level) {
    Matrix x(b);
    int64_t level = _level;
    int64_t rhs_offset = 0;
    std::vector<Matrix> x_splits;

    // Forward
    for (; level > 0; --level) {
      int64_t nblocks = level_blocks[level];
      bool lr_exists = false;
      for (int64_t block = 0; block < nblocks; ++block) {
        if (U.exists(block, level)) { lr_exists = true; }
      }
      if (!lr_exists) { break; }

      int64_t n = 0;
      for (int64_t i = 0; i < nblocks; ++i) { n += D(i, i, level).rows; }
      Matrix x_level(n, 1);
      for (int64_t i = 0; i < x_level.rows; ++i) {
        x_level(i, 0) = x(rhs_offset + i, 0);
      }

      solve_forward_level(x_level, level);

      for (int64_t i = 0; i < x_level.rows; ++i) {
        x(rhs_offset + i, 0) = x_level(i, 0);
      }

      rhs_offset = permute_forward(x, level, rhs_offset);
    }

    // Work with L0 and U0
    x_splits = x.split(std::vector<int64_t>(1, rhs_offset), {});
    Matrix x_last(x_splits[1]);
    int64_t last_nodes = level_blocks[level];
    auto x_last_splits = x_last.split(last_nodes, 1);

    for (int64_t i = 0; i < last_nodes; ++i) {
      for (int64_t j = 0; j < i; ++j) {
        matmul(D(i, j, level), x_last_splits[j], x_last_splits[i], false, false, -1.0, 1.0);
      }
      solve_triangular(D(i, i, level), x_last_splits[i], Hatrix::Left, Hatrix::Lower, true);
    }

    for (int64_t i = last_nodes-1; i >= 0; --i) {
      for (int64_t j = last_nodes-1; j > i; --j) {
        matmul(D(i, j, level), x_last_splits[j], x_last_splits[i], false, false, -1.0, 1.0);
      }
      solve_triangular(D(i, i, level), x_last_splits[i], Hatrix::Left, Hatrix::Upper, false);
    }
    x_splits[1] = x_last;

    level++;

    // Backward
    for (; level <= _level; ++level) {
      int64_t nblocks = level_blocks[level];

      bool lr_exists = false;
      for (int64_t block = 0; block < nblocks; ++block) {
        if (V.exists(block, level)) { lr_exists = true; }
      }
      if (!lr_exists) { break; }

      int64_t n = 0;
      for (int64_t i = 0; i < nblocks; ++i) { n += D(i, i, level).cols; }
      Matrix x_level(n, 1);

      rhs_offset = permute_backward(x, level, rhs_offset);

      for (int64_t i = 0; i < x_level.rows; ++i) {
        x_level(i, 0) = x(rhs_offset + i, 0);
      }

      solve_backward_level(x_level, level);

      for (int64_t i = 0; i < x_level.rows; ++i) {
        x(rhs_offset + i, 0) = x_level(i, 0);
      }
    }

    return x;
  }


  void
  H2::coarsen_blocks(int64_t level) {
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

  int64_t
  H2::geometry_admis_non_leaf(int64_t nblocks, int64_t level) {
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

  int64_t
  H2::calc_geometry_based_admissibility(const Domain& domain) {
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

    // is_admissible.insert(0, 0, level, false);
    // is_admissible.insert(0, 1, level, false);
    // is_admissible.insert(0, 2, level, false);
    // is_admissible.insert(0, 3, level, true);
    // is_admissible.insert(0, 4, level, false);
    // is_admissible.insert(0, 5, level, true);
    // is_admissible.insert(0, 6, level, true);
    // is_admissible.insert(0, 7, level, true);

    // is_admissible.insert(1, 0, level, false);
    // is_admissible.insert(1, 1, level, false);
    // is_admissible.insert(1, 2, level, false);
    // is_admissible.insert(1, 3, level, false);
    // is_admissible.insert(1, 4, level, false);
    // is_admissible.insert(1, 5, level, false);
    // is_admissible.insert(1, 6, level, false);
    // is_admissible.insert(1, 7, level, true);

    // is_admissible.insert(2, 0, level, false);
    // is_admissible.insert(2, 1, level, false);
    // is_admissible.insert(2, 2, level, false);
    // is_admissible.insert(2, 3, level, false);
    // is_admissible.insert(2, 4, level, true);
    // is_admissible.insert(2, 5, level, true);
    // is_admissible.insert(2, 6, level, false);
    // is_admissible.insert(2, 7, level, true);

    // is_admissible.insert(3, 0, level, true);
    // is_admissible.insert(3, 1, level, false);
    // is_admissible.insert(3, 2, level, false);
    // is_admissible.insert(3, 3, level, false);
    // is_admissible.insert(3, 4, level, false);
    // is_admissible.insert(3, 5, level, true);
    // is_admissible.insert(3, 6, level, false);
    // is_admissible.insert(3, 7, level, false);

    // is_admissible.insert(4, 0, level, false);
    // is_admissible.insert(4, 1, level, false);
    // is_admissible.insert(4, 2, level, true);
    // is_admissible.insert(4, 3, level, false);
    // is_admissible.insert(4, 4, level, false);
    // is_admissible.insert(4, 5, level, false);
    // is_admissible.insert(4, 6, level, false);
    // is_admissible.insert(4, 7, level, true);

    // is_admissible.insert(5, 0, level, true);
    // is_admissible.insert(5, 1, level, false);
    // is_admissible.insert(5, 2, level, true);
    // is_admissible.insert(5, 3, level, true);
    // is_admissible.insert(5, 4, level, false);
    // is_admissible.insert(5, 5, level, false);
    // is_admissible.insert(5, 6, level, false);
    // is_admissible.insert(5, 7, level, false);

    // is_admissible.insert(6, 0, level, true);
    // is_admissible.insert(6, 1, level, false);
    // is_admissible.insert(6, 2, level, false);
    // is_admissible.insert(6, 3, level, false);
    // is_admissible.insert(6, 4, level, false);
    // is_admissible.insert(6, 5, level, false);
    // is_admissible.insert(6, 6, level, false);
    // is_admissible.insert(6, 7, level, false);

    // is_admissible.insert(7, 0, level, true);
    // is_admissible.insert(7, 1, level, true);
    // is_admissible.insert(7, 2, level, true);
    // is_admissible.insert(7, 3, level, false);
    // is_admissible.insert(7, 4, level, true);
    // is_admissible.insert(7, 5, level, false);
    // is_admissible.insert(7, 6, level, false);
    // is_admissible.insert(7, 7, level, false);

    if (matrix_type == BLR2_MATRIX) {
      level_blocks.push_back(1);
      return 1;
    }
    else {
      return geometry_admis_non_leaf(nblocks / 2, level+1);
    }
  }

  void
  H2::calc_diagonal_based_admissibility(int64_t level) {
    int64_t nblocks = pow(2, level); // pow since we are using diagonal based admis.
    // std::cout << "nblocks: " << nblocks << " level: " << level << std::endl;
    level_blocks.push_back(nblocks);
    if (level == 0) { return; }
    if (level == height) {
      for (int64_t i = 0; i < nblocks; ++i) {
        for (int64_t j = 0; j < nblocks; ++j) {
          is_admissible.insert(i, j, level, std::abs(i - j) > admis);
        }
      }

      // is_admissible.erase(0, 2, level);
      // is_admissible.insert(0, 2, level, true);

      // is_admissible.erase(2, 0, level);
      // is_admissible.insert(2, 0, level, true);

      // is_admissible.erase(1, 3, level);
      // is_admissible.insert(1, 3, level, true);

      // is_admissible.erase(3, 1, level);
      // is_admissible.insert(3, 1, level, true);

    }
    else {
      coarsen_blocks(level);
    }

    calc_diagonal_based_admissibility(level-1);
  }


  Matrix
  H2::generate_column_block(int64_t block, int64_t block_size, const Domain& domain,
                            int64_t level) {
    Matrix AY(block_size, 0);
    for (int64_t j = 0; j < level_blocks[level]; ++j) {
      if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) { continue; }
      Hatrix::Matrix dense = Hatrix::generate_p2p_interactions(domain, block, j, level, height);
      AY = concat(AY, dense, 1);
    }

    return AY;
  }

  std::tuple<Matrix, Matrix>
  H2::generate_column_bases(int64_t block, int64_t block_size, const Domain& domain,
                            std::vector<Matrix>& Y, int64_t level) {
    // Row slice since column bases should be cutting across the columns.
    Matrix AY = generate_column_block(block, block_size, domain, level);
    Matrix Ui, Si, Vi; double error;
    std::tie(Ui, Si, Vi, error) = truncated_svd(AY, rank);

    return {std::move(Ui), std::move(Si)};
  }

  Matrix
  H2::generate_row_block(int64_t block, int64_t block_size, const Domain& domain, int64_t level) {
    Hatrix::Matrix YtA(0, block_size);
    for (int64_t i = 0; i < level_blocks[level]; ++i) {
      if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) { continue; }
      Hatrix::Matrix dense = Hatrix::generate_p2p_interactions(domain, i, block, level, height);
      YtA = concat(YtA, dense, 0);
    }

    return YtA;
  }


  std::tuple<Matrix, Matrix>
  H2::generate_row_bases(int64_t block, int64_t block_size, const Domain& domain,
                         std::vector<Matrix>& Y, int64_t level) {
    Matrix YtA = generate_row_block(block, block_size, domain, level);

    Matrix Ui, Si, Vi; double error;
    std::tie(Ui, Si, Vi, error) = Hatrix::truncated_svd(YtA, rank);

    return {std::move(Si), transpose(Vi)};
  }

  void
  H2::generate_leaf_nodes(const Domain& domain) {
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

    // Generate V leaf blocks
    for (int64_t j = 0; j < nblocks; ++j) {
      Matrix Stemp, Vtemp;
      std::tie(Stemp, Vtemp) =
        generate_row_bases(j, domain.boxes[j].num_particles, domain, Y, height);
      V.insert(j, height, std::move(Vtemp));
      Srow.insert(j, height, std::move(Stemp));
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

  Matrix
  H2::get_Ubig(int64_t node, int64_t level) {
    if (level == height) {
      return U(node, level);
    }

    int64_t child1 = node * 2;
    int64_t child2 = node * 2 + 1;

    Matrix Ubig_child1 = get_Ubig(child1, level+1);
    Matrix Ubig_child2 = get_Ubig(child2, level+1);

    int64_t block_size = Ubig_child1.rows + Ubig_child2.rows;

    Matrix Ubig(block_size, rank);

    std::vector<Matrix> Ubig_splits = Ubig.split(
                                                 std::vector<int64_t>(1,
                                                                      Ubig_child1.rows),
                                                 {});

    std::vector<Matrix> U_splits = U(node, level).split(2, 1);

    matmul(Ubig_child1, U_splits[0], Ubig_splits[0]);
    matmul(Ubig_child2, U_splits[1], Ubig_splits[1]);

    return Ubig;

  }

  Matrix
  H2::get_Vbig(int64_t node, int64_t level) {
    if (level == height) {
      return V(node, height);
    }

    int64_t child1 = node * 2;
    int64_t child2 = node * 2 + 1;

    Matrix Vbig_child1 = get_Vbig(child1, level+1);
    Matrix Vbig_child2 = get_Vbig(child2, level+1);

    int64_t block_size = Vbig_child1.rows + Vbig_child2.rows;

    Matrix Vbig(block_size, rank);

    std::vector<Matrix> Vbig_splits = Vbig.split(std::vector<int64_t>(1, Vbig_child1.rows), {});
    std::vector<Matrix> V_splits = V(node, level).split(2, 1);

    matmul(Vbig_child1, V_splits[0], Vbig_splits[0]);
    matmul(Vbig_child2, V_splits[1], Vbig_splits[1]);

    return Vbig;
  }

  int64_t
  H2::get_block_size_row(const Domain& domain, int64_t parent, int64_t level) {
    if (level == height) {
      return domain.boxes[parent].num_particles;
    }
    int64_t child_level = level + 1;
    int64_t child1 = parent * 2;
    int64_t child2 = parent * 2 + 1;

    return get_block_size_row(domain, child1, child_level) +
      get_block_size_row(domain, child2, child_level);
  }

  int64_t
  H2::get_block_size_col(const Domain& domain, int64_t parent, int64_t level) {
    if (level == height) {
      return domain.boxes[parent].num_particles;
    }
    int64_t child_level = level + 1;
    int64_t child1 = parent * 2;
    int64_t child2 = parent * 2 + 1;

    return get_block_size_col(domain, child1, child_level) +
      get_block_size_col(domain, child2, child_level);
  }

  bool
  H2::row_has_admissible_blocks(int64_t row, int64_t level) {
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

  bool
  H2::col_has_admissible_blocks(int64_t col, int64_t level) {
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

  std::tuple<Matrix, Matrix>
  H2::generate_U_transfer_matrix(Matrix& Ubig_child1, Matrix& Ubig_child2, int64_t node,
                                 int64_t block_size, const Domain& domain, int64_t level) {
    Matrix col_block = generate_column_block(node, block_size, domain, level);
    auto col_block_splits = col_block.split(2, 1);

    Matrix temp(Ubig_child1.cols + Ubig_child2.cols, col_block.cols);
    auto temp_splits = temp.split(2, 1);

    matmul(Ubig_child1, col_block_splits[0], temp_splits[0], true, false, 1, 0);
    matmul(Ubig_child2, col_block_splits[1], temp_splits[1], true, false, 1, 0);

    Matrix Utransfer, Si, Vi; double error;
    std::tie(Utransfer, Si, Vi, error) = truncated_svd(temp, rank);

    return {std::move(Utransfer), std::move(Si)};
  }

  std::tuple<Matrix, Matrix>
  H2::generate_V_transfer_matrix(Matrix& Vbig_child1, Matrix& Vbig_child2, int64_t node,
                                 int64_t block_size, const Domain& domain, int64_t level) {
    Matrix row_block = generate_row_block(node, block_size, domain, level);
    auto row_block_splits = row_block.split(1, 2);

    Matrix temp(row_block.rows, Vbig_child1.cols + Vbig_child2.cols);
    auto temp_splits = temp.split(1, 2);

    matmul(row_block_splits[0], Vbig_child1, temp_splits[0]);
    matmul(row_block_splits[1], Vbig_child2, temp_splits[1]);

    Matrix Ui, Si, Vtransfer; double error;
    std::tie(Ui, Si, Vtransfer, error) = truncated_svd(temp, rank);

    return {std::move(Si), transpose(Vtransfer)};
  }

  std::tuple<RowLevelMap, ColLevelMap>
  H2::generate_transfer_matrices(const Domain& domain,
                                 int64_t level, RowLevelMap& Uchild,
                                 ColLevelMap& Vchild) {
    int64_t nblocks = level_blocks[level];

    std::vector<Matrix> Y;
    // Generate the actual bases for the upper level and pass it to this
    // function again for generating transfer matrices at successive levels.
    RowLevelMap Ubig_parent;
    ColLevelMap Vbig_parent;

    for (int64_t i = 0; i < nblocks; ++i) {
      int64_t block_size = get_block_size_row(domain, i, level);
      Y.push_back(generate_random_matrix(block_size, rank + oversampling));
    }

    for (int64_t node = 0; node < nblocks; ++node) {
      int64_t child1 = node * 2;
      int64_t child2 = node * 2 + 1;
      int64_t child_level = level + 1;
      int64_t block_size = get_block_size_row(domain, node, level);

      if (row_has_admissible_blocks(node, level) && height != 1) {
        Matrix& Ubig_child1 = Uchild(child1, child_level);
        Matrix& Ubig_child2 = Uchild(child2, child_level);

        Matrix Utransfer, Stemp;
        std::tie(Utransfer, Stemp) = generate_U_transfer_matrix(Ubig_child1,
                                                                Ubig_child2,
                                                                node,
                                                                block_size,
                                                                domain,
                                                                level);

        // std::cout << "level: " << level << " node: " << node
        //           << " U : " << norm(generate_identity_matrix(rank, rank) - matmul(Utransfer, Utransfer, true, false))
        //           << std::endl;

        U.insert(node, level, std::move(Utransfer));
        Scol.insert(node, level, std::move(Stemp));

        // Generate the full bases to pass onto the parent.
        auto Utransfer_splits = U(node, level).split(2, 1);
        Matrix Ubig(block_size, rank);
        auto Ubig_splits = Ubig.split(2, 1);

        matmul(Ubig_child1, Utransfer_splits[0], Ubig_splits[0]);
        matmul(Ubig_child2, Utransfer_splits[1], Ubig_splits[1]);

        Ubig_parent.insert(node, level, std::move(Ubig));
      }

      if (col_has_admissible_blocks(node, level) && height != 1) {
        // Generate V transfer matrix.
        Matrix& Vbig_child1 = Vchild(child1, child_level);
        Matrix& Vbig_child2 = Vchild(child2, child_level);

        Matrix Vtransfer, Stemp;
        std::tie(Stemp, Vtransfer) = generate_V_transfer_matrix(Vbig_child1,
                                                                Vbig_child2,
                                                                node,
                                                                block_size,
                                                                domain,
                                                                level);
        V.insert(node, level, std::move(Vtransfer));
        Srow.insert(node, level, std::move(Stemp));

        // Generate the full bases for passing onto the upper level.
        std::vector<Matrix> Vtransfer_splits = V(node, level).split(2, 1);
        Matrix Vbig(rank, block_size);
        std::vector<Matrix> Vbig_splits = Vbig.split(1, 2);

        matmul(Vtransfer_splits[0], Vbig_child1, Vbig_splits[0], true, true, 1, 0);
        matmul(Vtransfer_splits[1], Vbig_child2, Vbig_splits[1], true, true, 1, 0);

        Vbig_parent.insert(node, level, transpose(Vbig));
      }
    }

    for (int64_t row = 0; row < nblocks; ++row) {
      for (int64_t col = 0; col < nblocks; ++col) {
        if (is_admissible.exists(row, col, level) && is_admissible(row, col, level)) {
          int64_t row_block_size = get_block_size_row(domain, row, level);
          int64_t col_block_size = get_block_size_col(domain, col, level);

          Matrix D = generate_p2p_interactions(domain, row, col, level, height);

          S.insert(row, col, level, matmul(matmul(Ubig_parent(row, level), D, true, false),
                                           Vbig_parent(col, level)));
        }
      }
    }

    return {Ubig_parent, Vbig_parent};
  }

  H2::H2(const Domain& domain, int64_t _N, int64_t _rank, int64_t _nleaf,
         double _admis, std::string& admis_kind, int64_t matrix_type) :
    N(_N), rank(_rank), nleaf(_nleaf), admis(_admis), admis_kind(admis_kind),
    matrix_type(matrix_type) {
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
    PV = 1e-3 * (1 / pow(10, height));

    int64_t all_dense_row = find_all_dense_row();
    if (all_dense_row != -1) {
      std::cout << "found all dense row at " << all_dense_row << ". Aborting.\n";
      abort();
    }

    generate_leaf_nodes(domain);
    RowLevelMap Uchild = U;
    ColLevelMap Vchild = V;

    for (int64_t level = height-1; level > 0; --level) {
      std::tie(Uchild, Vchild) = generate_transfer_matrices(domain, level, Uchild, Vchild);
    }
  }

  double
  H2::construction_relative_error(const Domain& domain) {
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

            // std::cout << "row: " << row << " col: " << col
            //           << " level: " << level
            //           << " error: " << error << std::endl;
          }
        }
      }
    }

    return std::sqrt(error / dense_norm);
  }

  void
  H2::actually_print_structure(int64_t level) {
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

  double
  H2::low_rank_block_ratio() {
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
}

Hatrix::Matrix lower(Hatrix::Matrix A) {
  Hatrix::Matrix mat(A.rows, A.cols);

  for (int64_t i = 0; i < A.rows; ++i) {
    mat(i, i) = 1.0;
    for (int64_t j = 0; j < i; ++j) {
      mat(i, j) = A(i, j);
    }
  }

  return mat;
}

Hatrix::Matrix upper(Hatrix::Matrix A) {
  Hatrix::Matrix mat(A.rows, A.cols);

  for (int64_t i = 0; i < A.rows; ++i) {
    for (int64_t j = i; j < A.cols; ++j) {
      mat(i, j) = A(i, j);
    }
  }

  return mat;
}

Hatrix::Matrix generate_L1(Hatrix::H2& A) {
  Matrix L1 = generate_identity_matrix(A.rank * 8, A.rank * 8);
  auto L1_splits = L1.split(8, 8);
  int64_t level = 1;
  int64_t num_nodes = A.level_blocks[level];

  for (int64_t i = 0; i < num_nodes; ++i) {
    for (int64_t j = 0; j <= i; ++j) {
      std::vector<int64_t> row_children({i * 2 + 4, i * 2 + 1 + 4});
      std::vector<int64_t> col_children({j * 2 + 4, j * 2 + 1 + 4});

      auto D_split = A.D(i, j, level).split(2, 2);

      if (i == j) {
        for (int64_t c1 = 0; c1 < 2; ++c1) {
          for (int64_t c2 = 0; c2 <= c1; ++c2) {
            if (c1 == c2) {
              L1_splits[row_children[c1] * 8 + col_children[c2]] = lower(D_split[c1 * 2 + c2]);
            }
            else {
              L1_splits[row_children[c1] * 8 + col_children[c2]] = D_split[c1 * 2 + c2];
            }
          }
        }
      }
      else {
        for (int64_t c1 = 0; c1 < 2; ++c1) {
          for (int64_t c2 = 0; c2 < 2; ++c2) {
            L1_splits[row_children[c1] * 8 + col_children[c2]] = D_split[c1 * 2 + c2];
          }
        }
      }
    }
  }

  return L1;
}

Hatrix::Matrix generate_U1(Hatrix::H2& A) {
  Matrix U1 = generate_identity_matrix(A.rank * 8, A.rank * 8);
  auto U1_splits = U1.split(8, 8);
  int64_t level = 1;
  int64_t num_nodes = A.level_blocks[level];

  for (int64_t i = 0; i < num_nodes; ++i) {
    for (int64_t j = i; j < num_nodes; ++j) {
      std::vector<int64_t> row_children({i * 2 + 4, i * 2 + 1 + 4});
      std::vector<int64_t> col_children({j * 2 + 4, j * 2 + 1 + 4});

      auto D_split = A.D(i, j, level).split(2, 2);

      if (i == j) {
        for (int64_t c1 = 0; c1 < 2; ++c1) {
          for (int64_t c2 = c1; c2 < 2; ++c2) {
            if (c1 == c2) {
              U1_splits[row_children[c1] * 8 + col_children[c2]] = upper(D_split[c1 * 2 + c2]);
            }
            else {
              U1_splits[row_children[c1] * 8 + col_children[c2]] = D_split[c1 * 2 + c2];
            }
          }
        }
      }
      else {
        for (int64_t c1 = 0; c1 < 2; ++c1) {
          for (int64_t c2 = 0; c2 < 2; ++c2) {
            U1_splits[row_children[c1] * 8 + col_children[c2]] = D_split[c1 * 2 + c2];
          }
        }
      }
    }
  }

  return U1;
}

void verify_A1_solve(Matrix& A1, H2& A, const Domain& domain) {
  Matrix b = generate_random_matrix(A.rank * 4, 1);
  auto A1_22 = Matrix(A1.split(2, 2)[3]);
  // A1_22.print();              //
  auto x1_solve = lu_solve(A1_22, b);

  auto x1_h2 = A.solve(b, 1);

  std::cout << "A1 solve error: " << norm(x1_h2 - x1_solve) / norm(x1_solve) << std::endl;
}

void verify_A1_factorization(Hatrix::H2& A, const Domain& domain) {
  Matrix L1 = generate_L1(A);
  Matrix U1 = generate_U1(A);
  Matrix A1_actual = matmul(L1, U1);

  Matrix diff = A1_actual - A1_expected;
  int64_t nblocks = 4;
  auto d_splits = diff.split(nblocks, nblocks);
  auto m_splits = A1_expected.split(nblocks, nblocks);

  std::cout << "A1 factorization rel error: " << norm(diff) / norm(A1_expected) << std::endl;

  verify_A1_solve(A1_actual, A, domain);
}

// Generates UF chain for the A2 matrix.
std::vector<Hatrix::Matrix> generate_UF_chain(Hatrix::H2& A) {
  std::vector<Hatrix::Matrix> U_F;
  int64_t level = 2;

  int64_t num_nodes = pow(2, level);
  for (int64_t block = 0; block < num_nodes; ++block) {
    Matrix UF_full = generate_identity_matrix(A.rank * 8, A.rank * 8);
    auto UF_full_splits = UF_full.split(8, 8);

    if (A.U.exists(block, level)) {
      int64_t block_size = A.U(block, level).rows;
      Matrix UF_block = make_complement(A.U(block, level));

      auto UF_block_splits = SPLIT_DENSE(UF_block, block_size - A.U(block, level).cols,
                                         block_size - A.U(block, level).cols);

      int64_t permuted_nblocks = 8;

      UF_full_splits[block * permuted_nblocks + block] = UF_block_splits[0];
      UF_full_splits[(block + num_nodes) * permuted_nblocks + block] = UF_block_splits[2];
      UF_full_splits[block * permuted_nblocks + (block + num_nodes)] = UF_block_splits[1];
      UF_full_splits[(block + num_nodes) * permuted_nblocks + block + num_nodes] =
        UF_block_splits[3];
    }

    U_F.push_back(UF_full);
  }

  return U_F;
}

std::vector<Hatrix::Matrix> generate_VF_chain(Hatrix::H2& A) {
  std::vector<Hatrix::Matrix> V_F;
  int64_t level = 2;

  int64_t num_nodes = pow(2, level);
  for (int64_t block = 0; block < num_nodes; ++block) {
    Matrix VF_full = generate_identity_matrix(A.rank * 8, A.rank * 8);
    auto VF_full_splits = VF_full.split(8, 8);

    if (A.V.exists(block, level)) {
      int64_t block_size = A.V(block, level).rows;
      Matrix VF_block = make_complement(A.V(block, level));

      auto VF_block_splits = SPLIT_DENSE(VF_block, block_size - A.V(block, level).cols,
                                         block_size - A.V(block, level).cols);

      int64_t permuted_nblocks = 8;

      VF_full_splits[block * permuted_nblocks + block] = VF_block_splits[0];
      VF_full_splits[(block + num_nodes) * permuted_nblocks + block] = VF_block_splits[2];
      VF_full_splits[block * permuted_nblocks + (block + num_nodes)] = VF_block_splits[1];
      VF_full_splits[(block + num_nodes) * permuted_nblocks + block + num_nodes] =
        VF_block_splits[3];
    }

    V_F.push_back(VF_full);
  }

  return V_F;

}

std::vector<Hatrix::Matrix> generate_L2_chain(Hatrix::H2& A) {
  std::vector<Hatrix::Matrix> L2;
  int64_t block_size = A.rank;
  int64_t permuted_nblocks = 8;
  int64_t level = 2;
  int64_t nblocks = A.level_blocks[level];

  for (int64_t block = 0; block < nblocks; ++block) {
    Matrix L_block = generate_identity_matrix(A.rank * 8, A.rank * 8);
    auto L_block_splits = L_block.split(8, 8);

    for (int64_t j = 0; j <= block; ++j) {
      if (A.is_admissible.exists(block, j, level) && !A.is_admissible(block, j, level)) {
        auto D_splits = A.D(block, j, level).split(2, 2);

        // Copy the cc parts
        if (block == j) {
          L_block_splits[block * permuted_nblocks + j] = lower(D_splits[0]);
        }
        else {
          L_block_splits[block * permuted_nblocks + j] = D_splits[0];
        }

        L_block_splits[(block + nblocks) * permuted_nblocks + j] = D_splits[2];
      }
    }

    // Copy oc parts belonging to the 'upper' parts of the matrix
    for (int64_t i = 0; i < block; ++i) {
      if (A.is_admissible.exists(i, block, level) && !A.is_admissible(i, block, level)) {
        auto D_splits = A.D(i, block, level).split(2, 2);
        L_block_splits[(i + nblocks) * permuted_nblocks + block] = D_splits[2];
      }
    }

    L2.push_back(L_block);
  }

  return L2;
}

std::vector<Hatrix::Matrix> generate_U2_chain(Hatrix::H2& A) {
  std::vector<Hatrix::Matrix> U2;
  int64_t block_size = A.rank;
  int64_t permuted_nblocks = 8;
  int64_t level = 2;
  int64_t nblocks = A.level_blocks[level];

  for (int64_t block = 0; block < nblocks; ++block) {
    Matrix U_block = generate_identity_matrix(A.rank * 8, A.rank * 8);
    auto U_splits = U_block.split(8, 8);

    for (int64_t i = 0; i <= block; ++i) {
      if (A.is_admissible.exists(i, block, level) && !A.is_admissible(i, block, level)) {
        auto D_splits = A.D(i, block, level).split(2, 2);

        // Copy the cc blocks
        if (block == i) {
          U_splits[i * permuted_nblocks + block] = upper(D_splits[0]);
        }
        else {
          U_splits[i * permuted_nblocks + block] = D_splits[0];
        }

        // Copy the co parts
        U_splits[i * permuted_nblocks + block + nblocks] = D_splits[1];
      }
    }

    for (int64_t j = 0; j < block; ++j) {
      if (A.is_admissible.exists(block, j, level) && !A.is_admissible(block, j, level)) {
        auto D_splits = A.D(block, j, level).split(2, 2);
        U_splits[block * permuted_nblocks + (j + nblocks)] = D_splits[1];
      }
    }

    U2.push_back(U_block);
  }

  return U2;
}

Matrix unpermute_matrix(Matrix PA, H2& A) {
  Matrix M(A.rank * 8, A.rank * 8);

  int64_t level = 2;
  int64_t block_size = A.rank * 2;
  int64_t permuted_nblocks = 8;
  std::vector<int64_t> row_offsets, col_offsets;
  int64_t num_nodes = A.level_blocks[level];

  auto PA_splits = PA.split(8, 8);
  auto M_splits = M.split(4, 4);

  for (int64_t i = 0; i < num_nodes; ++i) {
    for (int64_t j = 0; j < num_nodes; ++j) {
      Matrix block(block_size, block_size);
      auto block_splits = SPLIT_DENSE(block,
                                      block_size - A.rank,
                                      block_size - A.rank);

      block_splits[0] = PA_splits[(i) * permuted_nblocks + j];
      block_splits[1] = PA_splits[i * permuted_nblocks + j + num_nodes];
      block_splits[2] = PA_splits[(i + num_nodes) * permuted_nblocks + j];
      block_splits[3] = PA_splits[(i + num_nodes) * permuted_nblocks + j + num_nodes];

      M_splits[i * num_nodes + j] = block;
    }
  }

  return M;
}



void verify_A2_solve(Matrix& A2, H2& A, const Domain& domain) {
  Matrix b = generate_random_matrix(A.rank * 8, 1);

  auto x2_dense = lu_solve(A2, b);
  auto x2_h2 = A.solve(b, 2);

  // b.print();

  // (x2_h2).print();
  // x2_dense.print();

  std::cout << "A2 solve error: " << norm(x2_h2 - x2_dense) / norm(x2_dense) << std::endl;
}


void verify_A2_factorization(Hatrix::H2& A, const Domain& domain) {
  auto UF = generate_UF_chain(A);
  auto VF = generate_VF_chain(A);
  auto L2 = generate_L2_chain(A);
  auto U2 = generate_U2_chain(A);
  Hatrix::Matrix L1 = generate_L1(A);
  Hatrix::Matrix U1 = generate_U1(A);

  auto product = generate_identity_matrix(A.rank * 8, A.rank * 8);

  for (int64_t i = 0; i < 4; ++i) {
    product = matmul(product, UF[i]);
    product = matmul(product, L2[i]);
  }

  product = matmul(product, L1);
  product = matmul(product, U1);

  for (int64_t i = 3; i >= 0; --i) {
    product = matmul(product, U2[i]);
    product = matmul(product, VF[i], false, true);
  }

  auto A2_actual = unpermute_matrix(product, A);

  auto diff = (A2_expected - A2_actual);

  auto diff_splits = diff.split(4, 4);
  auto A2_expected_splits = A2_expected.split(4, 4);
  auto A2_act_splits = A2_actual.split(4,4);

  std::cout << "A2 factorization error: "
            <<  norm(A2_expected - A2_actual) / norm(A2_expected) << std::endl;

  std::cout << "A2 block wise factorization error: \n";

  double err = 0;

  for (int64_t i = 0; i < 4; ++i) {
    for (int64_t j = 0; j < 4; ++j) {
      std::cout << "<i, j>: " << i << ", " << j
                << " -- "
                << std::setprecision(8)
                << norm(diff_splits[i * 4 + j]) / norm(A2_expected_splits[i * 4 + j])
                << "   ";
    }
    std::cout << std::endl;
  }

  verify_A2_solve(A2_actual, A, domain);
}

std::tuple<std::vector<int64_t>, std::vector<int64_t> >
generate_offsets(const Hatrix::H2& A, int64_t nblocks) {
  int64_t level = 1;
  std::vector<int64_t> row_offsets, col_offsets;
  int64_t c_size_offset_rows = 0, c_size_offset_cols = 0;
  for (int i = 0; i < nblocks; ++i) {
    int64_t row_split = A.D(i, i, level).rows - A.rank;
    int64_t col_split = A.D(i, i, level).cols - A.rank;

    row_offsets.push_back(c_size_offset_rows + row_split);
    col_offsets.push_back(c_size_offset_cols + col_split);

    c_size_offset_rows += row_split;
    c_size_offset_cols += col_split;
  }

  int64_t row_rank_offset = 0, col_rank_offset = 0;
  for (int i = 0; i < nblocks-1; ++i) {
    row_offsets.push_back(c_size_offset_rows + row_rank_offset + A.rank);
    col_offsets.push_back(c_size_offset_cols + col_rank_offset + A.rank);

    row_rank_offset += A.rank;
    col_rank_offset += A.rank;
  }

  return {row_offsets, col_offsets};
}

Matrix
generate_UF(const Hatrix::H2& A, int64_t block, int64_t nblocks) {
  std::vector<int64_t> row_offsets, col_offsets;
  std::tie(row_offsets, col_offsets) = generate_offsets(A, nblocks);

  Matrix UF_full = generate_identity_matrix(A.N, A.N);
  Matrix UF_block = make_complement(A.U(block, A.height));

  auto UF_full_splits = UF_full.split(row_offsets, col_offsets);
  auto UF_block_splits = SPLIT_DENSE(UF_block,
                                     A.D(block, block, A.height).rows - A.rank,
                                     A.D(block, block, A.height).cols - A.rank);

  int64_t permuted_nblocks = nblocks * 2;

  UF_full_splits[block * permuted_nblocks + block] = UF_block_splits[0];
  UF_full_splits[(block + nblocks) * permuted_nblocks + block] = UF_block_splits[2];
  UF_full_splits[block * permuted_nblocks + block + nblocks] = UF_block_splits[1];
  UF_full_splits[(block + nblocks) * permuted_nblocks + block + nblocks] = UF_block_splits[3];

  return UF_full;
}

Matrix generate_L2(const Hatrix::H2& A, int64_t block, int64_t nblocks) {
  int64_t permuted_nblocks = nblocks * 2;
  std::vector<int64_t> row_offsets, col_offsets;
  std::tie(row_offsets, col_offsets) = generate_offsets(A, nblocks);

  Matrix L2_block = generate_identity_matrix(A.N, A.N);
  auto L2_splits = L2_block.split(row_offsets, col_offsets);

  for (int j = 0; j <= block; ++j) {
    if (!A.is_admissible(block, j, A.height)) {
      int64_t row_split = A.D(block, j, A.height).rows - A.rank;
      int64_t col_split = A.D(block, j, A.height).cols - A.rank;
      auto D_splits = SPLIT_DENSE(A.D(block, j, A.height), row_split, col_split);

      // Copy the cc parts
      if (block == j) {
        L2_splits[block * permuted_nblocks + j] = lower(D_splits[0]);
      }
      else {
        L2_splits[block * permuted_nblocks + j] = D_splits[0];
      }

      L2_splits[(block + nblocks) * permuted_nblocks + j] = D_splits[2];
    }
  }

  // Copy oc parts belonging to the 'upper' parts of the matrix
  for (int i = 0; i < block; ++i) {
    if (!A.is_admissible(i, block, A.height)) {
      int64_t row_split = A.D(i, block, A.height).rows - A.rank;
      int64_t col_split = A.D(i, block, A.height).cols - A.rank;
      auto D_splits = A.D(i, block, A.height).split(std::vector<int64_t>(1, row_split),
                                                    std::vector<int64_t>(1, col_split));
      L2_splits[(i + nblocks) * permuted_nblocks + block] = D_splits[2];
    }
  }

  return L2_block;
}

std::tuple<std::vector<int64_t>, std::vector<int64_t> >
generate_rank_offsets(const Hatrix::H2& A, int64_t nblocks) {
  std::vector<int64_t> row_rank_splits, col_rank_splits;
  int64_t nrows = 0, ncols = 0;
  for (int i = 0; i < nblocks; ++i) {
    row_rank_splits.push_back(nrows + A.rank);
    col_rank_splits.push_back(ncols + A.rank);
    nrows += A.rank;
    ncols += A.rank;
  }

  return {row_rank_splits, col_rank_splits};
}

Matrix generate_L0_permuted(const Hatrix::H2& A, int64_t nblocks) {
  int64_t permuted_nblocks = nblocks * 2;
  int64_t level = 1;
  std::vector<int64_t> row_offsets, col_offsets;
  std::tie(row_offsets, col_offsets) = generate_offsets(A, nblocks);

  Matrix L0 = generate_identity_matrix(A.N, A.N);
  auto L0_splits = L0.split(row_offsets, col_offsets);

  std::vector<int64_t> row_rank_splits, col_rank_splits;
  std::tie(row_rank_splits, col_rank_splits)= generate_rank_offsets(A, nblocks);

  auto last_splits = A.D(0, 0, 0).split(row_rank_splits, col_rank_splits);

  for (int i = 0; i < nblocks; ++i) {
    for (int j = 0; j <= i; ++j) {
      if (!A.is_admissible(i, j, A.height)) {
        int64_t row_split = A.D(i, j, A.height).rows - A.rank;
        int64_t col_split = A.D(i, j, A.height).cols - A.rank;
        auto D_splits = A.D(i, j, A.height).split(std::vector<int64_t>(1, row_split),
                                                  std::vector<int64_t>(1, col_split));

        // Copy the oo parts
        if (i == j) {
          L0_splits[(i + nblocks) * permuted_nblocks + (j + nblocks)] =
            lower(last_splits[i * nblocks + j]);
        }
        else {
          L0_splits[(i + nblocks) * permuted_nblocks + (j + nblocks)] =
            last_splits[i * nblocks + j];
        }
      }
      else {
        // Copy S blocks into the lower right corner
        L0_splits[(i + nblocks) * permuted_nblocks + (j + nblocks)] =
          last_splits[i * nblocks + j];
      }
    }
  }

  return L0;
}


Matrix generate_U0_permuted(const Hatrix::H2& A, int64_t nblocks) {
  int64_t permuted_nblocks = nblocks * 2;
  int64_t level = 1;
  std::vector<int64_t> row_offsets, col_offsets;
  std::tie(row_offsets, col_offsets) = generate_offsets(A, nblocks);

  Matrix U0 = generate_identity_matrix(A.N, A.N);
  auto U0_splits = U0.split(row_offsets, col_offsets);

  std::vector<int64_t> row_rank_splits, col_rank_splits;
  std::tie(row_rank_splits, col_rank_splits)= generate_rank_offsets(A, nblocks);
  auto last_splits = A.D(0, 0, 0).split(row_rank_splits, col_rank_splits);

  for (int i = 0; i < nblocks; ++i) {
    for (int j = i; j < nblocks; ++j) {
      if (!A.is_admissible(i, j, A.height)) {
        int64_t row_split = A.D(i, j,  A.height).rows - A.rank;
        int64_t col_split = A.D(i, j, A.height).cols - A.rank;
        auto D_splits = A.D(i, j, A.height).split(std::vector<int64_t>(1, row_split),
                                                  std::vector<int64_t>(1, col_split));

        // Copy the oo parts
        if (i == j) {
          U0_splits[(i + nblocks) * permuted_nblocks + (j + nblocks)] =
            upper(last_splits[i * nblocks + j]);
        }
        else {
          U0_splits[(i + nblocks) * permuted_nblocks + (j + nblocks)] =
            last_splits[i * nblocks + j];
        }
      }
      else {
        // Copy S blocks
        U0_splits[(i + nblocks) * permuted_nblocks + (j + nblocks)] =
          last_splits[i * nblocks + j];
      }
    }
  }

  return U0;
}

Matrix generate_U2(const Hatrix::H2& A, int64_t block, int64_t nblocks) {
  int64_t level = 1;
  int64_t permuted_nblocks = nblocks * 2;
  std::vector<int64_t> row_offsets, col_offsets;
  std::tie(row_offsets, col_offsets) = generate_offsets(A, nblocks);
  std::vector<Matrix> U;

  Matrix U_block = generate_identity_matrix(A.N, A.N);
  auto U_splits = U_block.split(row_offsets, col_offsets);

  for (int i = 0; i <= block; ++i) {
    if (!A.is_admissible(i, block, A.height)) {
      int64_t row_split = A.D(i, block, A.height).rows - A.rank;
      int64_t col_split = A.D(i, block, A.height).cols - A.rank;
      auto D_splits = SPLIT_DENSE(A.D(i, block, A.height), row_split, col_split);

      // Copy the cc blocks
      if (block == i) {
        U_splits[i * permuted_nblocks + block] = upper(D_splits[0]);
      }
      else {
        U_splits[i * permuted_nblocks + block] = D_splits[0];
      }

      // Copy the co parts
      U_splits[i * permuted_nblocks + block + nblocks] = D_splits[1];
    }
  }

  for (int j = 0; j < block; ++j) {
    if (!A.is_admissible(block, j, A.height)) {
      int64_t row_split = A.D(block, j, A.height).rows - A.rank;
      int64_t col_split = A.D(block, j, A.height).cols - A.rank;
      auto D_splits = SPLIT_DENSE(A.D(block, j, A.height), row_split, col_split);
      U_splits[block * permuted_nblocks + (j + nblocks)] = D_splits[1];
    }
  }

  return U_block;
}

Matrix
generate_VF(const Hatrix::H2& A, int64_t block, int64_t nblocks) {
  std::vector<int64_t> row_offsets, col_offsets;
  std::tie(row_offsets, col_offsets) = generate_offsets(A, nblocks);

  Matrix VF_full = generate_identity_matrix(A.N, A.N);
  Matrix VF_block = make_complement(A.V(block, A.height));

  auto VF_full_splits = VF_full.split(row_offsets, col_offsets);
  auto VF_block_splits = SPLIT_DENSE(VF_block,
                                     A.D(block, block, A.height).rows - A.rank,
                                     A.D(block, block, A.height).cols - A.rank);
  int64_t permuted_nblocks = nblocks * 2;

  VF_full_splits[block * permuted_nblocks + block] = VF_block_splits[0];
  VF_full_splits[(block + nblocks) * permuted_nblocks + block] = VF_block_splits[2];
  VF_full_splits[block * permuted_nblocks + block + nblocks] = VF_block_splits[1];
  VF_full_splits[(block + nblocks) * permuted_nblocks + block + nblocks] = VF_block_splits[3];

  return VF_full;
}

Matrix unpermute_matrix(const Matrix& PA, const Hatrix::H2& A, int64_t nblocks) {
  Matrix M(A.N, A.N);
  int64_t permuted_nblocks = nblocks * 2;
  std::vector<int64_t> row_offsets, col_offsets;
  std::tie(row_offsets, col_offsets) = generate_offsets(A, nblocks);

  auto PA_splits = PA.split(row_offsets, col_offsets);

  std::vector<int64_t> M_row_offsets, M_col_offsets;
  int64_t rows = 0, cols = 0;
  for (int i = 0; i < nblocks; ++i) {
    M_row_offsets.push_back(rows + A.V(i, A.height).rows);
    M_col_offsets.push_back(cols + A.U(i, A.height).rows);

    rows += A.V(i, A.height).rows;
    cols += A.U(i, A.height).rows;
  }

  auto M_splits = M.split(M_row_offsets, M_col_offsets);

  for (int i = 0; i < nblocks; ++i) {
    for (int j = 0; j < nblocks; ++j) {
      Matrix block(A.V(i, A.height).rows, A.U(j, A.height).rows);
      auto block_splits = SPLIT_DENSE(block,
                                      A.V(i, A.height).rows - A.rank,
                                      A.U(j, A.height).rows - A.rank);

      block_splits[0] = PA_splits[(i) * permuted_nblocks + j];
      block_splits[1] = PA_splits[i * permuted_nblocks + j + nblocks];
      block_splits[2] = PA_splits[(i + nblocks) * permuted_nblocks + j];
      block_splits[3] = PA_splits[(i + nblocks) * permuted_nblocks + j + nblocks];

      M_splits[i * nblocks + j] = block;
    }
  }

  return M;

}

Matrix
regenerate_BLR2_matrix(const Hatrix::H2& A, const Hatrix::Domain& domain) {
  int64_t nblocks = domain.boxes.size();

  Matrix out = generate_identity_matrix(A.N, A.N);

  for (int64_t i = 0; i < nblocks; ++i) {
    out = matmul(out, generate_UF(A, i, nblocks));
    out = matmul(out, generate_L2(A, i, nblocks));
  }

  out = matmul(out, generate_L0_permuted(A, nblocks));
  out = matmul(out, generate_U0_permuted(A, nblocks));

  for (int64_t i = nblocks-1; i >= 0; --i) {
    out = matmul(out, generate_U2(A, i, nblocks));
    out = matmul(out, generate_VF(A, i, nblocks), false, true);
  }

  out = unpermute_matrix(out, A, nblocks);

  return out;
}


int main(int argc, char ** argv) {
  int64_t N = atoi(argv[1]);
  int64_t rank = atoi(argv[2]);
  int64_t nleaf = atoi(argv[3]);
  double admis = atof(argv[4]);
  int64_t ndim = atoi(argv[5]);
  std::string admis_kind(argv[6]);
  int64_t kernel_func = atoi(argv[7]);
  // 0 - BLR2
  // 1 - H2 matrix
  int64_t matrix_type = atoi(argv[8]);

  Hatrix::Context::init();

  Hatrix::Domain domain(N, ndim);

  switch(kernel_func) {
  case 0: {                     // laplace kernel
    domain.generate_particles(0.0, 1.0 * N);
    Hatrix::kernel_function = Hatrix::laplace_kernel;
    break;
  }
  case 1: {                     // sqrexp
    beta = 1;                   // supposed to be 0.1 in Cao's code.
    nu = 0.5;     //in matern, nu=0.5 exp (half smooth), nu=inf sqexp (inifinetly smooth)
    noise = 1.e-1;
    sigma = 1.0;
    domain.generate_starsh_grid_particles();
    Hatrix::kernel_function = Hatrix::sqrexp_kernel;
    break;
  }
  case 2: {                     // sin kernel 2D
    wave_k = 50;
    add_diag = N;               // hicma_parsec.c says set add_diag to N.
    domain.generate_starsh_grid_particles();
    Hatrix::kernel_function = Hatrix::block_sin;
    break;
  }
  }

  domain.divide_domain_and_create_particle_boxes(nleaf);

  Matrix rank_map = domain.generate_rank_heat_map();
  rank_map.print();

  Hatrix::H2 A(domain, N, rank, nleaf, admis, admis_kind, matrix_type);
  double construct_error, lr_ratio, solve_error;
  construct_error = A.construction_relative_error(domain);
  lr_ratio = A.low_rank_block_ratio();
  A.print_structure();
  A.factorize(domain);

  if  (false) {
    std::cout << "-- H2 verification --\n";
    verify_A1_factorization(A, domain);
    verify_A2_factorization(A, domain);
  }

  // Adense = dense before the compression.
  Hatrix::Matrix Adense = Hatrix::generate_p2p_matrix(domain);
  Adense.out_file("dense_matrix.data");

  if (false) {
    // regenA = permute(U * L * L0 * U0 * U * VF)
    Matrix regenA = regenerate_BLR2_matrix(A, domain);

    std::vector<int64_t> M_row_offsets, M_col_offsets;
    int64_t rows = 0, cols = 0, level = 1;

    int64_t nblocks = domain.boxes.size();

    for (int i = 0; i < nblocks; ++i) {
      M_row_offsets.push_back(rows + A.D(i, i, level).rows);
      M_col_offsets.push_back(cols + A.D(i, i, level).rows);

      rows += A.D(i, i, level).rows;
      cols += A.D(i, i, level).rows;
    }

    Matrix diff = (regenA - Adense);
    auto d_splits = diff.split(M_row_offsets, M_col_offsets);
    auto m_splits = Adense.split(M_row_offsets, M_col_offsets);
    auto regen_splits = regenA.split(M_row_offsets, M_col_offsets);

    // std::cout <<  "BLOCK WISE NORM:\n";
    // for (int64_t i = 0; i < nblocks; ++i) {
    //   for (int64_t j = 0; j < nblocks; ++j) {
    //     std::cout << "<i, j>: " << i << ", " << j
    //               << " -- "
    //               << std::setprecision(5)
    //               << norm(m_splits[i * nblocks + j])
    //               << std::setw(5)
    //               << std::endl;
    //   }
    // }

    std::cout << "ERROR\n";

    double tot_err = 0;
    for (int i = 0; i < nblocks; ++i) {
      for (int j = 0; j < nblocks; ++j) {
        double error = norm(d_splits[i * nblocks + j]) / norm(m_splits[i * nblocks + j]);

        if ((i != 3 && j != 5) || (i != 5 && j != 3)) {
          tot_err += norm(d_splits[i * nblocks + j]);
        }

        std::cout << "<i, j>: " << i << ", " << j
                  << " -- norm -> "
                  << std::setprecision(5)
                  << error
                  << std::setw(5)
                  << " diff -> " << norm(d_splits[i * nblocks + j])
                  << " m -> " << norm(m_splits[i * nblocks + j])
                  << std::endl;


      }
    }

    for (int i = 0; i < nblocks; ++i) {
      for (int j = 0; j < nblocks; ++j) {
        double error = norm(d_splits[i * nblocks + j]) / norm(m_splits[i * nblocks + j]);
        if (error > 1e-4) {
          std::cout << "i-> " << i << " j-> " << j << " large error: " << error << std::endl;;
        }

      }
    }

    std::cout << "block error: " << tot_err / norm(Adense)  << std::endl;
    std::cout << "factorization error = " << norm(diff) / norm(Adense) << std::endl;
    regenA.block_ranks(domain.boxes.size(), 1e-9).print();
  }

  Hatrix::Matrix b = Hatrix::generate_random_matrix(N, 1);
  Hatrix::Matrix x = A.solve(b, A.height);
  Hatrix::Matrix x_solve = lu_solve(Adense, b);
  solve_error = Hatrix::norm(x - x_solve) / Hatrix::norm(x_solve);

  auto x_splits = x.split(8, 1);
  auto x_solve_splits = x_solve.split(8, 1);

  // for (int i = 0; i < 8; ++i) {
  //   std::cout << "i -> " << i << " rel. err. -> "
  //             << norm(x_splits[i] - x_solve_splits[i]) / norm(x_solve_splits[i]) << std::endl;
  // }

  // (x - x_solve).print();

  Hatrix::Context::finalize();

  std::cout << "N=" << N << " admis=" << admis << " nleaf=" << nleaf << " ndim=" << ndim
            << " height= " << A.height << " rank=" << rank
            << " construct error= " << construct_error
            << " solve error= " << solve_error
            << " kernel func= " << kernel_func
            << " LR%= " << lr_ratio * 100 << "%"
            << " matrix type= " << (matrix_type == BLR2_MATRIX ? "BLR2" : "H2") << std::endl;

  std::ofstream file;

  file.open("result.txt", std::ios::app | std::ios::out);
  file << "N=" << N << " admis=" << admis << " nleaf=" << nleaf << " ndim=" << ndim
       << " height= " << A.height << " rank=" << rank
       << " construct error= " << construct_error
       << " solve error= " << solve_error
       << " kernel func= " << kernel_func
       << " LR%= " << lr_ratio * 100 << "%"
       << " matrix type= " << (matrix_type == BLR2_MATRIX ? "BLR2" : "H2") << std::endl;
  file.close();

}
