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

#include "Hatrix/Hatrix.h"

// Construction of BLR2 strong admis matrix based on geometry based admis condition.
double PV = 1e-3;

#define SPLIT_DENSE(dense, row_split, col_split)        \
  dense.split(std::vector<int64_t>(1, row_split),       \
              std::vector<int64_t>(1, col_split));


double beta, nu, noise = 1.e-1, sigma;

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
  std::function<double(const std::vector<double>& coords_row, const std::vector<double>& coords_col)> kernel_function;

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
    void print_file(std::string file_name);
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

  private:
    int64_t permute_forward(Matrix& x, const int64_t level, int64_t rank_offset);
    int64_t permute_backward(Matrix& x, const int64_t level, int64_t rank_offset);
    void solve_forward_level(Matrix& x_level, int level);
    void solve_backward_level(Matrix& x_level, int level);

    void generate_leaf_nodes(const Domain& domain);
    void actually_print_structure(int64_t level);
    bool row_has_admissible_blocks(int64_t row, int64_t level);
    bool col_has_admissible_blocks(int64_t col, int64_t level);
    Matrix generate_column_block(int64_t block, int64_t block_size, const Domain& domain, int64_t level);
    std::tuple<Matrix, Matrix>
    generate_column_bases(int64_t block, int64_t block_size, const Domain& domain,
                          std::vector<Matrix>& Y, int64_t level);
    Matrix generate_row_block(int64_t block, int64_t block_size, const Domain& domain, int64_t level);
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
    void factorize_level(int level, int nblocks, const Domain& domain,
                         RowMap& r, RowMap& t);
  public:
    H2(const Domain& domain, int64_t _N, int64_t _rank, int64_t _nleaf, double _admis,
       std::string& admis_kind);
    double construction_relative_error(const Domain& domain);
    void print_structure();
    double low_rank_block_ratio();
    void factorize(const Domain& domain);
    Matrix solve(const Matrix& b, int64_t _level);
  };
}

namespace Hatrix {
  double sqrexp_kernel(const std::vector<double>& coords_row,
                       const std::vector<double>& coords_col) {
    int64_t ndim = coords_row.size();
    double dist = 0, temp;

    // Copied from kernel_sqrexp.c in stars-H.

    for (int64_t k = 0; k < ndim; ++k) {
      temp = coords_row[k] - coords_col[k];
      dist += temp * temp;
    }
    dist = dist / beta;
    if (dist == 0) {
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
    for (int i = 0; i < leaf_rows.size(); ++i) { nrows += domain.boxes[leaf_rows[i]].num_particles; }
    for (int i = 0; i < leaf_cols.size(); ++i) { ncols += domain.boxes[leaf_cols[i]].num_particles; }

    Matrix out(nrows, ncols);

    std::vector<Particle> source_particles, target_particles;
    for (int i = 0; i < leaf_rows.size(); ++i) {
      int64_t source_box = leaf_rows[i];
      int64_t source = domain.boxes[source_box].start_index;
      for (int n = 0; n < domain.boxes[source_box].num_particles; ++n) {
        source_particles.push_back(domain.particles[source + n]);
      }
    }

    for (int i = 0; i < leaf_cols.size(); ++i) {
      int64_t target_box = leaf_cols[i];
      int64_t target = domain.boxes[target_box].start_index;
      for (int n = 0; n < domain.boxes[target_box].num_particles; ++n) {
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

  Matrix generate_p2p_matrix(const Domain& domain, int64_t rows, int64_t cols,
                             int64_t row_start, int64_t col_start) {
    Matrix out(rows, cols);

    for (int64_t i = 0; i < rows; ++i) {
      for (int64_t j = 0; j < cols; ++j) {
        out(i, j) = kernel_function(domain.particles[i + row_start].coords,
                                    domain.particles[j + col_start].coords);
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
    for (int k = 0; k < ndim; ++k) {
      dist += pow(b.center[k] - center[k], 2);
    }
    return std::sqrt(dist);
  }

  void
  Domain::print_file(std::string file_name) {
    std::vector<char> coords{'x', 'y', 'z'};

    std::ofstream file;
    file.open(file_name, std::ios::app | std::ios::out);
    for (int k = 0; k < ndim; ++k) {
      file << coords[k] << ",";
    }
    file << std::endl;

    for (int i = 0; i < N; ++i) {
      for (int k = 0; k < ndim; ++k) {
        file << particles[i].coords[k] << ",";
      }
      file << std::endl;
    }

    file.close();
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
        for (int k = 0; k < ndim; ++k) {
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
        for (int k = 0; k < ndim; ++k) {
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


  Domain::Domain(int64_t N, int64_t ndim) : N(N), ndim(ndim) {}

  void Domain::generate_starsh_grid_particles() {
    int64_t side = ceil(pow(N, 1.0 / ndim)); // size of each size of the grid.
    int64_t total = side;
    for (int i = 1; i < ndim; ++i) { total *= side; }

    int64_t ncoords = ndim * side;
    std::vector<double> coord(ncoords);

    for (int i = 0; i < side; ++i) {
      double val = double(i) / side;
      for (int j = 0; j < ndim; ++j) {
        coord[j * side + i] = val;
      }
    }

    std::vector<int64_t> pivot(ndim, 0);

    int64_t k = 0;
    for (int64_t i = 0; i < N; ++i) {
      std::vector<double> points(ndim);
      for (k = 0; k < ndim; ++k) {

        points[k] = coord[pivot[k] + k * side];
        // std::cout << points[k] << " ";
      }
      // std::cout << std::endl;
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
      std::random_device rd;  // Will be used to obtain a seed for the random number engine
      std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
      std::uniform_real_distribution<> dis(0.0, 2.0 * M_PI);
      double radius = 1.0;
      for (int64_t i = 0; i < N; ++i) {
        // double phi = dis(gen);


        double phi = (i * 2.0 * M_PI) / N;
        double theta = dis(gen);
        // double theta = (i * 2.0 * M_PI) / N;

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


  void H2::factorize_level(int level, int nblocks, const Domain& domain,
                           RowMap& r, RowMap& t) {
    RowColMap<Matrix> F;      // fill-in blocks.

    for (int block = 0; block < nblocks; ++block) {
      if (block > 0) {
        {
          // Scan for fill-ins in the same row as this diagonal block.
          int64_t block_size = D(block, block, level).rows;
          Matrix row_concat(block_size, 0);
          bool found_row_fill_in = false;
          for (int j = 0; j < nblocks; ++j) {
            if (F.exists(block, j)) {
              found_row_fill_in = true;
              break;
            }
          }

          if (found_row_fill_in) {
            row_concat = concat(row_concat, matmul(U(block, level),
                                                   Scol(block, level)), 1);
            for (int j = 0; j < nblocks; ++j) {
              if (F.exists(block, j)) {
                Matrix Fp = matmul(F(block, j), V(j, level), false, true);
                row_concat = concat(row_concat, Fp, 1);
              }
            }

            Matrix UN1, _SN1, _VN1T; double error;
            std::tie(UN1, _SN1, _VN1T, error) = truncated_svd(row_concat, rank);
            Scol.erase(block, level);
            Scol.insert(block, level, std::move(_SN1));

            Matrix r_block = matmul(UN1, U(block, level), true, false);

            for (int j = 0; j < nblocks; ++j) {
              if (is_admissible.exists(block, j, level) && is_admissible(block, j, level)) {
                Matrix Sbar_block_j = matmul(r_block, S(block, j, level));

                Matrix SpF(rank, rank);
                if (F.exists(block, j)) {
                  SpF = matmul(UN1, F(block, j), true, false);
                  Sbar_block_j = Sbar_block_j + SpF;
                }

                S.erase(block, j, level);
                S.insert(block, j, level, std::move(Sbar_block_j));
              }
            }
            U.erase(block, level);
            U.insert(block, level, std::move(UN1));
            r.insert(block, std::move(r_block));
          }
        }

        {
          // Scan for fill-ins in the same col as this diagonal block.
          int64_t block_size = D(block, block, level).cols;
          Matrix col_concat(0, block_size);
          std::vector<int64_t> UN2_row_splits;
          bool found_col_fill_in = false;
          for (int i = 0; i < nblocks; ++i) {
            if (F.exists(i, block)) {
              found_col_fill_in = true;
              break;
            }
          }

          if (found_col_fill_in) {
            col_concat = concat(col_concat, matmul(Srow(block, level),
                                                   transpose(V(block, level))), 0);
            for (int i = 0; i < nblocks; ++i) {
              if (F.exists(i, block)) {
                Matrix Fp = matmul(U(i, level), F(i, block));
                col_concat = concat(col_concat, Fp, 0);
              }
            }

            Matrix _UN2, _SN2, VN2T; double error;
            std::tie(_UN2, _SN2, VN2T, error) = truncated_svd(col_concat, rank);
            Srow.erase(block, level);
            Srow.insert(block, level, std::move(_SN2));

            Matrix t_block = matmul(V(block, level), VN2T, true, true);

            for (int i = 0; i < nblocks; ++i) {
              if (is_admissible.exists(i, block, level) && is_admissible(i, block, level)) {
                Matrix Sbar_i_block = matmul(S(i, block,level), t_block);
                if (F.exists(i, block)) {
                  Matrix SpF = matmul(F(i, block), VN2T, false, true);
                  Sbar_i_block = Sbar_i_block + SpF;
                }

                S.erase(i, block, level);
                S.insert(i, block, level, std::move(Sbar_i_block));
              }
            }

            V.erase(block, level);
            V.insert(block, level, transpose(VN2T));
            t.insert(block, std::move(t_block));
          }
        }
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
      for (int j = block + 1; j < nblocks; ++j) {
        if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) {
          int64_t col_split = D(block, j, level).cols - rank;
          auto D_splits = SPLIT_DENSE(D(block, j, level), row_split, col_split);
          solve_triangular(Dcc, D_splits[0], Hatrix::Left, Hatrix::Lower, true);
        }
      }

      // TRSM with co blocks on this row
      for (int j = 0; j < nblocks; ++j) {
        if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) {
          int64_t col_split = D(block, j, level).rows - rank;
          auto D_splits = SPLIT_DENSE(D(block, j, level), row_split, col_split);
          solve_triangular(Dcc, D_splits[1], Hatrix::Left, Hatrix::Lower, true);
        }
      }

      // TRSM with cc blocks on the column
      for (int i = block + 1; i < nblocks; ++i) {
        if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) {
          int64_t row_split = D(i, block, level).rows - rank;
          auto D_splits = SPLIT_DENSE(D(i, block, level), row_split, col_split);
          solve_triangular(Dcc, D_splits[0], Hatrix::Right, Hatrix::Upper, false);
        }
      }

      // TRSM with oc blocks on the column
      for (int i = 0; i < nblocks; ++i) {
        if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) {
          auto D_splits = SPLIT_DENSE(D(i, block, level),
                                      D(i, block, level).rows - rank,
                                      col_split);
          solve_triangular(Dcc, D_splits[2], Hatrix::Right, Hatrix::Upper, false);
        }
      }

      // Schur's compliment between cc blocks
      for (int i = block+1; i < nblocks; ++i) {
        for (int j = block+1; j < nblocks; ++j) {
          if ((is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) &&
              (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) &&
              (is_admissible.exists(i, j, level) && !is_admissible(i, j, level))) {
            auto lower_splits = SPLIT_DENSE(D(i, block, level),
                                            D(i, block, level).rows - rank,
                                            col_split);
            auto right_splits = SPLIT_DENSE(D(block, j, level),
                                            row_split,
                                            D(block, j, level).cols - rank);
            auto reduce_splits = SPLIT_DENSE(D(i, j, level),
                                             row_split,
                                             col_split);

            matmul(lower_splits[0], right_splits[0], reduce_splits[0], false, false, -1.0, 1.0);
          }
        }
      }

      // Schur's compliment between oc and co blocks
      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
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
              matmul(lower_splits[2], right_splits[1], reduce_splits[3], false, false, -1.0, 1.0);
            }
          }
        }
      }

      for (int i = block+1; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
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
              matmul(lower_splits[0], right_splits[1], reduce_splits[1], false, false, -1.0, 1.0);
            }
            // Schur's compliement between co and cc blocks where a new fill-in is created.
            // The product is a (co; oo)-sized matrix.
            else {
              if (!F.exists(i, j)) {
                int64_t block_size = D(i, block, level).rows;
                Matrix fill_in(block_size, rank);
                auto fill_splits = fill_in.split(std::vector<int64_t>(1, block_size - rank),
                                                 {});
                // Update the co block within the fill-in.
                matmul(lower_splits[0], right_splits[1], fill_splits[0], false, false, -1.0, 1.0);

                // Update the oo block within the fill-in.
                matmul(lower_splits[2], right_splits[1], fill_splits[1], false, false, -1.0, 1.0);

                F.insert(i, j, std::move(fill_in));
              }
              else {
                Matrix &fill_in = F(i, j);
                auto fill_splits = fill_in.split(std::vector<int64_t>(1, D(i, block, level).rows - rank),
                                                 {});
                // Update the co block within the fill-in.
                matmul(lower_splits[0], right_splits[1], fill_splits[0], false, false, -1.0, 1.0);
                // Update the oo block within the fill-in.
                matmul(lower_splits[2], right_splits[1], fill_splits[1], false, false, -1.0, 1.0);
              }
            }
          }
        }
      }

      // Schur's compliment between co and cc blocks
      for (int i = 0; i < nblocks; ++i) {
        for (int j = block+1; j < nblocks; ++j) {
          if ((is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) &&
              (is_admissible.exists(i, block, level) && !is_admissible(i, block, level))) {
            auto lower_splits = SPLIT_DENSE(D(i, block, level),
                                            D(i, block, level).rows - rank,
                                            col_split);
            auto right_splits = SPLIT_DENSE(D(block, j, level),
                                            row_split,
                                            D(block, j, level).cols - rank);
            // Schur's compliement between co and cc blocks where product exists as dense.
            if (is_admissible.exists(i, j, level) && !is_admissible(i, j, level)) {
              auto reduce_splits = SPLIT_DENSE(D(i, j, level),
                                               D(i, j, level).rows - rank,
                                               D(i, j, level).cols - rank);
              matmul(lower_splits[2], right_splits[0], reduce_splits[2],
                     false, false, -1.0, 1.0);
            }
            // Schur's compliement between co and cc blocks where a new fill-in is created.
            // The product is a (co, oo)-sized block.
            else {
              if (!F.exists(i, j)) {
                int64_t block_size = D(block, j, level).cols;
                Matrix fill_in(rank, block_size);
                auto fill_splits = fill_in.split({},
                                                 std::vector<int64_t>(1,
                                                                      block_size - rank));
                // Update the co block within the fill-ins.
                matmul(lower_splits[2], right_splits[0], fill_splits[0], false, false, -1.0, 1.0);
                // Update the oo block within the fill-ins.
                matmul(lower_splits[2], right_splits[1], fill_splits[1], false, false, -1.0, 1.0);
                F.insert(i, j, std::move(fill_in));
              }
              else {
                Matrix& fill_in = F(i, j);
                auto fill_splits = fill_in.split({},
                                                 std::vector<int64_t>(1,
                                                                      D(block, j, level).cols - rank));
                // Update the co block within the fill-ins.
                matmul(lower_splits[2], right_splits[0], fill_splits[0], false, false, -1.0, 1.0);
                // Update the oo block within the fill-ins.
                matmul(lower_splits[2], right_splits[1], fill_splits[1], false, false, -1.0, 1.0);
              }
            }
          }
        }
      }
    } // for (int block = 0; block < nblocks; ++block)

    F.erase_all();
  }

  void
  H2::factorize(const Domain& domain) {
    int64_t level = height;
    RowColLevelMap<Matrix> F;
    RowMap r, t;

    for (; level > 0; --level) {
      int64_t nblocks = level_blocks[level-1];
      bool is_all_dense_level = false;
      for (int64_t i = 0; i < nblocks; ++i) {
        if (!U.exists(i, level)) {
          is_all_dense_level = true;
        }
      }
      if (is_all_dense_level) {
        break;
      }

      factorize_level(level, nblocks, domain, r, t);

      int parent_level = level - 1;

      // Update transfer matrices on one level higher.
      for (int block = 0; block < nblocks; block += 2) {
        int parent_node = block / 2;
        int c1 = block;
        int c2 = block+1;

        if (row_has_admissible_blocks(parent_node, parent_level)) {
          Matrix& Utransfer = U(parent_node, parent_level);
          auto Utransfer_splits = Utransfer.split(2, 1);

          Matrix temp(Utransfer);
          auto temp_splits = temp.split(2, 1);

          if (r.exists(c1)) {
            matmul(r(c1), Utransfer_splits[0], temp_splits[0], false, false, 1, 0);
            r.erase(c1);
          }

          if (r.exists(c2)) {
            matmul(r(c2), Utransfer_splits[1], temp_splits[1], false, false, 1, 0);
            r.erase(c2);
          }

          U.erase(parent_node, parent_level);
          U.insert(parent_node, parent_level, std::move(temp));
        }

        if (col_has_admissible_blocks(parent_node, parent_level)) {
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

        // Merge the unfactorized parts.
        int64_t parent_nblocks = level_blocks[parent_level-1];
        for (int64_t i = 0; i < parent_nblocks; ++i) {
          for (int64_t j = 0; j < parent_nblocks; ++j) {
            if (is_admissible.exists(i, j, parent_level) && !is_admissible(i, j, parent_level)) {
              std::vector<int64_t> i_children({i * 2, i * 2 + 1}), j_children({j * 2, j * 2 + 1});
              Matrix D_unelim(rank*2, rank*2);
              auto D_unelim_splits = D_unelim.split(2, 2);

              for (int64_t ic1 = 0; ic1 < 2; ++ic1) {
                for (int64_t jc2 = 0; jc2 < 2; ++jc2) {
                  int64_t c1 = i_children[ic1], c2 = j_children[jc2];
                  if (!U.exists(c1, level)) { continue; }

                  if (is_admissible.exists(c1, c2, level) && !is_admissible(c1, c2, level)) {
                    auto D_splits = SPLIT_DENSE(D(c1, c2, level),
                                                D(c1, c2, level).rows-rank,
                                                D(c1, c2, level).cols-rank);
                    D_unelim_splits[ic1 * 2 + jc2] = D_splits[3];
                  }
                  else {
                    D_unelim_splits[ic1 * 2 + jc2] = S(c1, c2, level);
                  }
                }
              }

              // std::cout << "Merge insert: i-> " << i
              //           << " j -> " << " parent_level -> " << parent_level << std::endl;
              D.insert(i, j, parent_level, std::move(D_unelim));
            }
          }
        }
      } // for (block = 0; block < nblocks; block += 2)
    } // for (; level > 0; --level)

    int64_t last_nodes = level_blocks[level-1];

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
    int64_t num_nodes = level_blocks[level-1];
    int64_t c_offset = rank_offset;
    for (int64_t block = 0; block < num_nodes; ++block) {
      rank_offset += D(block, block, level).rows - rank;
    }

    for (int64_t block = 0; block < num_nodes; ++block) {
      int64_t rows = D(block, block, level).rows;
      int64_t c_size = rows - rank;

      // copy the complement part of the vector into the temporary vector
      for (int64_t i = 0; i < c_size; ++i) {
        copy(c_offset + c_size * block + i, 0) = x(c_offset + block * rows + i, 0);
      }
      // copy the rank part of the vector into the temporary vector
      for (int64_t i = 0; i < rank; ++i) {
        copy(rank_offset + rank * block + i, 0) = x(c_offset + block * rows + c_size + i, 0);
      }
    }

    x = copy;

    return rank_offset;
  }

  int64_t H2::permute_backward(Matrix& x, const int64_t level, int64_t rank_offset) {
    Matrix copy(x);
    int64_t num_nodes = level_blocks[level-1];
    int64_t c_offset = rank_offset;
    for (int64_t block = 0; block < num_nodes; ++block) {
      c_offset -= D(block, block, level).cols - rank;
    }

    for (int64_t block = 0; block < num_nodes; ++block) {
      int64_t cols = D(block, block, level).cols;
      int64_t c_size = cols - rank;

      for (int64_t i = 0; i < c_size; ++i) {
        copy(c_offset + block * cols + i, 0) = x(c_offset + block * c_size + i, 0);
      }

      for (int64_t i = 0; i < rank; ++i) {
        copy(c_offset + block * cols + c_size + i, 0) = x(rank_offset + rank * block + i, 0);
      }
    }

    x = copy;

    return c_offset;
  }

  void H2::solve_forward_level(Matrix& x_level, int level) {
    int nblocks = level_blocks[level-1];
    std::vector<int64_t> row_offsets;
    int64_t nrows = 0;
    for (int i = 0; i < nblocks; ++i) {
      row_offsets.push_back(nrows + D(i, i, level).rows);
      nrows += D(i, i, level).rows;
    }
    std::vector<Matrix> x_level_split = x_level.split(row_offsets, {});

    for (int block = 0; block < nblocks; ++block) {
      Matrix U_F = make_complement(U(block, level));
      Matrix prod = matmul(U_F, x_level_split[block], true);
      x_level_split[block] = prod;
    }

    // forward substitution with cc blocks
    for (int block = 0; block < nblocks; ++block) {
      int64_t row_split = D(block, block, level).rows - rank;
      int64_t col_split = D(block, block, level).cols - rank;
      auto block_splits = SPLIT_DENSE(D(block, block, level), row_split, col_split);

      Matrix x_block(x_level_split[block]);
      auto x_block_splits = x_block.split(std::vector<int64_t>(1, row_split), {});

      solve_triangular(block_splits[0], x_block_splits[0], Hatrix::Left, Hatrix::Lower, true);
      matmul(block_splits[2], x_block_splits[0], x_block_splits[1], false, false, -1.0, 1.0);
      x_level_split[block] = x_block;

      // Forward with the big c blocks on the lower part.
      for (int irow = block+1; irow < nblocks; ++irow) {
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
      for (int irow = 0; irow < block; ++irow) {
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

  void H2::solve_backward_level(Matrix& x_level, int level) {
    int nblocks = level_blocks[level-1];
    std::vector<int64_t> col_offsets;
    int64_t nrows = 0;
    for (int i = 0; i < nblocks; ++i) {
      col_offsets.push_back(nrows + D(i, i, level).cols);
      nrows += D(i, i, level).cols;
    }
    std::vector<Matrix> x_level_split = x_level.split(col_offsets, {});

    // backward substition using cc blocks
    for (int block = nblocks-1; block >= 0; --block) {
      int64_t row_split = D(block, block, level).rows - rank;
      int64_t col_split = D(block, block, level).cols - rank;
      auto block_splits = SPLIT_DENSE(D(block, block, level), row_split, col_split);
      // Apply co block.
      for (int left_col = block-1; left_col >= 0; --left_col) {
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
      for (int right_col = nblocks-1; right_col > block; --right_col) {
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

    for (int block = nblocks-1; block >= 0; --block) {
      auto V_F = make_complement(V(block, level));
      Matrix prod = matmul(V_F, x_level_split[block]);
      x_level_split[block] = prod;
    }
  }


  Matrix
  H2::solve(const Matrix& b, int64_t _level) {
    Matrix x(b);
    int level = _level;
    int64_t rhs_offset = 0;
    std::vector<Matrix> x_splits;

    // Forward
    for (; level > 0; --level) {
      int64_t nblocks = level_blocks[level-1];
      bool lr_exists = false;
      for (int block = 0; block < nblocks; ++block) {
        if (U.exists(block, level)) { lr_exists = true; }
      }
      if (!lr_exists) { break; }

      int n = 0;
      for (int i = 0; i < nblocks; ++i) { n += D(i, i, level).rows; }
      Matrix x_level(n, 1);
      for (int i = 0; i < x_level.rows; ++i) {
        x_level(i, 0) = x(rhs_offset + i, 0);
      }

      solve_forward_level(x_level, level);

      for (int i = 0; i < x_level.rows; ++i) {
        x(rhs_offset + i, 0) = x_level(i, 0);
      }

      rhs_offset = permute_forward(x, level, rhs_offset);
    }

    // Work with L0 and U0
    x_splits = x.split(std::vector<int64_t>(1, rhs_offset), {});
    Matrix x_last(x_splits[1]);
    int64_t last_nodes = level_blocks[level-1];
    auto x_last_splits = x_last.split(last_nodes, 1);

    for (int i = 0; i < last_nodes; ++i) {
      for (int j = 0; j < i; ++j) {
        matmul(D(i, j, level), x_last_splits[j], x_last_splits[i], false, false, -1.0, 1.0);
      }
      solve_triangular(D(i, i, level), x_last_splits[i], Hatrix::Left, Hatrix::Lower, true);
    }

    for (int i = last_nodes-1; i >= 0; --i) {
      for (int j = last_nodes-1; j > i; --j) {
        matmul(D(i, j, level), x_last_splits[j], x_last_splits[i], false, false, -1.0, 1.0);
      }
      solve_triangular(D(i, i, level), x_last_splits[i], Hatrix::Left, Hatrix::Upper, false);
    }
    x_splits[1] = x_last;

    level++;

    // Backward
    for (; level <= _level; ++level) {
      int64_t nblocks = level_blocks[level-1];

      bool lr_exists = false;
      for (int block = 0; block < nblocks; ++block) {
        if (V.exists(block, level)) { lr_exists = true; }
      }
      if (!lr_exists) { break; }

      int n = 0;
      for (int i = 0; i < nblocks; ++i) { n += D(i, i, level).cols; }
      Matrix x_level(n, 1);

      rhs_offset = permute_backward(x, level, rhs_offset);

      for (int i = 0; i < x_level.rows; ++i) {
        x_level(i, 0) = x(rhs_offset + i, 0);
      }

      solve_backward_level(x_level, level);

      for (int i = 0; i < x_level.rows; ++i) {
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

    if (nblocks == 1) { return level; }
    level_blocks.push_back(nblocks);

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
    std::cout << "nb: " << nblocks << std::endl;
    level_blocks.push_back(nblocks);
    int64_t level = 0;
    for (int64_t i = 0; i < nblocks; ++i) {
      for (int64_t j = 0; j < nblocks; ++j) {
        is_admissible.insert(i, j, level,
                             std::min(domain.boxes[i].diameter, domain.boxes[j].diameter) <=
                             admis * domain.boxes[i].distance_from(domain.boxes[j]));
      }
    }

    return geometry_admis_non_leaf(nblocks / 2, level+1);
  }

  void
  H2::calc_diagonal_based_admissibility(int64_t level) {
    if (level == 0) { return; }
    int64_t nblocks = pow(2, level);
    level_blocks.push_back(nblocks);
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

  void
  H2::actually_print_structure(int64_t level) {
    if (level == 0) { return; }
    int64_t nblocks = level_blocks[level-1];
    std::cout << "LEVEL: " << level << " NBLOCKS: " << nblocks << std::endl;
    for (int64_t i = 0; i < nblocks; ++i) {
      if (level == height) {
        std::cout << U(i, height).rows << " ";
      }
      std::cout << "| " ;
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

  Matrix
  H2::generate_column_block(int64_t block, int64_t block_size, const Domain& domain,
                            int64_t level) {
    Matrix AY(block_size, 0);
    int64_t nblocks = level_blocks[level-1];
    for (int64_t j = 0; j < nblocks; ++j) {
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
    for (int64_t i = 0; i < pow(2, level); ++i) {
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
      int nblocks = level_blocks[height-1];
      std::vector<Hatrix::Matrix> Y;

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
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
        std::tie(Utemp, Stemp)= generate_column_bases(i, domain.boxes[i].num_particles, domain, Y, height);
        U.insert(i, height, std::move(Utemp));
        Scol.insert(i, height, std::move(Stemp));
      }

      // Generate V leaf blocks
      for (int64_t j = 0; j < nblocks; ++j) {
        Matrix Stemp, Vtemp;
        std::tie(Stemp, Vtemp) = generate_row_bases(j, domain.boxes[j].num_particles, domain, Y, height);
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

    int block_size = Ubig_child1.rows + Ubig_child2.rows;

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

    int child1 = node * 2;
    int child2 = node * 2 + 1;

    Matrix Vbig_child1 = get_Vbig(child1, level+1);
    Matrix Vbig_child2 = get_Vbig(child2, level+1);

    int block_size = Vbig_child1.rows + Vbig_child2.rows;

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

    return get_block_size_row(domain, child1, child_level) + get_block_size_row(domain, child2, child_level);
  }

  int64_t
  H2::get_block_size_col(const Domain& domain, int64_t parent, int64_t level) {
    if (level == height) {
      return domain.boxes[parent].num_particles;
    }
    int64_t child_level = level + 1;
    int64_t child1 = parent * 2;
    int64_t child2 = parent * 2 + 1;

    return get_block_size_col(domain, child1, child_level) + get_block_size_col(domain, child2, child_level);
  }

  bool
  H2::row_has_admissible_blocks(int64_t row, int64_t level) {
    bool has_admis = false;
    for (int i = 0; i < pow(2, level); ++i) {
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
    for (int j = 0; j < pow(2, level); ++j) {
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
    int64_t nblocks = level_blocks[level-1];

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

      if (row_has_admissible_blocks(node, level)) {
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

      if (col_has_admissible_blocks(node, level)) {
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

    for (int row = 0; row < nblocks; ++row) {
      for (int col = 0; col < nblocks; ++col) {
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
           double _admis, std::string& admis_kind) :
      N(_N), rank(_rank), nleaf(_nleaf), admis(_admis), admis_kind(admis_kind) {
      if (admis_kind == "geometry_admis") {
        // TODO: use dual tree traversal for this.
        height = calc_geometry_based_admissibility(domain);
        // reverse the levels stored in the admis blocks.
        RowColLevelMap<bool> temp_is_admissible;

        for (int level = 0; level < height; ++level) {
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
        height = int64_t(log2(N / nleaf));
        calc_diagonal_based_admissibility(height);
        std::reverse(std::begin(level_blocks), std::end(level_blocks));
      }
      else {
        std::cout << "wrong admis condition: " << admis_kind << std::endl;
        abort();
      }

      is_admissible.insert(0, 0, 0, false);
      PV = height;

      generate_leaf_nodes(domain);
      RowLevelMap Uchild = U;
      ColLevelMap Vchild = V;

      for (int level = height-1; level > 0; --level) {
        std::tie(Uchild, Vchild) = generate_transfer_matrices(domain, level, Uchild, Vchild);
      }
    }

  double
  H2::construction_relative_error(const Domain& domain) {
    double error = 0;
    double dense_norm = 0;
    int64_t nblocks = level_blocks[height-1];

    for (int i = 0; i < nblocks; ++i) {
      for (int j = 0; j < nblocks; ++j) {
        if (is_admissible.exists(i, j, height) && !is_admissible(i, j, height)) {
          Matrix actual = Hatrix::generate_p2p_interactions(domain, i, j);
          Matrix expected = D(i, j, height);
          error += pow(norm(actual - expected), 2);
          dense_norm += pow(norm(actual), 2);
        }
      }
    }

    for (int level = height; level > 0; --level) {
      int64_t nblocks = level_blocks[level-1];

      for (int row = 0; row < nblocks; ++row) {
        for (int col = 0; col < nblocks; ++col) {
          if (is_admissible.exists(row, col, level) && is_admissible(row, col, level)) {
            Matrix Ubig = get_Ubig(row, level);
            Matrix Vbig = get_Vbig(col, level);

            Matrix expected_matrix = matmul(matmul(Ubig, S(row, col, level)), Vbig, false, true);
            Matrix actual_matrix = Hatrix::generate_p2p_interactions(domain, row, col, level, height);

            dense_norm += pow(norm(actual_matrix), 2);
            error += pow(norm(expected_matrix - actual_matrix), 2);
          }
        }
      }
    }

    return std::sqrt(error / dense_norm);
  }

  void H2::print_structure() {
    actually_print_structure(height);
  }

  double
  H2::low_rank_block_ratio() {
    double total = 0, low_rank = 0;

    int nblocks = level_blocks[height-1];
    for (int i = 0; i < nblocks; ++i) {
      for (int j = 0; j < nblocks; ++j) {
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

int main(int argc, char ** argv) {
  int64_t N = atoi(argv[1]);
  int64_t rank = atoi(argv[2]);
  int64_t nleaf = atoi(argv[3]);
  double admis = atof(argv[4]);
  int64_t ndim = atoi(argv[5]);
  std::string admis_kind(argv[6]);
  int64_t kernel_func = atoi(argv[7]);

  beta = 0.1;
  nu = 0.5;     //in matern, nu=0.5 exp (half smooth), nu=inf sqexp (inifinetly smooth)
  noise = 1.e-1;
  sigma = 1.0;

  Hatrix::Context::init();

  Hatrix::Domain domain(N, ndim);

  switch(kernel_func) {
  case 0: {                     // laplace kernel
    domain.generate_particles(0.0, 1.0 * N);
    Hatrix::kernel_function = Hatrix::laplace_kernel;
    break;
  }
  case 1: {                     // sqrexp
    domain.generate_starsh_grid_particles();
    Hatrix::kernel_function = Hatrix::sqrexp_kernel;
    break;
  }
  }

  domain.divide_domain_and_create_particle_boxes(nleaf);

  Hatrix::H2 A(domain, N, rank, nleaf, admis, admis_kind);
  // A.print_structure();
  double construct_error = A.construction_relative_error(domain);
  double lr_ratio = A.low_rank_block_ratio();

  A.factorize(domain);

  Hatrix::Matrix b = Hatrix::generate_random_matrix(N, 1);
  Hatrix::Matrix x = A.solve(b, A.height);
  Hatrix::Matrix Adense = Hatrix::generate_p2p_matrix(domain, N, N, 0, 0);
  Hatrix::Matrix x_solve = lu_solve(Adense, b);

  double solve_error = Hatrix::norm(x - x_solve) / Hatrix::norm(x_solve);

  Hatrix::Context::finalize();

  std::cout << "N=" << N << " admis=" << admis << " nleaf=" << nleaf << " ndim=" << ndim
            << " height= " << A.height << " rank=" << rank
            << " construct error= " << construct_error
            << " solve error= " << solve_error
            << " kernel func= " << kernel_func
            << " LR%= " << lr_ratio * 100 << "%" << std::endl;

}