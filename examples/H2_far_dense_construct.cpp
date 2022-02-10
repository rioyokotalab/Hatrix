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

#include "Hatrix/Hatrix.h"

// Construction of BLR2 strong admis matrix based on geometry based admis condition.
double PV = 1e-3;

#define SPLIT_DENSE(dense, row_split, col_split)        \
  dense.split(std::vector<int64_t>(1, row_split),       \
              std::vector<int64_t>(1, col_split));


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
  class Particle {
  public:
    double value;
    std::vector<double> coords;

    Particle(double x, double _value);
    Particle(double x, double y, double _value);
    Particle(double x, double y, double z, double _value);
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
    void generate_leaf_nodes(const Domain& domain);
    void actually_print_structure(int64_t level);
    bool row_has_admissible_blocks(int row, int64_t level);
    bool col_has_admissible_blocks(int64_t col, int64_t level);
    Matrix generate_column_block(int64_t block, int64_t block_size, const Domain& domain, int64_t level);
    std::tuple<Matrix, Matrix>
    generate_column_bases(int64_t block, int64_t block_size, const Domain& domain,
                          std::vector<Matrix>& Y, int64_t level);
    Matrix generate_row_block(int64_t block, int64_t block_size, const Domain& domain, int64_t level);
    std::tuple<Matrix, Matrix>
    generate_row_bases(int64_t block, int64_t block_size, const Domain& domain,
                       std::vector<Matrix>& Y, int64_t level);
    std::tuple<RowLevelMap, ColLevelMap> generate_transfer_matrices(const Domain& domain,
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
                               int64_t block_size, const Domain& domain, int level);

    void
    calc_geometry_based_admissibility(int64_t level, const Domain& domain);
    void calc_diagonal_based_admissibility(int64_t level);
    void coarsen_blocks(int64_t level);
    int64_t geometry_admis_non_leaf(int64_t nblocks, int64_t level);
    int64_t calc_geometry_based_admissibility(const Domain& domain);
  public:
    H2(const Domain& domain, int64_t _N, int64_t _rank, int64_t _nleaf, double _admis,
       std::string& admis_kind);
    double construction_relative_error(const Domain& domain);
    void print_structure();
  };
}

namespace Hatrix {
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

  // Generate a full dense laplacian matrix assuming unit charges.
  Matrix generate_laplacend_matrix(const std::vector<Hatrix::Particle>& particles,
                                   int64_t nrows, int64_t ncols) {
    Matrix out(nrows, ncols);

    for (int64_t i = 0; i < nrows; ++i) {
      for (int64_t j = 0; j < ncols; ++j) {
        out(i, j) = laplace_kernel(particles[i].coords, particles[j].coords);
      }
    }
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

        out(i, j) = laplace_kernel(domain.particles[source+i].coords,
                                   domain.particles[target+j].coords);
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


  // https://www.csd.uwo.ca/~mmorenom/cs2101a_moreno/Barnes-Hut_Algorithm.pdf
  void
  Domain::orthogonal_recursive_bisection_1dim(int64_t start,
                                              int64_t end,
                                              std::string morton_index,
                                              int64_t nleaf) {
    // Sort the particles only by the X axis since that is the only axis that needs to be bisected.
    std::sort(particles.begin()+start,
              particles.begin()+end, [](const Particle& lhs, const Particle& rhs) {
      return lhs.coords[0] <= rhs.coords[0];
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
        double theta = dis(gen);
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

    actually_print_structure(level-1);
  }

  Matrix
  H2::generate_column_block(int64_t block, int64_t block_size, const Domain& domain,
                            int64_t level) {
    Matrix AY(block_size, 0);
    int64_t nblocks = level_blocks[level-1];
    for (int64_t j = 0; j < nblocks; ++j) {
      if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) { continue; }
      Hatrix::Matrix dense = Hatrix::generate_p2p_interactions(domain, block, j);
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
      Hatrix::Matrix dense = Hatrix::generate_p2p_interactions(domain, i, block);
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
  }

  Matrix
  H2::get_Vbig(int64_t node, int64_t level) {
    if (level == height) {
      return V(node, height);
    }
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
    }
    is_admissible.insert(0, 0, 0, false);

    generate_leaf_nodes(domain);

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

    for (int level = height; level > height-1; --level) {
      int64_t nblocks = level_blocks[level-1];

      for (int row = 0; row < nblocks; ++row) {
        for (int col = 0; col < nblocks; ++col) {
          if (is_admissible.exists(row, col, level) && is_admissible(row, col, level)) {
            Matrix Ubig = get_Ubig(row, level);
            Matrix Vbig = get_Vbig(col, level);

            Ubig.print_meta();
            Vbig.print_meta();
            S(row, col, level).print_meta();

            Matrix expected_matrix = matmul(matmul(Ubig, S(row, col, level)), Vbig, false, true);
            Matrix actual_matrix = Hatrix::generate_p2p_interactions(domain, row, col);

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
}

int main(int argc, char ** argv) {
  int64_t N = atoi(argv[1]);
  int64_t rank = atoi(argv[2]);
  int64_t nleaf = atoi(argv[3]);
  double admis = atof(argv[4]);
  int64_t ndim = atoi(argv[5]);
  std::string admis_kind(argv[6]);

  Hatrix::Context::init();

  Hatrix::Domain domain(N, ndim);
  domain.generate_particles(0.0, 1.0 * N);
  domain.divide_domain_and_create_particle_boxes(nleaf);

  Hatrix::H2 A(domain, N, rank, nleaf, admis, admis_kind);
  A.print_structure();
  double construct_error = A.construction_relative_error(domain);


  Hatrix::Context::finalize();

  std::cout << "construct error: " << construct_error << std::endl;

}
