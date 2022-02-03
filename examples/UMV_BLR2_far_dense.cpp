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

  for (int i = 0; i < Q_F.rows; ++i) {
    for (int j = 0; j < Q_F.cols - Q.cols; ++j) {
      Q_F(i, j) = Q_full(i, j + Q.cols);
    }
  }

  for (int i = 0; i < Q_F.rows; ++i) {
    for (int j = 0; j < Q.cols; ++j) {
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
    void orthogonal_recursive_bisection_1dim(int64_t start, int64_t end, std::string morton_index, int64_t nleaf);
    void orthogonal_recursive_bisection_2dim(int64_t start, int64_t end, std::string morton_index, int64_t nleaf, int64_t axis);
    void orthogonal_recursive_bisection_3dim(int64_t start, int64_t end, std::string morton_index,
                                             int64_t nleaf, int64_t axis);
  public:
    Domain(int64_t N, int64_t ndim);
    void generate_particles(double min_val, double max_val);
    void divide_domain_and_create_particle_boxes(int64_t nleaf);
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
        double rij = 0;
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
  void Domain::orthogonal_recursive_bisection_1dim(int64_t start, int64_t end, std::string morton_index, int64_t nleaf) {
    // Sort the particles only by the X axis since that is the only axis that needs to be bisected.
    std::sort(particles.begin()+start, particles.begin()+end, [](const Particle& lhs, const Particle& rhs) {
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

  void Domain::orthogonal_recursive_bisection_2dim(int64_t start, int64_t end, std::string morton_index, int64_t nleaf, int64_t axis) {
    std::sort(particles.begin() + start, particles.begin() + end, [&](const Particle& lhs, const Particle& rhs) {
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

        boxes.push_back(Box(diameter, center_x, center_y, start_index, end_index, morton_index, num_points));
      }
      else {
        orthogonal_recursive_bisection_2dim(start, end, morton_index, nleaf, (axis + 1) % ndim);
      }
    }
    else {
      int64_t middle = (start + end) / 2;
      orthogonal_recursive_bisection_2dim(start, middle, morton_index + "0", nleaf, (axis + 1) % ndim);
      orthogonal_recursive_bisection_2dim(middle, end, morton_index + "1", nleaf, (axis + 1) % ndim);
    }
  }

  void Domain::orthogonal_recursive_bisection_3dim(int64_t start, int64_t end, std::string morton_index,
                                                   int64_t nleaf, int64_t axis) {
    std::sort(particles.begin() + start, particles.begin() + end, [&](const Particle& lhs, const Particle& rhs) {
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

        boxes.push_back(Box(diameter, center_x, center_y, center_z, start_index, end_index, morton_index, num_points));
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
        // std::cout << "adding point: " << vec[i] << std::endl;
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

        particles.push_back(Hatrix::Particle(x, y, z, min_val + (double(i) / double(range))));
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

  class BLR2 {
  private:
    const int64_t level = 1;
    // Map of inadmissible indices.
    RowColLevelMap<bool> is_admissible;
    // Vector of vector for storing the actual indices of all the inadmissible blocks in a given row.
    std::vector<std::vector<int64_t> > inadmissible_row_indices, admissible_row_indices;
    std::vector<std::vector<int64_t> > inadmissible_col_indices, admissible_col_indices;

    RowLevelMap U;
    ColLevelMap V;
    RowColLevelMap<Matrix> D, S;

    int64_t N, nleaf, rank, ndim, nblocks;
    double admis;

    std::string admis_kind;

    RowLevelMap Srow, Scol;

    void calc_geometry_based_admissibility(const Domain& domain) {
      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          is_admissible.insert(i, j, level, std::min(domain.boxes[i].diameter, domain.boxes[j].diameter) <=
                               admis * domain.boxes[i].distance_from(domain.boxes[j]));
        }
      }
    }

    void calc_diagonal_based_admissibility() {
      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          is_admissible.insert(i, j, level, std::abs(i - j) >= admis);
        }
      }
    }

    void populate_admis_indices() {
      inadmissible_row_indices.resize(nblocks);
      admissible_row_indices.resize(nblocks);

      inadmissible_col_indices.resize(nblocks);
      admissible_col_indices.resize(nblocks);

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          if (is_admissible(i, j, level)) {
            admissible_row_indices[i].push_back(j);
          }
          else {
            inadmissible_row_indices[i].push_back(j);
          }
        }
      }

      for (int j = 0; j < nblocks; ++j) {
        for (int i = 0; i < nblocks; ++i) {
          if (is_admissible(j, i, level)) {
            admissible_col_indices[j].push_back(i);
          }
          else {
            inadmissible_col_indices[j].push_back(i);
          }
        }
      }
    }

    void factorize_level(int level) {
      RowColMap<Matrix> F;      // fill-in blocks.

      for (int block = 0; block < nblocks; ++block) {
        if (block > 0 && admis != 0) {
          {
            // Scan for fill-ins in the same row as this diagonal block.
            Matrix row_concat(U(block, level).rows, 0);
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
                if (is_admissible.exists(block, j, level) && is_admissible(block, j, level)) {
                  if (F.exists(block, j)) {
                    Matrix Fp = matmul(F(block, j), V(j, level), false, true);
                    row_concat = concat(row_concat, Fp, 1);
                  }
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
                    Matrix Fp = matmul(F(block, j), V(j, level), false, true);
                    SpF = matmul(matmul(UN1, Fp, true, false), V(j, level));
                    Sbar_block_j = Sbar_block_j + SpF;
                    F.erase(block, j);
                  }

                  S.erase(block, j, level);
                  S.insert(block, j, level, std::move(Sbar_block_j));
                }
              }
              U.erase(block, level);
              U.insert(block, level, std::move(UN1));            }
          }

          {
            // Scan for fill-ins in the same col as this diagonal block.
            Matrix col_concat(0, V(block, level).rows);
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
                if (is_admissible.exists(i, block, level) && is_admissible(i, block, level)) {
                  if (F.exists(i, block)) {
                    Matrix Fp = matmul(U(i, level), F(i, block));
                    col_concat = concat(col_concat, Fp, 0);
                  }
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
                    Matrix Fp = matmul(U(i,level), F(i, block));
                    Matrix SpF = matmul(matmul(U(i,level), Fp, true, false), VN2T, false, true);
                    Sbar_i_block = Sbar_i_block + SpF;

                    F.erase(i, block);
                  }

                  S.erase(i, block, level);
                  S.insert(i, block, level, std::move(Sbar_i_block));
                }
              }

              V.erase(block, level);
              V.insert(block, level, transpose(VN2T));
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

        int64_t row_split = V(block, level).rows - rank, col_split = U(block, level).rows - rank;

        // The diagonal block is split along the row and column.
        auto diagonal_splits = SPLIT_DENSE(D(block, block, level), row_split, col_split);
        Matrix& Dcc = diagonal_splits[0];
        lu(Dcc);

        // TRSM_L with CC blocks on the row
        for (int j = block + 1; j < nblocks; ++j) {
          if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) {
            int64_t col_split = U(j, level).rows - rank;
            auto D_splits = SPLIT_DENSE(D(block, j, level), row_split, col_split);
            solve_triangular(Dcc, D_splits[0], Hatrix::Left, Hatrix::Lower, true);
          }
        }

        // TRSM_L with co blocks on this row
        for (int j = 0; j < nblocks; ++j) {
          if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) {
            int64_t col_split = U(j, level).rows - rank;
            auto D_splits = SPLIT_DENSE(D(block, j, level), row_split, col_split);
            solve_triangular(Dcc, D_splits[1], Hatrix::Left, Hatrix::Lower, true);
          }
        }

        // TRSM_U with cc blocks on the column
        for (int i = block + 1; i < nblocks; ++i) {
          if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) {
            int64_t row_split = V(i, level).rows - rank;
            auto D_splits = SPLIT_DENSE(D(i, block, level), row_split, col_split);
            solve_triangular(Dcc, D_splits[0], Hatrix::Right, Hatrix::Upper, false);
          }
        }

        // TRSM_U with oc blocks on the column
        for (int i = 0; i < nblocks; ++i) {
          if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) {
            int64_t row_split = V(i, level).rows - rank;
            auto D_splits = SPLIT_DENSE(D(i, block, level), row_split, col_split);
            solve_triangular(Dcc, D_splits[2], Hatrix::Right, Hatrix::Upper, false);
          }
        }

        // Schur's compliment between cc blocks
        for (int i = block+1; i < nblocks; ++i) {
          for (int j = block+1; j < nblocks; ++j) {
            if ((is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) &&
                (is_admissible.exists(i, block, level) && !is_admissible(i, block, level))) {
              auto lower_splits = SPLIT_DENSE(D(i, block, level),
                                              V(i, level).rows - rank,
                                              col_split);
              auto right_splits = SPLIT_DENSE(D(block, j, level),
                                              row_split,
                                              U(j, level).rows - rank);

              if (is_admissible.exists(i, j, level) && !is_admissible(i, j, level)) {
                auto reduce_splits = SPLIT_DENSE(D(i, j, level),
                                                 V(i, level).rows - rank,
                                                 U(j, level).rows - rank);

                matmul(lower_splits[0], right_splits[0], reduce_splits[0], false, false, -1.0, 1.0);
              }
              else {
                // Fill in between cc blocks.
              }
            }
          }
        }

        // Schur's compliment between oc and co blocks where the results exists before the diagonal block.
        for (int i = 0; i <= block; ++i) {
          for (int j = 0; j <= block; ++j) {
            if ((is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) &&
                (is_admissible.exists(i, block, level) && !is_admissible(i, block, level))) {
              auto lower_splits = SPLIT_DENSE(D(i, block, level),
                                              V(i, level).rows - rank,
                                              col_split);
              auto right_splits = SPLIT_DENSE(D(block, j, level),
                                              row_split,
                                              U(j, level).rows - rank);

              if (!is_admissible(i, j, level)) {
                auto reduce_splits = SPLIT_DENSE(D(i, j, level),
                                                 V(i, level).rows - rank,
                                                 U(j, level).rows - rank);
                matmul(lower_splits[2], right_splits[1], reduce_splits[3], false, false, -1.0, 1.0);
              }
            }
          }
        }

        // Schur's compliment between cc and co blocks where the result exists before the diagonal block.
        for (int i = block+1; i < nblocks; ++i) {
          for (int j = 0; j <= block; ++j) {
            if ((is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) &&
                (is_admissible.exists(i, block, level) && !is_admissible(i, block, level))) {
              auto lower_splits = SPLIT_DENSE(D(i, block, level),
                                              V(i, level).rows - rank,
                                              col_split);
              auto right_splits = SPLIT_DENSE(D(block, j, level),
                                              row_split,
                                              V(j, level).rows - rank);
              // Schur's compliment between co and cc blocks where product exists as dense.
              if (is_admissible.exists(i, j, level) && !is_admissible(i, j, level)) {
                auto reduce_splits = SPLIT_DENSE(D(i, j, level),
                                                 V(i, level).rows - rank,
                                                 U(j, level).rows - rank);
                matmul(lower_splits[0], right_splits[1], reduce_splits[1], false, false, -1.0, 1.0);
              }
              // Schur's compliement between co and cc blocks where a new fill-in is created.
              // The product is a (co; oo)-sized matrix.
              else {
                if (!F.exists(i, j)) {
                  Matrix fill_in(V(i, level).rows, rank);
                  auto fill_splits = fill_in.split(std::vector<int64_t>(1, V(i, level).rows - rank),
                                                   {});
                  // Update the co block within the fill-in.
                  matmul(lower_splits[0], right_splits[1], fill_splits[0], false, false, -1.0, 1.0);

                  // Update the oo block within the fill-in.
                  matmul(lower_splits[2], right_splits[1], fill_splits[1], false, false, -1.0, 1.0);

                  F.insert(i, j, std::move(fill_in));
                }
                else {
                  Matrix &fill_in = F(i, j);
                  auto fill_splits =
                    fill_in.split(std::vector<int64_t>(1, V(block, level).rows - rank),
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

        // Schur's compliment between oc and cc blocks where the result exists before the diagonal blocks.
        for (int i = 0; i <= block; ++i) {
          for (int j = block+1; j < nblocks; ++j) {
            if ((is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) &&
                (is_admissible.exists(i, block, level) && !is_admissible(i, block, level))) {
              auto lower_splits = SPLIT_DENSE(D(i, block, level),
                                              V(i, level).rows - rank,
                                              col_split);
              auto right_splits = SPLIT_DENSE(D(block, j, level),
                                              row_split,
                                              U(j, level).rows - rank);
              // Schur's compliement between oc and cc blocks where product exists as dense.
              if (is_admissible.exists(i, j, level) && !is_admissible(i, j, level)) {
                auto reduce_splits = SPLIT_DENSE(D(i, j, level),
                                                 V(i, level).rows - rank,
                                                 U(j, level).rows - rank);
                matmul(lower_splits[2], right_splits[0], reduce_splits[2],
                       false, false, -1.0, 1.0);
              }
              // Schur's compliement between co and cc blocks where a new fill-in is created.
              // The product is a (oc, oo)-sized block.
              else {
                if (!F.exists(i, j)) {
                  Matrix fill_in(rank, U(j, level).rows);
                  auto fill_splits =
                    fill_in.split({},
                                  std::vector<int64_t>(1, U(j, level).rows - rank));
                  // Update the oc block within the fill-ins.
                  matmul(lower_splits[2], right_splits[0], fill_splits[0], false, false, -1.0, 1.0);
                  // Update the oo block within the fill-ins.
                  matmul(lower_splits[2], right_splits[1], fill_splits[1], false, false, -1.0, 1.0);
                  F.insert(i, j, std::move(fill_in));
                }
                else {
                  Matrix& fill_in = F(i, j);
                  auto fill_splits =
                    fill_in.split({},
                                  std::vector<int64_t>(1, U(j, level).rows - rank));
                  // Update the oc block within the fill-ins.
                  matmul(lower_splits[2], right_splits[0], fill_splits[0], false, false, -1.0, 1.0);
                  // Update the oo block within the fill-ins.
                  matmul(lower_splits[2], right_splits[1], fill_splits[1], false, false, -1.0, 1.0);
                }
              }
            }
          }
        }
      } // for (int block = 0; block < nblocks; ++block)
    }

    void solve_forward_level(Matrix& x_level, int level) {
      std::vector<int64_t> row_offsets;
      int nrows = 0;
      for (int i = 0; i < nblocks; ++i) {
        row_offsets.push_back(nrows + U(i, level).rows);
        nrows += U(i, level).rows;
      }
      std::vector<Matrix> x_level_split = x_level.split(row_offsets, {});

      for (int block = 0; block < nblocks; ++block) {
        int block_size = U(block, level).rows;
        Matrix U_F = make_complement(U(block, level));
        std::cout << "block -> " << block << " r -> "  << x_level_split[block].rows << std::endl;
        Matrix prod = matmul(U_F, x_level_split[block], true);
        x_level_split[block] = prod;

        // for (int64_t i = 0; i < block_size; ++i) {
        //   x_level(block * block_size + i, 0) = prod(i, 0);
        // }
      }

      // forward substitution with cc blocks
      for (int block = 0; block < nblocks; ++block) {
        int block_size = V(block, level).rows;
        int c_size = block_size - rank;

        int64_t row_split = block_size - U(block, level).cols, col_split = block_size - V(block, level).cols;
        auto block_splits = SPLIT_DENSE(D(block, block, level), row_split, col_split);

        Matrix x_block(x_level_split[block]);
        auto x_block_splits = x_block.split(std::vector<int64_t>(1, row_split), {});

        solve_triangular(block_splits[0], x_block_splits[0], Hatrix::Left, Hatrix::Lower, true);
        matmul(block_splits[2], x_block_splits[0], x_block_splits[1], false, false, -1.0, 1.0);
        for (int64_t i = 0; i < block_size; ++i) {
          x_level(block * block_size + i, 0) = x_block(i, 0);
        }

        // Forward with the big c blocks on the lower part.
        for (int irow = block+1; irow < nblocks; ++irow) {
          if (is_admissible.exists(irow, block, level) && !is_admissible(irow, block, level)) {
            int64_t row_split = V(irow, level).rows - rank;
            int64_t col_split = U(block, level).rows - rank;
            auto lower_splits = D(irow, block, level).split({}, std::vector<int64_t>(1, row_split));

            Matrix x_block(x_level_split[block]), x_level_irow(x_level_split[irow]);
            auto x_block_splits = x_block.split(std::vector<int64_t>(1, col_split), {});

            matmul(lower_splits[0], x_block_splits[0], x_level_irow, false, false, -1.0, 1.0);
            for (int64_t i = 0; i < block_size; ++i) {
              x_level(irow * block_size + i, 0) = x_level_irow(i, 0);
            }
          }
        }

        // Forward with the oc parts of the block that are actually in the upper part of the matrix.
        for (int irow = 0; irow < block; ++irow) {
          if (is_admissible.exists(irow, block, level) && !is_admissible(irow, block, level)) {
            int64_t row_split = V(irow, level).rows - rank;
            int64_t col_split = U(block, level).rows - rank;
            auto top_splits = SPLIT_DENSE(D(irow, block, level), row_split, col_split);

            Matrix x_irow(x_level_split[irow]), x_block(x_level_split[block]);
            auto x_irow_splits = x_irow.split(std::vector<int64_t>(1, row_split), {});
            auto x_block_splits = x_block.split(std::vector<int64_t>(1, col_split), {});

            matmul(top_splits[2], x_block_splits[0], x_irow_splits[1], false, false, -1.0, 1.0);

            for (int64_t i = 0; i < block_size; ++i) {
              x_level(irow * block_size + i, 0) = x_irow(i, 0);
            }
          }
        }
      }
    }

    void solve_backward_level(Matrix& x_level, int level) {
      std::vector<int64_t> col_offsets;
      for (int i = 0; i < nblocks; ++i) { col_offsets.push_back(U(i, level).rows); }
      std::vector<Matrix> x_level_split = x_level.split(col_offsets, {});

      // backward substition using cc blocks
      for (int block = nblocks-1; block >= 0; --block) {
        int block_size = V(block, level).rows;
        int64_t row_split = U(block, level).rows - rank, col_split = V(block, level).rows - rank;
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
            for (int64_t i = 0; i < block_size; ++i) {
              x_level(block * block_size + i, 0) = x_block(i, 0);
            }
          }
        }

        // Apply c block present on the right of this diagonal block.
        for (int right_col = nblocks-1; right_col > block; --right_col) {
          if (is_admissible.exists(block, right_col, level) &&
              !is_admissible(block, right_col, level)) {
            int64_t row_split = U(block, level).rows - rank;
            auto right_splits = D(block, right_col, level).
              split(std::vector<int64_t>(1, row_split), {});

            Matrix x_block(x_level_split[block]);
            auto x_block_splits = x_block.split(std::vector<int64_t>(1, row_split), {});

            matmul(right_splits[0], x_level_split[right_col],
                   x_block_splits[0], false, false, -1.0, 1.0);
            for (int64_t i = 0; i < block_size; ++i) {
              x_level(block * block_size + i, 0) = x_block(i, 0);
            }
          }
        }

        Matrix x_block(x_level_split[block]);
        auto x_block_splits = x_block.split(std::vector<int64_t>(1, row_split), {});
        matmul(block_splits[1], x_block_splits[1], x_block_splits[0], false, false, -1.0, 1.0);
        solve_triangular(block_splits[0], x_block_splits[0], Hatrix::Left, Hatrix::Upper, false);
        for (int64_t i = 0; i < block_size; ++i) {
          x_level(block * block_size + i, 0) = x_block(i, 0);
        }
      }

      for (int block = nblocks-1; block >= 0; --block) {
        int block_size = V(block, level).rows;
        auto V_F = make_complement(V(block, level));
        Matrix prod = matmul(V_F, x_level_split[block]);
        for (int i = 0; i < block_size; ++i) {
          x_level(block * block_size + i, 0) = prod(i, 0);
        }
      }
    }

    // permute the vector forward and return the offset at which the new vector begins.
    int64_t permute_forward(Matrix& x, const int64_t level, int64_t rank_offset) {
      Matrix copy(x);
      int64_t num_nodes = pow(2, level);
      int64_t c_offset = rank_offset;
      for (int64_t block = 0; block < num_nodes; ++block) {
        rank_offset += D(block, block, level).rows - U(block, level).cols;
      }

      for (int64_t block = 0; block < num_nodes; ++block) {
        int64_t rows = U(block, level).rows;
        int64_t rank = U(block, level).cols;
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

    int64_t permute_backward(Matrix& x, const int64_t level, int64_t rank_offset) {
      Matrix copy(x);
      int64_t num_nodes = pow(2, level);
      int64_t c_offset = rank_offset;
      for (int64_t block = 0; block < num_nodes; ++block) {
        c_offset -= D(block, block, level).rows - U(block, level).cols;
      }

      for (int64_t block = 0; block < num_nodes; ++block) {
        int64_t rows = U(block, level).rows;
        int64_t rank = U(block, level).cols;
        int64_t c_size = rows - rank;

        for (int64_t i = 0; i < c_size; ++i) {
          copy(c_offset + block * rows + i, 0) = x(c_offset + block * c_size + i, 0);
        }

        for (int64_t i = 0; i < rank; ++i) {
          copy(c_offset + block * rows + c_size + i, 0) = x(rank_offset + rank * block + i, 0);
        }
      }

      x = copy;

      return c_offset;
    }


  public:
    BLR2(const Domain& domain, int64_t N,
         int64_t nleaf, int64_t rank, int64_t ndim, double admis, std::string& admis_kind) :
      N(N), nleaf(nleaf), rank(rank), ndim(ndim), admis(admis), admis_kind(admis_kind) {
      nblocks = domain.boxes.size();

      if (admis_kind == "geometry_admis") {
        calc_geometry_based_admissibility(domain);
      }
      else if (admis_kind == "diagonal_admis") {
        calc_diagonal_based_admissibility();
      }

      populate_admis_indices();

      int64_t oversampling = 5;
      Hatrix::Matrix Utemp, Stemp, Vtemp;
      double error;
      std::vector<Hatrix::Matrix> Y;

      for (int64_t i = 0; i < nblocks; ++i) {
        Y.push_back(generate_random_matrix(domain.boxes[i].num_particles, rank + oversampling));
      }

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          if (!is_admissible(i, j, level)) {
            D.insert(i, j, level, generate_p2p_interactions(domain, i, j));
          }
        }
      }

      for (int64_t i = 0; i < nblocks; ++i) {
        Hatrix::Matrix AY(domain.boxes[i].num_particles, rank + oversampling);
        for (unsigned j = 0; j < admissible_row_indices[i].size(); ++j) {
          int64_t jcol = admissible_row_indices[i][j];
          Hatrix::Matrix dense = generate_p2p_interactions(domain, i, jcol);
          Hatrix::matmul(dense, Y[jcol], AY);
        }
        std::tie(Utemp, Stemp, Vtemp, error) = Hatrix::truncated_svd(AY, rank);
        U.insert(i, level, std::move(Utemp));
        Scol.insert(i, level, std::move(Stemp));
      }

      for (int64_t j = 0; j < nblocks; ++j) {
        Hatrix::Matrix YtA(rank + oversampling, domain.boxes[j].num_particles);

        for (long unsigned int i = 0; i < admissible_col_indices[j].size(); ++i) {
          int64_t irow = admissible_col_indices[j][i];
          Hatrix::Matrix dense = Hatrix::generate_p2p_interactions(domain, irow, j);
          Hatrix::matmul(Y[irow], dense, YtA, true);
        }
        std::tie(Utemp, Stemp, Vtemp, error) = Hatrix::truncated_svd(YtA, rank);
        V.insert(j, level, std::move(transpose(Vtemp)));
        Srow.insert(j, level, std::move(Stemp));
      }

      for (int i = 0; i < nblocks; ++i) {
        for (unsigned j = 0; j < admissible_row_indices[i].size(); ++j) {
          int64_t jcol = admissible_row_indices[i][j];
          Hatrix::Matrix dense = Hatrix::generate_p2p_interactions(domain, i, jcol);
          S.insert(i, jcol, level, Hatrix::matmul(Hatrix::matmul(U(i, level), dense, true), V(jcol, level)));
        }
      }

      // col dimensions
      std::cout << "Column dims:\n";
      for (int i = 0; i < nblocks; ++i) {
        std::cout << U(i, level).rows << " ";
      }
      std::cout << std::endl;

      std::cout << "Row dims:\n";
      for (int i = 0; i < nblocks; ++i) {
        std::cout << V(i, level).rows << " ";
      }
      std::cout << std::endl;
    }

    void factorize(const Domain& domain) {
      factorize_level(level);

      std::vector<int64_t> row_splits, col_splits;
      int64_t nrows = 0, ncols = 0;
      for (int i = 0; i < nblocks; ++i) {
        int64_t row_rank = U(i, level).cols, col_rank = V(i, level).cols;
        row_splits.push_back(nrows + row_rank);
        col_splits.push_back(ncols + col_rank);
        nrows += row_rank;
        ncols += col_rank;
      }

      Matrix last(nrows, ncols);
      auto last_splits = last.split(row_splits, col_splits);

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          if (is_admissible(i, j, level)) {
            last_splits[i * nblocks + j] = S(i, j, level);
          }
          else {
            auto D_splits = SPLIT_DENSE(D(i, j, level),
                                        V(i, level).rows - rank,
                                        U(j, level).rows - rank);
            last_splits[i * nblocks + j] = D_splits[3];
          }
        }
      }

      lu(last);

      D.insert(0, 0, 0, std::move(last));
    }

    Matrix solve(const Matrix& b) {
      Matrix x(b);
      int rhs_offset;

      solve_forward_level(x, level);

      rhs_offset = permute_forward(x, level, rhs_offset);

      auto permute_splits = x.split(std::vector<int64_t>(1, rhs_offset), {});
      solve_triangular(D(0,0,0), permute_splits[1], Hatrix::Left, Hatrix::Lower, true);
      solve_triangular(D(0,0,0), permute_splits[1], Hatrix::Left, Hatrix::Upper, false);

      rhs_offset = permute_backward(x, level, rhs_offset);

      solve_backward_level(x, level);

      return x;
    }

    double construction_error(const Domain& domain) {
      // Check dense blocks
      double error = 0; double dense_norm = 0;

      for (int row = 0; row < nblocks; ++row) {

        for (int j = 0; j < inadmissible_row_indices[row].size(); ++j) {
          int64_t col = inadmissible_row_indices[row][j];
          auto dense = Hatrix::generate_p2p_interactions(domain, row, col);
          dense_norm += pow(norm(dense), 2);
          error += pow(norm(D(row, col, level) -  dense), 2);
          j++;
        }
      }

      for (unsigned i = 0; i < nblocks; ++i) {
        for (unsigned j = 0; j < admissible_row_indices[i].size(); ++j) {
          int64_t jcol = admissible_row_indices[i][j];
          auto dense = generate_p2p_interactions(domain, i, jcol);
          Matrix& Ubig = U(i, level);
          Matrix& Vbig = V(jcol, level);
          Matrix expected = matmul(matmul(Ubig, S(i, jcol, level)), Vbig, false, true);
          Matrix actual = generate_p2p_interactions(domain, i, jcol);
          error += pow(norm(expected - actual), 2);
          dense_norm += pow(norm(actual), 2);
        }
      }

      return std::sqrt(error / dense_norm);
    }

    double low_rank_block_ratio() {
      double total = 0, low_rank = 0;
      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          if (is_admissible(i, j, level)) {
            low_rank += 1;
          }
          total += 1;
        }
      }

      return low_rank / total;
    }

    void print_structure() {
      std::cout << "BLR " << nblocks << " x " << nblocks << std::endl;
      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          std::cout << " | " << is_admissible(i, j, level);
        }
        std::cout << " | \n";
      }
    }
  };
}


int main(int argc, char** argv) {
  int64_t N = atoi(argv[1]);
  int64_t nleaf = atoi(argv[2]);
  int64_t rank = atoi(argv[3]);
  double admis = atof(argv[4]);
  int64_t ndim = atoi(argv[5]);
  std::string admis_kind(argv[6]);

  Hatrix::Context::init();
  Hatrix::Domain domain(N, ndim);
  domain.generate_particles(0.0, 1.0 * N);
  domain.divide_domain_and_create_particle_boxes(nleaf);

  if (N % nleaf != 0) {
    std::cout << "N % nleaf != 0. Aborting.\n";
    abort();
  }

  Hatrix::BLR2 A(domain, N, nleaf, rank, ndim, admis, admis_kind);
  A.print_structure();
  double construct_error = A.construction_error(domain);

  A.factorize(domain);

  Hatrix::Matrix b = Hatrix::generate_random_matrix(N, 1);

  Hatrix::Matrix x = A.solve(b);
  Hatrix::Matrix Adense = Hatrix::generate_laplacend_matrix(domain.particles, N, N);
  Hatrix::Matrix x_solve = lu_solve(Adense, b);

  Hatrix::Context::finalize();

  std::cout << "N: " << N << " rank: " << rank << " nleaf: " << nleaf
            << " admis: " <<  admis << " ndim: " << ndim
            << " construct error: " << construct_error
            << " LR%: " << A.low_rank_block_ratio() * 100 << "%" <<  "\n";
}
