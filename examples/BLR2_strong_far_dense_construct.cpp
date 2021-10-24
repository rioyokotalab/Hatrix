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

namespace Hatrix {
  class Particle {
  public:
    double value;
    std::vector<double> coords;

    Particle(double x, double _value) : value(_value)  {
      coords.push_back(x);
    }

    Particle(double x, double y, double z, double _value) : value(_value)  {
      coords.push_back(x);
      coords.push_back(y);
      coords.push_back(z);
    }

    double x() const { return coords[0]; }
  };

  class Box {
  public:
    double diameter;
    int64_t ndim;
    // Store the center, start and end co-ordinates of this box. Each number
    // in corresponds to the x, y, and z co-oridinate.
    std::vector<double> center, start, end;

    std::string morton_index;

    Box() {}

    Box(double _diameter, double center_x, double start_x, double end_x, std::string _morton_index) :
      diameter(_diameter), ndim(1) {
      center.push_back(center_x);
      start.push_back(start_x);
      end.push_back(end_x);
      morton_index = _morton_index;
    }

    double distance_from(const Box& b) const {
      return std::sqrt(pow(b.center[0] - center[0], 2));
    }
  };

  // Generate a laplacian kernel assuming that each particle has unit charge.
  Matrix generate_laplacend_matrix(const std::vector<Hatrix::Particle>& particles,
                                   int64_t nrows, int64_t ncols,
                                   int64_t row_start, int64_t col_start, int64_t ndim) {
    Matrix out(nrows, ncols);

    for (int64_t i = 0; i < nrows; ++i) {
      for (int64_t j = 0; j < ncols; ++j) {
        double rij = 0;
        for (int64_t k = 0; k < ndim; ++k) {
          rij += pow(particles[i+row_start].coords[k] - particles[j+col_start].coords[k], 2);
        }
        out(i, j) = 1 / (std::sqrt(rij) + 1e-3);
      }
    }
    return out;
  }

  class BLR2 {
  private:
    // Store the dense blocks in a multimap for faster iteration without hash lookups.
    std::multimap<int64_t, Matrix> D;
    // Map of inadmissible indices.
    RowColMap<bool> is_admissible;
    // Vector of vector for storing the actual indices of all the inadmissible blocks in a given row.
    std::vector<std::vector<int64_t> > inadmissible_row_indices, admissible_row_indices;
    std::vector<std::vector<int64_t> > inadmissible_col_indices, admissible_col_indices;

    RowMap U;
    ColMap V;
    RowColMap<Matrix> S;

    int64_t N, nleaf, rank, ndim;
    double admis;
    int64_t nblocks=-1;

    void orthogonal_recursive_bisection_1dim(const std::vector<Hatrix::Particle>& particles, int64_t start,
                                             int64_t end, std::vector<double>& sorted_x,
                                             std::string morton_index, std::vector<Box>& boxes) {
      int num_points = end - start;
      // found a box with the correct number of points.
      if (num_points <= nleaf) {
        auto start_x = particles[start].coords[0];
        auto stop_x = particles[end-1].coords[0];
        auto center_x = (start_x + stop_x) / 2;
        auto diameter = stop_x - start_x;
        boxes.push_back(Box(diameter, center_x, start_x, stop_x, morton_index));
      }
      else {                    // recurse further and split again.
        int64_t middle = (start + end) / 2;
        // first half
        orthogonal_recursive_bisection_1dim(particles, start, middle, sorted_x, morton_index + "0", boxes);
        // second half
        orthogonal_recursive_bisection_1dim(particles, middle, end, sorted_x, morton_index + "1", boxes);
      }
    }

    std::vector<Box> divide_domain_and_create_particle_boxes(const std::vector<Hatrix::Particle>& particles) {
      std::vector<Box> boxes;

      if (ndim == 1) {
        // Keep a sorted
        std::vector<double> sorted_x(N);
        for (int i = 0; i < N; ++i) { sorted_x[i] = particles[i].coords[0]; }
        std::sort(sorted_x.begin(), sorted_x.end());

        int64_t start = 0;
        int64_t end = N;
        std::string morton_index = "";

        orthogonal_recursive_bisection_1dim(particles, start, end, sorted_x, morton_index, boxes);
        nblocks = boxes.size();

        // for (int64_t i = 0; i < nblocks; ++i) {
        //   auto start_x = particles[i * nleaf].x();
        //   auto stop_x = particles[i == nblocks-1 ? N-1 : (i+1) * nleaf - 1].x();
        //   auto center_x = (start_x + stop_x) / 2;
        //   auto diameter = stop_x - start_x;

        //   boxes.push_back(Box(diameter, center_x, start_x, stop_x));
        // }
      }
      else if (ndim == 3) {

      }

      return boxes;
    }

  public:
    BLR2(const std::vector<Hatrix::Particle>& particles, int64_t N, int64_t nleaf,
         int64_t rank, int64_t ndim, double admis) :
      N(N), nleaf(nleaf), rank(rank), ndim(ndim), admis(admis) {
      auto boxes = divide_domain_and_create_particle_boxes(particles);

      inadmissible_row_indices.resize(nblocks);
      admissible_row_indices.resize(nblocks);

      inadmissible_col_indices.resize(nblocks);
      admissible_col_indices.resize(nblocks);

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          is_admissible.insert(i, j, std::min(boxes[i].diameter, boxes[j].diameter) <=
                               admis * boxes[i].distance_from(boxes[j]));

          if (!is_admissible(i, j)) {
            D.insert({i, generate_laplacend_matrix(particles,
                                                   nleaf, nleaf,
                                                   i*nleaf, j*nleaf, ndim)});
          }
        }
      }

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          if (is_admissible(i, j)) {
            admissible_row_indices[i].push_back(j);
          }
          else {
            inadmissible_row_indices[i].push_back(j);
          }
        }
      }

      for (int j = 0; j < nblocks; ++j) {
        for (int i = 0; i < nblocks; ++i) {
          if (is_admissible(j, i)) {
            admissible_col_indices[j].push_back(i);
          }
          else {
            inadmissible_col_indices[j].push_back(i);
          }
        }
      }

      int64_t oversampling = 5;
      Hatrix::Matrix Utemp, Stemp, Vtemp;
      double error;
      std::vector<Hatrix::Matrix> Y;

      // Generate a bunch of random matrices.
      for (int64_t i = 0; i < nblocks; ++i) {
        Y.push_back(
                    Hatrix::generate_random_matrix(nleaf, rank + oversampling));
      }

      for (int64_t i = 0; i < nblocks; ++i) {
        Hatrix::Matrix AY(nleaf, rank + oversampling);
        for (unsigned j = 0; j < admissible_row_indices[i].size(); ++j) {
          int64_t jcol = admissible_row_indices[i][j];
          Hatrix::Matrix dense = generate_laplacend_matrix(particles,
                                                                   nleaf, nleaf,
                                                                   i*nleaf, jcol*nleaf, ndim);
          Hatrix::matmul(dense, Y[jcol], AY);
        }
        std::tie(Utemp, Stemp, Vtemp, error) = Hatrix::truncated_svd(AY, rank);
        U.insert(i, std::move(Utemp));
      }

      for (int64_t j = 0; j < nblocks; ++j) {
        Hatrix::Matrix YtA(rank + oversampling, nleaf);

        for (long unsigned int i = 0; i < admissible_col_indices[j].size(); ++i) {
          int64_t irow = admissible_col_indices[j][i];
          Hatrix::Matrix dense = Hatrix::generate_laplacend_matrix(particles,
                                                                   nleaf, nleaf,
                                                                   irow*nleaf, j*nleaf, ndim);
          Hatrix::matmul(Y[irow], dense, YtA, true);
        }
        std::tie(Utemp, Stemp, Vtemp, error) = Hatrix::truncated_svd(YtA, rank);
        V.insert(j, std::move(transpose(Vtemp)));
      }

      for (int i = 0; i < nblocks; ++i) {
        for (unsigned j = 0; j < admissible_row_indices[i].size(); ++j) {
          int64_t jcol = admissible_row_indices[i][j];
          Hatrix::Matrix dense = Hatrix::generate_laplacend_matrix(particles,
                                                                   nleaf, nleaf,
                                                                   i*nleaf, jcol*nleaf, ndim);
          S.insert(i, jcol, Hatrix::matmul(Hatrix::matmul(U[i], dense, true), V[jcol]));
        }
      }
    }

    double construction_error(const std::vector<Hatrix::Particle>& particles) {
      // Check dense blocks
      double error = 0; double dense_norm = 0;

      for (int i = 0; i < nblocks; ++i) {
        std::pair<std::multimap<int64_t, Matrix>::iterator,
                  std::multimap<int64_t, Matrix>::iterator> row_dense_blocks = D.equal_range(i);

        int j = 0;
        for (std::multimap<int64_t, Matrix>::iterator it = row_dense_blocks.first; it != row_dense_blocks.second; ++it) {
          int64_t jcol = inadmissible_row_indices[i][j];
          auto dense = generate_laplacend_matrix(particles, nleaf, nleaf,
                                                 i*nleaf, jcol*nleaf, ndim);
          dense_norm += pow(norm(dense), 2);
          error += pow(norm(it->second -  dense), 2);
          j++;
        }
      }

      for (unsigned i = 0; i < nblocks; ++i) {
        for (unsigned j = 0; j < admissible_row_indices[i].size(); ++j) {
          int64_t jcol = admissible_row_indices[i][j];
          auto dense = generate_laplacend_matrix(particles, nleaf, nleaf,
                                                 i*nleaf, jcol*nleaf, ndim);
          Matrix& Ubig = U(i);
          Matrix& Vbig = V(jcol);
          Matrix expected = matmul(matmul(Ubig, S(i, jcol)), Vbig, false, true);
          Matrix actual = generate_laplacend_matrix(particles, nleaf, nleaf,
                                                            i * nleaf, jcol * nleaf, ndim);
          error += pow(norm(expected - actual), 2);
          dense_norm += pow(norm(actual), 2);
        }
      }

      return std::sqrt(error / dense_norm);
    }

    void print_structure() {
      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          std::cout << " | " << is_admissible(i, j);
        }
        std::cout << " | \n";
      }
    }
  };
}

std::vector<Hatrix::Particle> equally_spaced_particles(int64_t ndim, int64_t N,
                                                       double min_val, double max_val) {
  std::vector<Hatrix::Particle> particles;
  double range = max_val - min_val;

  if (ndim == 1) {
    for (int64_t i = 0; i < N; ++i) {
      particles.push_back(Hatrix::Particle(i*0.4, min_val + (double(i) / double(range))));
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


  return particles;
}

int main(int argc, char** argv) {
  int64_t N = atoi(argv[1]);
  int64_t nleaf = atoi(argv[2]);
  int64_t rank = atoi(argv[3]);
  double admis = atof(argv[4]);
  int64_t ndim = atoi(argv[5]);

  Hatrix::Context::init();
  auto particles = equally_spaced_particles(ndim, N, 0.0, 1.0 * N);

  if (N % nleaf != 0) {
    std::cout << "N % nleaf != 0. Aborting.\n";
    abort();
  }

  Hatrix::BLR2 A(particles, N, nleaf, rank, ndim, admis);
  A.print_structure();
  double construct_error = A.construction_error(particles);

  Hatrix::Matrix dense = Hatrix::generate_laplacend_matrix(particles, N, N, 0, 0, 1);

  Hatrix::Context::finalize();

  std::cout << "N: " << N << " rank: " << rank << " nleaf: " << nleaf << " admis: " <<  admis
            << " construct error: " << construct_error << "\n";
}
