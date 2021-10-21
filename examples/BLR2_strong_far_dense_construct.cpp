#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

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

    double x() const { return coords[0]; }
  };

  class Box {
  public:
    double diameter;
    int64_t ndim;
    // Store the center, start and end co-ordinates of this box. Each number
    // in corresponds to the x, y, and z co-oridinate.
    std::vector<double> center, start, end;

    Box(double _diameter, double center_x, double start_x, double end_x) :
      diameter(_diameter), ndim(1) {
      center.push_back(center_x);
      start.push_back(start_x);
      end.push_back(end_x);
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
          rij += pow(particles[i].coords[k] - particles[j].coords[k], 2);
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
    std::vector<std::vector<int64_t> > inadmissible_indices;
    int64_t N, nblocks, rank, ndim;
    double admis;

    std::vector<Box> create_particle_boxes(const std::vector<Hatrix::Particle>& particles) {
      std::vector<Box> boxes;
      int64_t nleaf = N / nblocks;
      for (int64_t i = 0; i < nblocks; ++i) {
        auto start_x = particles[i * nleaf].x();
        auto stop_x = particles[i == nblocks-1 ? N-1 : (i+1) * nleaf - 1].x();
        auto center_x = (start_x + stop_x) / 2;
        auto diameter = stop_x - start_x;

        boxes.push_back(Box(diameter, center_x, start_x, stop_x));
      }

      return boxes;
    }

  public:
    BLR2(const std::vector<Hatrix::Particle>& particles, int64_t N, int64_t nblocks,
         int64_t rank, double admis, int64_t ndim) :
      N(N), nblocks(nblocks), rank(rank), admis(admis), ndim(ndim) {
      int64_t block_size = N / nblocks;
      inadmissible_indices.resize(nblocks);
      auto boxes = create_particle_boxes(particles);

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          is_admissible.insert(i, j, std::min(boxes[i].diameter, boxes[j].diameter) <=
                               admis * boxes[i].distance_from(boxes[j]));

          if (!is_admissible(i, j)) {
            inadmissible_indices[i].push_back(j);
            D.insert({i, generate_laplacend_matrix(particles,
                                                   block_size, block_size,
                                                   i*block_size, j*block_size, ndim)});
          }
        }
      }
    }

    double construction_error(const std::vector<Hatrix::Particle>& particles) {
      // Check dense blocks
      double error = 0; double dense_norm = 0;
      int64_t block_size = N / nblocks;

      for (int i = 0; i < nblocks; ++i) {
        std::pair<std::multimap<int64_t, Matrix>::iterator,
                  std::multimap<int64_t, Matrix>::iterator> row_dense_blocks = D.equal_range(i);

        int j = 0;
        for (std::multimap<int64_t, Matrix>::iterator it = row_dense_blocks.first; it != row_dense_blocks.second; ++it) {
          int64_t icol = inadmissible_indices[i][j];
          auto dense = generate_laplacend_matrix(particles,
                                                 block_size, block_size,
                                                 i*block_size, icol*block_size, ndim);
          dense_norm += pow(norm(dense), 2);
          error += pow(norm(it->second -  dense), 2);

          j++;
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

  for (int64_t i = 0; i < N; ++i) {
    particles.push_back(Hatrix::Particle(i*0.4, min_val + (double(i) / double(range))));
  }

  return particles;
}

int main(int argc, char** argv) {
  int64_t N = atoi(argv[1]);
  int64_t nblocks = atoi(argv[2]);
  int64_t rank = atoi(argv[3]);
  double admis = atof(argv[4]);
  int64_t ndim = 1;

  Hatrix::Context::init();
  auto particles = equally_spaced_particles(ndim, N, 0.0, 1.0 * N);

  if (N % nblocks != 0) {
    std::cout << "N % nblocks != 0. Aborting.\n";
    abort();
  }

  Hatrix::BLR2 A(particles, N, nblocks, rank, admis, ndim);
  A.print_structure();
  double construct_error = A.construction_error(particles);

  Hatrix::Matrix dense = Hatrix::generate_laplacend_matrix(particles, N, N, 0, 0, 1);

  Hatrix::Context::finalize();

  std::cout << "N: " << N << " rank: " << rank << " nblocks: " << nblocks << " admis: " <<  admis
            << " construct error: " << construct_error << "\n";
}
