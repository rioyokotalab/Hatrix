#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Hatrix/Hatrix.h"


// Construction of BLR2 strong admis matrix based on geometry based admis condition.

namespace Hatrix {
  class Particle {
  private:
    double value;
    std::vector<int64_t> coords;

  public:
    Particle(int64_t x, double _value) : value(_value)  {
      coords.push_back(x);
    }

    int64_t x() { return coords[0]; }
  };

  class Box {
  private:
    double diameter;
    int64_t ndim;
    // Store the center, start and end co-ordinates of this box. Each number
    // in corresponds to the x, y, and z co-oridinate.
    std::vector<int64_t> center, start, end;

  public:
    Box(double _diameter, int64_t center_x, int64_t start_x, int64_t end_x) : diameter(_diameter), ndim(1) {
      center.push_back(center_x);
      start.push_back(start_x);
      end.push_back(end_x);
    }
  };

  class BLR2 {
  private:
    RowColMap<bool> is_admissible;
    int64_t N, nblocks, rank, admis;

    std::vector<Box> create_particle_boxes(const std::vector<Hatrix::Particle>& particles) {
      std::vector<Box> boxes;
      int64_t nleaf = N / nblocks;
      for (int64_t i = 0; i < nblocks; ++i) {

        for (int64_t p = 0; p < nleaf; ++p) {

        }
      }
    }

  public:
    BLR2(const std::vector<Hatrix::Particle>& particles, int64_t N, int64_t nblocks, int64_t rank, int64_t admis) :
      N(N), nblocks(nblocks), rank(rank), admis(admis) {
      auto boxes = create_particle_boxes(particles);
    }

    double construction_error(const std::vector<Hatrix::Particle>& randpts) {
    }
  };
}



std::vector<Hatrix::Particle> equally_spaced_particles(int64_t ndim, int64_t N,
                                                       double min_val, double max_val) {
  std::vector<Hatrix::Particle> particles;
  double range = max_val - min_val;

  for (int64_t i = 0; i < N; ++i) {
    particles.push_back(Hatrix::Particle(i, min_val + (double(i) / double(range))));
  }

  return particles;
}

int main(int argc, char** argv) {
  int64_t N = atoi(argv[1]);
  int64_t nblocks = atoi(argv[2]);
  int64_t rank = atoi(argv[3]);
  int64_t admis = atoi(argv[4]);

  Hatrix::Context::init();
  auto particles = equally_spaced_particles(1, N, 0.0, 1.0 * N);
  // randvec_t randpts;
  // randpts.push_back(equally_spaced_vector(N, 0.0, 1.0 * N)); // 1D
  // randpts.push_back(equally_spaced_vector(N, 0.0, 1.0 * N)); // 2D
  // randpts.push_back(equally_spaced_vector(N, 0.0, 1.0 * N)); // 3D

  if (N % nblocks != 0) {
    std::cout << "N % nblocks != 0. Aborting.\n";
    abort();
  }

  Hatrix::BLR2 A(particles, N, nblocks, rank, admis);
  double construct_error = A.construction_error(particles);

  Hatrix::Context::finalize();

  std::cout << "N: " << N << " rank: " << rank << " nblocks: " << nblocks << " admis: " <<  admis
            << " construct error: " << construct_error << "\n";
}
