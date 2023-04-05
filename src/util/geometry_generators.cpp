#include "Hatrix/util/geometry_generators.h"
#include <cmath>
#include <cstdint>
#include <random>

namespace Hatrix {
  std::vector<double>
  generate_uniform_grid(int ndim, int64_t N) {
    int64_t side = pow(N, 1.0/ndim);
    assert(pow(side, ndim) == N);

    std::vector<double> particles(N * ndim, 0);
    double space = 1.0 / side;

    if (ndim == 1) {

    }
  }
}
