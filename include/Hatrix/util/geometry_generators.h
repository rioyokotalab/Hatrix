#pragma once
#include <cmath>
#include <cstdint>
#include <vector>

#include "Hatrix/classes/Matrix.h"

namespace Hatrix {
  // Return the particles as a vector of length N for each dimension.
  // So a 3D grid would have the form [[x0,x1...]*N,[y0,y1...]*N,[z0,z1...]*N].
  // Note that this method will fail if N is a not a perfect power of ndim.
  std::vector<double> generate_uniform_grid(int ndim, int64_t N);

  std::vector<double> generate_uniform_sphere(int ndim, int64_t N);
}
