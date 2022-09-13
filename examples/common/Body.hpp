#pragma once

#include <cstdint>
#include <vector>

namespace Hatrix {

constexpr int64_t MAX_NDIM = 3;

class Body {
 public:
  double value;
  double X[MAX_NDIM];  // Position

  Body() {
    for (int64_t axis = 0; axis < MAX_NDIM; axis++) {
      X[axis] = 0;
    }
  }

  Body(const double x, const double _value)
      : value(_value) {
    X[0] = x;
    X[1] = 0;
    X[2] = 0;
  }

  Body(const double x, const double y, const double _value)
      : value(_value) {
    X[0] = x;
    X[1] = y;
    X[2] = 0;
  }

  Body(const double x, const double y, const double z,
       const double _value) : value(_value) {
    X[0] = x;
    X[1] = y;
    X[2] = z;
  }

  Body(const std::vector<double>& _X, const double _value)
      : value(_value) {
    X[0] = 0;
    X[1] = 0;
    X[2] = 0;
    const auto ndim = std::min(MAX_NDIM, (int64_t)_X.size());
    for (int64_t axis = 0; axis < ndim; axis++) {
      X[axis] = _X[axis];
    }
  }

};

} // namespace Hatrix

