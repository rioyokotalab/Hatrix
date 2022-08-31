#pragma once

#include <cstdint>
#include <vector>

namespace Hatrix {

class Body {
 public:
  double value;
  double X[3]; // Position

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
    const auto ndim = std::min((uint64_t)3, (uint64_t)_X.size());
    for (uint64_t axis = 0; axis < ndim; axis++) {
      X[axis] = _X[axis];
    }
  }

};

} // namespace Hatrix

