#pragma once

#include <vector>

namespace Hatrix {

class Particle {
 public:
  double value;
  std::vector<double> coords;

  Particle(const double x, const double _value)
      : value(_value) {
    coords.push_back(x);
  }

  Particle(const double x, const double y, const double _value)
      : value(_value) {
    coords.push_back(x);
    coords.push_back(y);
  }

  Particle(const double x, const double y, const double z,
           const double _value) : value(_value) {
    coords.push_back(x);
    coords.push_back(y);
    coords.push_back(z);
  }

  Particle(const std::vector<double>& _coords, const double _value)
      : coords(_coords), value(_value) {}
};

} // namespace Hatrix

