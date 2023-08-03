#pragma once

#include <vector>
#include <cstdint>

namespace Hatrix {
  class Particle {
  public:
    std::vector<double> coords;
    double value;

    void print();
    Particle (int64_t ndim);
    Particle(double x, double _value);
    Particle(double x, double y, double _value);
    Particle(double x, double y, double z, double _value);
    Particle(std::vector<double> coords, double _value);
  };
}
