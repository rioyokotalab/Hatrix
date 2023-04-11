#pragma once

#include <vector>

namespace Hatrix {
  class Particle {
  public:
    std::vector<double> coords;
    double value;

    void print();
    Particle(double x, double _value);
    Particle(double x, double y, double _value);
    Particle(double x, double y, double z, double _value);
    Particle(std::vector<double> coords, double _value);
  };
}
