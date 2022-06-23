#include "Particle.hpp"

#include <vector>

namespace Hatrix {
  Particle::Particle(double x, double _value) : value(_value)  {
    coords.push_back(x);
  }

  Particle::Particle(double x, double y, double _value) : value(_value) {
    coords.push_back(x);
    coords.push_back(y);
  }

  Particle::Particle(double x, double y, double z, double _value) : value(_value)  {
    coords.push_back(x);
    coords.push_back(y);
    coords.push_back(z);
  }

  Particle::Particle(std::vector<double> _coords, double _value) :
    coords(_coords), value(_value) {}

}
