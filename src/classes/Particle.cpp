#include "Hatrix/classes/Particle.hpp"
#include <vector>
#include <iostream>

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

  void Particle::print() {
    std::cout << "<";
    for (int64_t i = 0; i < coords.size(); ++i) {
      std::cout << coords[i];
      if (i < coords.size()-1) {
        std::cout << ",";
      }
    }
    std::cout << ">";
  }
}
