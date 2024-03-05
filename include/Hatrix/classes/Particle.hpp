#pragma once

#include <vector>
#include <cstdint>

namespace Hatrix {
  // Class for storing data about a single particle that exists in space.
  class Particle {
  public:
    // Vector that stores ndim values, each corresponding to a single co-ordinate.
    std::vector<double> coords;
    double value;

    // Print the co-ordinates and value to stdout.
    void print();

    // Initialize the Particle class with default values. The coords vector will have
    // size 0.
    Particle();

    // Initialize the co-ordinates for a space of size ndim. The coords vector will be
    // allocated with a size of ndim.
    Particle (int64_t ndim);

    // Initialize a 1D particle with only the X co-ordinate and a value.
    Particle(double x, double _value);

    // Initialize a 2D particle with the X and Y co-ordinate and a value.
    Particle(double x, double y, double _value);

    // Initialize a 3D particle with the X, Y and Z co-ordinate and a value.
    Particle(double x, double y, double z, double _value);

    // Initialize an N-D particle by passing a vector of co-ordinates and a value.
    Particle(std::vector<double> coords, double _value);
  };
}
