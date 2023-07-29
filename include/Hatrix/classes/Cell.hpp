#pragma once

#include <vector>
#include "Particle.hpp"
#include "Hatrix/Hatrix.h"

namespace Hatrix {
  // A single cell that contains indices for references lists of particles
  // within the sorted domain.
  class Cell {
  public:
    std::vector<Cell> cells;
    std::vector<double> center;
    std::vector<double> radii;
    int64_t start_index, end_index, level;
    uint32_t level_index;
    double radius;

    Cell(std::vector<double> _center, int64_t pstart, int64_t pend, double _radius);
    Cell(int64_t ndim);
    Cell();

    // print the structure of the tree.
    void print() const;
    int64_t height() const;
  };
}
