#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <random>

#include "Hatrix/Hatrix.h"

namespace Hatrix {
  Cell::Cell(int64_t ndim) {
    center.resize(ndim);
    radii.resize(ndim);
    start_index = -1;
    end_index = -1;
    level = -1;
    radius = -1;
  }

  Cell::Cell() : start_index(-1), end_index(-1), level(-1), radius(-1) {}

  Cell::Cell(std::vector<double> _center, int64_t pstart,
             int64_t pend, double _radius) :
    center(_center), start_index(pstart), end_index(pend), radius(_radius) {}

  void
  Cell::print() const {
    std::cout << "level: " << level << std::endl;
    std::cout << "start: " << start_index << " stop: " << end_index
              << " radius: " << radius
              << " index: " << level_index
              << std::endl;
    for (int i = 0; i < cells.size(); ++i) {
      cells[i].print();
    }
  }

  int64_t
  Cell::height() const {
    if (cells.size() > 0) {
      int64_t left = cells[0].height() + 1;
      int64_t right = cells[1].height() + 1;

      return (left > right) ? left : right;
    }
    else {
      return 1;
    }
  }
}
