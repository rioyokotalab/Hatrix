#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Hatrix {

class Cell {
 public:
  uint64_t body;         // Index of first body within Domain.bodies
  uint64_t nbodies;      // Number of bodies within Cell
  uint64_t child;        // Index of first child within Domain.cells
  uint64_t nchilds;      // Number of children
  uint64_t level;        // Level in tree
  uint64_t index;        // Corresponding block index in matrix
  double center[3];
  double radius[3];
  std::vector<uint64_t> near_siblings;
  std::vector<uint64_t> far_siblings;

  Cell() : body(0), nbodies(0), child(0),
           nchilds(0), level(0), index(0) {
    for (uint64_t axis = 0; axis < 3; axis++) {
      center[axis] = 0;
      radius[axis] = 0;
    }
  }

  bool is_leaf() const {
    return (nchilds == 0);
  }

  double get_radius() const {
    double rad = 0.;
    for (uint64_t axis = 0; axis < 3; axis++) {
      rad += radius[axis] * radius[axis];
    }
    return rad;
  }

  double get_diameter() const {
    return 4. * get_radius();
  }

  double distance_from(const Cell& other) const {
    double dist = 0;
    for (uint64_t axis = 0; axis < 3; axis++) {
      dist += (center[axis] - other.center[axis]) *
              (center[axis] - other.center[axis]);
    }
    return dist;
  }
};

} // namespace Hatrix

