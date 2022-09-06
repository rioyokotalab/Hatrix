#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Hatrix {

class Cell {
 public:
  int64_t body;          // Loc of first body within Domain.bodies
  uint64_t nbodies;      // Number of bodies within Cell
  int64_t child;         // Loc of first child within Domain.cells
  uint64_t nchilds;      // Number of children
  int64_t parent;        // Loc of parent within Domain cells
  uint64_t level;        // Level in tree
  uint64_t index;        // Corresponding block index in matrix
  double center[3];      // Center coordinates of cell
  double radius[3];      // Size of the bounding box
  std::vector<uint64_t> near_list;       // Loc of cells within near interaction list
  std::vector<uint64_t> far_list;        // Loc of cells within far interaction list
  std::vector<int64_t> sample_bodies;    // Loc of sample bodies representing this cell
  std::vector<int64_t> sample_farfield;  // Loc of sample bodies representing the farfield

  Cell() : body(-1), nbodies(0), child(-1),
           nchilds(0), parent(-1), level(0), index(0) {
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

