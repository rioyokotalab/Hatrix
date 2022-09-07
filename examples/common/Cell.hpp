#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "Body.hpp"

namespace Hatrix {

class Cell {
 public:
  int64_t body;                          // Loc of first body within Domain.bodies
  int64_t nbodies;                       // Number of bodies within Cell
  int64_t child;                         // Loc of first child within Domain.cells
  int64_t nchilds;                       // Number of children
  int64_t parent;                        // Loc of parent within Domain cells
  int64_t level;                         // Level in tree
  int64_t index;                         // Corresponding block index in matrix
  double center[MAX_NDIM];               // Center coordinates of cell
  double radius[MAX_NDIM];               // Size of the bounding box
  std::vector<int64_t> near_list;        // Loc of cells within near interaction list
  std::vector<int64_t> far_list;         // Loc of cells within far interaction list
  std::vector<int64_t> sample_bodies;    // Loc of sample bodies representing this cell
  std::vector<int64_t> sample_farfield;  // Loc of sample bodies representing the farfield

  Cell() : body(-1), nbodies(0), child(-1),
           nchilds(0), parent(-1), level(0), index(0) {
    for (int64_t axis = 0; axis < MAX_NDIM; axis++) {
      center[axis] = 0;
      radius[axis] = 0;
    }
  }

  bool is_leaf() const {
    return (nchilds == 0);
  }

  double get_radius() const {
    double rad = 0.;
    for (int64_t axis = 0; axis < MAX_NDIM; axis++) {
      rad += radius[axis] * radius[axis];
    }
    return rad;
  }

  double get_diameter() const {
    return 4. * get_radius();
  }

  double distance_from(const Cell& other) const {
    double dist = 0;
    for (int64_t axis = 0; axis < MAX_NDIM; axis++) {
      dist += (center[axis] - other.center[axis]) *
              (center[axis] - other.center[axis]);
    }
    return dist;
  }
};

} // namespace Hatrix

