#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "Body.hpp"

namespace Hatrix {

class Cell {
 public:
  int64_t body_offset;                   // Index of first body within Domain.bodies
  int64_t nbodies;                       // Number of bodies within Cell
  int64_t child;                         // Index of first child within Domain.cells
  int64_t nchilds;                       // Number of children
  int64_t parent;                        // Index of parent within Domain cells
  int64_t level;                         // Level in cluster tree
  int64_t block_index;                   // Corresponding block index within matrix partition
  double center[MAX_NDIM];               // Center coordinates of cell
  double radius[MAX_NDIM];               // Size of the bounding box
  std::vector<int64_t> near_list;        // Index of cells in near interaction list
  std::vector<int64_t> far_list;         // Index of cells in far interaction list
  std::vector<int64_t> sample_bodies;    // Index of sample bodies representing this cell
  std::vector<int64_t> sample_farfield;  // Index of sample bodies representing the farfield

  Cell() : body_offset(-1), nbodies(0), child(-1),
           nchilds(0), parent(-1), level(0), block_index(0) {
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

  std::vector<int64_t> get_bodies() const {
    std::vector<int64_t> bodies;
    bodies.reserve(nbodies);
    for (int64_t i = 0; i < nbodies; i++) {
      bodies.push_back(body_offset + i);
    }
    return bodies;
  }
};

} // namespace Hatrix
