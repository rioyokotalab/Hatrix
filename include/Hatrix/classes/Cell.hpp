#pragma once

#include <vector>
#include "Particle.hpp"

namespace Hatrix {
  // A single cell that contains references lists of particles within the sorted
  // domain. Can be used as a node in a tree in order to split the domain.
  class Cell {
  public:
    std::vector<Cell> cells;    // TODO: remove this array.
    // The co-ordinates of the center of this cell.
    std::vector<double> center;

    // Array storing the maximum distance between points in each dimension.
    std::vector<double> radii;

    // The index in the array of sorted particles that corresponds to the first particle
    // in the cell.
    int64_t start_index;

    // The index in the array of sorted particles that corresponds to the last particle
    // in the cell.
    int64_t end_index;

    // The level of this cell in the tree of sorted particles.
    int64_t level;

    // The index in the array of sorted particles that corresponds to the start of a new
    // level in the tree.
    uint32_t level_index;

    // The longest dimension from radii.
    double radius;
    int64_t nchild;             // number of children to accomodate non-binary tree.
    int64_t key;                // Key showing ordering among other cells.
    Cell *child;                // Pointer to first child node of this cell.

    Cell(std::vector<double> _center, int64_t pstart, int64_t pend, double _radius);
    Cell(int64_t ndim);
    Cell();

    // print the structure of the tree.
    void print() const;
    // Return the height of the tree whose head is this cell.
    int64_t height() const;
  };
}
