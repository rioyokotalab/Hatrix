#pragma once

#include <cstdint>
#include <vector>

namespace Hatrix {
  // A single cell that contains indices for references lists of particles
  // within the sorted domain.
  class Cell {
  public:
    std::vector<Cell> cells;     // TODO: remove this array.
    int64_t start_index;         // Index of the first particle within the cell
    int64_t end_index;           // Index after the last particle within the cell
    int64_t level;               // Level in cluster tree
    int64_t level_index;         // Local index of the cell within its level
    int64_t nchild;              // Number of children to accomodate non-binary tree
    int64_t key;                 // Key showing ordering among other cells
    Cell *child;                 // Pointer to first child node of this cell -> TODO remove this
    std::vector<double> center;  // Center coordinate of the cell
    std::vector<double> radii;   // Size of the bounding box in each direction

    Cell();
    Cell(int64_t ndim);
    Cell(std::vector<double> _center, int64_t pstart, int64_t pend, double _radius);

    // print the structure of the tree -> TODO remove this
    void print() const;
    int64_t height() const;
  };
}
