#pragma once

#include "Hatrix/Hatrix.h"

#include "distributed/internal_types.hpp"

namespace Hatrix {
  typedef struct SymmetricSharedBasisMatrix {
    int64_t min_level, max_level;
    ColLevelMap U, US;
    RowColLevelMap<Matrix> D, S;
    RowColLevelMap<bool> is_admissible;
    RowColMap<int64_t> ranks;
    std::vector<int64_t> num_blocks;

    Matrix Ubig(int64_t node, int64_t level) const;
    int64_t max_rank();
    int64_t average_rank();
    void print_structure();
    int64_t leaf_dense_blocks();
    double Csp(int64_t level);

    SymmetricSharedBasisMatrix();
    SymmetricSharedBasisMatrix(const SymmetricSharedBasisMatrix& A); // deep copy

  private:
    void actually_print_structure(int64_t level);
  } SymmetricSharedBasisMatrix;
}
