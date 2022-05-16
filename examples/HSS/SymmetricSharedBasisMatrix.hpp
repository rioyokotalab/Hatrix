#pragma once

#include "Hatrix/Hatrix.h"

#include "internal_types.hpp"

namespace Hatrix {
  typedef struct SymmetricSharedBasisMatrix {
    int64_t height;
    ColLevelMap U;
    RowColLevelMap<Matrix> D, S;
    RowColLevelMap<bool> is_admissible;
    RowColMap<int64_t> ranks;
    std::vector<int64_t> level_blocks;

    Matrix Ubig(int64_t node, int64_t level) const;
    int64_t max_rank();
  } SymmetricSharedBasisMatrix;
}
