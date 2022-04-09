#pragma once

#include "Hatrix/Hatrix.h"

#include "internal_types.hpp"

namespace Hatrix {
  class HSS {
  public:
    int64_t N, rank, nleaf;
    double admis;
    ColLevelMap U;
    RowLevelMap V;
    RowColLevelMap<Matrix> D, S;
    RowColLevelMap<bool> is_admissible;
  };
}
