#pragma once
#include "Hatrix/classes/IndexedMap.h"

namespace Hatrix {
  class HSS {
  public:
    int N, rank, height;
    RowColLevelMap D, S;
    RowLevelMap U;
    ColLevelMap V;
  };
} // namespace Hatrix
