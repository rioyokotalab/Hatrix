#pragma once
#include "Hatrix/classes/IndexedMap.h"

namespace Hatrix {
  class HSS {
  public:
    RowColLevelMap D, S;
    RowLevelMap U;
    ColLevelMap V;
  };
} // namespace Hatrix
