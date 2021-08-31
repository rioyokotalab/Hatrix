#pragma once
#include "Hatrix/classes/IndexedMap.h"

namespace Hatrix {

class BLR {
 public:
  RowColMap<bool> is_admissible;
  RowColMap<Matrix> D, S;
  RowMap U;
  ColMap V;
};

}  // namespace Hatrix
