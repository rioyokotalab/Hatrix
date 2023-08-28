#pragma once
#include "Hatrix/classes/IndexedMap.hpp"

namespace Hatrix {

class BLR {
 public:
  RowColMap<bool> is_admissible;
  RowColMap<Matrix> D, S;
  RowMap<Matrix> U;
  ColMap<Matrix> V;
};

}  // namespace Hatrix
