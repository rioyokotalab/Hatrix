#pragma once
#include "Hatrix/classes/IndexedMap.h"

namespace Hatrix {

template <typename DT>
class BLR {
 public:
  RowColMap<bool> is_admissible;
  RowColMap<Matrix<DT>> D, S;
  RowMap<DT> U;
  ColMap<DT> V;
};

}  // namespace Hatrix
