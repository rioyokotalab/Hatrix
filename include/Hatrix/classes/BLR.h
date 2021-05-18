#pragma once
#include "Hatrix/classes/IndexedMap.h"


namespace Hatrix {

class BLR {
 public:
  RowColMap D, S;
  RowMap U;
  ColMap V;
};

} // namespace Hatrix

