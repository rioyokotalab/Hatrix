#pragma once
#include "Hatrix/classes/IndexedMap.h"

namespace Hatrix {
  class HSS {
  private:
    int N, rank, height;
    RowColLevelMap D, S;
    RowLevelMap U, U_generators;
    ColLevelMap V, V_generators;

    void generate_leaf_nodes();
  public:
    HSS(int _N, int _rank, int _height);
  };
} // namespace Hatrix
