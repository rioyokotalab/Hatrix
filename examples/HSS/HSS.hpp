#pragma once

#include "Hatrix/Hatrix.h"

namespace Hatrix {
  constexpr int INIT_VALUE = -1;
  enum KIND_OF_PROBLEM {LAPLACE, SQR_EXP, SINE};
  enum KIND_OF_GEOMETRY {GRID, CIRCULAR};
  enum ADMIS_KIND {DIAGONAL, GEOMETRY};
  enum CONSTRUCT_ALGORITHM {MIRO, ID_RANDOM};

  class HSS {
  public:
    int64_t N, rank;
  };
}
