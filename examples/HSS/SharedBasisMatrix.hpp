#pragma once

#include "Hatrix/Hatrix.h"

#include "Domain.hpp"
#include "internal_types.hpp"

namespace Hatrix {
  class SharedBasisMatrix {
  public:
    int64_t N, rank, nleaf;
    double admis;
    ColLevelMap U;
    RowLevelMap V;
    RowColLevelMap<Matrix> D, S;
    RowColLevelMap<bool> is_admissible;

    SharedBasisMatrix(const Domain& domain, int64_t N, int64_t nleaf, int64_t rank, double accuracy,
        const kernel_function& kernel, CONSTRUCT_ALGORITHM construct_algorithm,
        bool use_shared_basis);
  };
}
