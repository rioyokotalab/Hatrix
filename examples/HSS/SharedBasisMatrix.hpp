#pragma once

#include "Hatrix/Hatrix.h"

#include "Domain.hpp"
#include "internal_types.hpp"

namespace Hatrix {
  class SharedBasisMatrix {
  private:
    int64_t N, nleaf, rank;
    double accuracy;
    double admis;
    ADMIS_KIND admis_kind;
    CONSTRUCT_ALGORITHM construct_algorithm;
    bool use_shared_basis;
    int64_t height;

    void coarsen_blocks(int64_t level);
    void calc_diagonal_based_admissibility(int64_t level);
  public:
    ColLevelMap U;
    RowLevelMap V;
    RowColLevelMap<Matrix> D, S;
    RowColLevelMap<bool> is_admissible;
    std::vector<int64_t> level_blocks;

    SharedBasisMatrix(int64_t N, int64_t nleaf, int64_t rank, double accuracy,
                      double admis, ADMIS_KIND admis_kind,
                      CONSTRUCT_ALGORITHM construct_algorithm, bool use_shared_basis,
                      const Domain& domain, const kernel_function& kernel);
  };
}
