#pragma once

#include "Hatrix/Hatrix.h"

#include "Domain.hpp"
#include "internal_types.hpp"

namespace Hatrix {

  class SharedBasisMatrix {
  private:
    void coarsen_blocks(int64_t level);
    void calc_diagonal_based_admissibility(int64_t level);
    Matrix get_Ubig(int64_t node, int64_t level);
    Matrix get_Vbig(int64_t node, int64_t level);
  public:
    int64_t N, nleaf, rank;
    double accuracy;
    double admis;
    ADMIS_KIND admis_kind;
    CONSTRUCT_ALGORITHM construct_algorithm;
    bool use_shared_basis;
    const Domain domain;
    const kernel_function kernel;
    const bool is_symmetric;

    int64_t height;
    ColLevelMap U;
    RowLevelMap V;
    RowColLevelMap<Matrix> D, S;
    RowColLevelMap<bool> is_admissible;
    std::vector<int64_t> level_blocks;

    int64_t get_block_size(int64_t parent, int64_t level);
    SharedBasisMatrix(int64_t N, int64_t nleaf, int64_t rank, double accuracy,
                      double admis, ADMIS_KIND admis_kind,
                      CONSTRUCT_ALGORITHM construct_algorithm, bool use_shared_basis,
                      const Domain& domain, const kernel_function& kernel,
                      const bool is_symmetric);

    // Obtain construction error w.r.t dense matrix with matvec.
    double construction_error();
    Matrix matvec(const Matrix& x);
  };

  // Strategy pattern class hierarchy for implementation of various
  // construction schemes.
  class ConstructAlgorithm {
  public:
    SharedBasisMatrix *context;
    ConstructAlgorithm(SharedBasisMatrix* context);
    virtual void construct() = 0;
  };
}
