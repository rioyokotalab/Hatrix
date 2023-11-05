#pragma once

#include "Hatrix/Hatrix.hpp"

namespace Hatrix {
  enum ADMIS_TYPE { DUAL_TREE_TRAVERSAL, DIAGONAL_ADMIS };
  typedef struct SymmetricSharedBasisMatrix {
    int64_t min_level, max_level;
    ColLevelMap U;
    RowColLevelMap<Matrix> D, S;
    RowColLevelMap<bool> is_admissible;

    int64_t max_rank();
    void print_structure();
    int64_t leaf_dense_blocks();
    double Csp(int64_t level);

    SymmetricSharedBasisMatrix();
    SymmetricSharedBasisMatrix(const SymmetricSharedBasisMatrix& A); // deep copy

    void
    generate_admissibility(const Hatrix::Domain& domain, const bool use_nested_basis, const ADMIS_TYPE admis_algorithm);

  private:
    void actually_print_structure(int64_t level);
  } SymmetricSharedBasisMatrix;
}
