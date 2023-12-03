#pragma once

#include "Domain.hpp"

namespace Hatrix {
  enum ADMIS_ALGORITHM { DUAL_TREE_TRAVERSAL, DIAGONAL_ADMIS };
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
    generate_admissibility(const Domain& domain, const bool use_nested_basis,
                           const ADMIS_ALGORITHM admis_algorithm,
                           const double admis);

  private:
    void actually_print_structure(int64_t level);
    void dual_tree_traversal(const Domain& domain,
                             const int64_t Ci_index,
                             const int64_t Cj_index,
                             const bool use_nested_basis,
                             const double admis);

  } SymmetricSharedBasisMatrix;
}
