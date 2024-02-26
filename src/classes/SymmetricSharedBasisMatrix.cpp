#include "Hatrix/Hatrix.hpp"

#include <cassert>
#include <cmath>

using namespace Hatrix;

void SymmetricSharedBasisMatrix::actually_print_structure(int64_t level) {
  int64_t nblocks = pow(2, level);
  std::cout << "LEVEL:" << level << " NBLOCKS: " << nblocks << " MIN: " << min_level << std::endl;
  for (int64_t i = 0; i < nblocks; ++i) {
    std::cout << "| " ;
    for (int64_t j = 0; j < nblocks; ++j) {
      if (is_admissible.exists(i, j, level)) {
        std::cout << is_admissible(i, j, level) << " | " ;
      }
      else {
        std::cout << "  | ";
      }
    }
    std::cout << std::endl;
  }

  std::cout << std::endl;

  if (level <= min_level) { return; }

  actually_print_structure(level-1);
}

void
SymmetricSharedBasisMatrix::print_structure() {
  actually_print_structure(max_level);
}

double
SymmetricSharedBasisMatrix::Csp(int64_t level) {
  assert(level >= min_level && level <= max_level);

  int64_t nblocks = pow(2, level);
  double avg_csp = 0;
  for (int64_t i = 0; i < nblocks; ++i) {
    int64_t i_dense = 0;
    for (int64_t j = 0; j < nblocks; ++j) {
      if (is_admissible.exists(i, j, level) && !is_admissible(i, j, level)) {
        avg_csp++;
      }
    }
  }

  return avg_csp / nblocks;
}

int64_t
SymmetricSharedBasisMatrix::leaf_dense_blocks() {
  int64_t nblocks = pow(2, max_level);
  int64_t ndense_blocks = 0;

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j <= i; ++j) {
      if (is_admissible.exists(i, j, max_level) && !is_admissible(i, j, max_level)) {
        ndense_blocks += 1;
      }
    }
  }

  return ndense_blocks;
}

SymmetricSharedBasisMatrix::SymmetricSharedBasisMatrix(const SymmetricSharedBasisMatrix& A) :
  min_level(A.min_level), max_level(A.max_level) {
  is_admissible.deep_copy(A.is_admissible);
  S.deep_copy(A.S);
  D.deep_copy(A.D);
  U.deep_copy(A.U);
}

SymmetricSharedBasisMatrix::SymmetricSharedBasisMatrix() : min_level(-1), max_level(-1) {}

void
SymmetricSharedBasisMatrix::dual_tree_traversal(const Hatrix::Domain& domain,
                                                const int64_t Ci_index,
                                                const int64_t Cj_index,
                                                const bool use_nested_basis,
                                                const double admis) {
  const Cell& Ci = domain.tree_list[Ci_index];
  const Cell& Cj = domain.tree_list[Cj_index];
  const int64_t i_level = Ci.level;
  const int64_t j_level = Cj.level;

  bool well_separated = false;
  if (i_level == j_level &&
      ((!use_nested_basis && i_level == max_level) || use_nested_basis)) {
    double distance = 0;

    for (int64_t k = 0; k < domain.ndim; ++k) {
      distance += pow(Ci.center[k] - Cj.center[k], 2);
    }
    distance = sqrt(distance);

    double ci_size = 0, cj_size = 0;
    ci_size = Ci.radius;
    cj_size = Cj.radius;
    // for (int axis = 0; axis < opts.ndim; ++axis) {
    //   ci_size += pow(Ci.radii[axis], 2);
    //   cj_size += pow(Cj.radii[axis], 2);
    // }
    // ci_size = sqrt(ci_size);
    // cj_size = sqrt(cj_size);

    if (distance >= ((ci_size + cj_size) * admis)) {
      well_separated = true;
    }

    bool val = well_separated;
    is_admissible.insert(Ci.key, Cj.key, i_level, std::move(val));
  }

  // Only descend down the tree if you are currently at a higher level and the blocks
  // at the current level are inadmissible. You then want to refine the tree further
  // since it has been found that the higher blocks are inadmissible.
  //
  // Alternatively, to create a BLR2 matrix you want to down to the finest level of granularity
  // anyway and populate the blocks at that level. So that puts another OR condition to check
  // if the use of nested basis is enabled.
  if (!well_separated || !use_nested_basis) {
    if (i_level <= j_level && Ci.nchild > 0) {
      // j is at a higher level and i is not leaf.
      const int64_t c1_index = pow(2, i_level+1) - 1 + Ci.key * 2;
      const int64_t c2_index = pow(2, i_level+1) - 1 + Ci.key * 2 + 1;
      dual_tree_traversal(domain, c1_index, Cj_index, use_nested_basis, admis);
      dual_tree_traversal(domain, c2_index, Cj_index, use_nested_basis, admis);
    }
    else if (j_level <= i_level && Cj.nchild > 0) {
      // i is at a higheer level and j is not leaf.
      const int64_t c1_index = pow(2, j_level+1) - 1 + Cj.key * 2;
      const int64_t c2_index = pow(2, j_level+1) - 1 + Cj.key * 2 + 1;
      dual_tree_traversal(domain, Ci_index, c1_index, use_nested_basis, admis);
      dual_tree_traversal(domain, Ci_index, c2_index, use_nested_basis, admis);
    }
  }
}

void
SymmetricSharedBasisMatrix::compute_matrix_structure(int64_t level, const double admis) {
  if (level == 0) { return; }
  int64_t nodes = pow(2, level);
  if (level == max_level) {
    for (int i = 0; i < nodes; ++i) {
      for (int j = 0; j < nodes; ++j) {
        is_admissible.insert(i, j, level, std::abs(i - j) > admis);
      }
    }
  }
  else {
    int64_t child_level = level + 1;
    for (int i = 0; i < nodes; ++i) {
      std::vector<int> row_children({i * 2, i * 2 + 1});
      for (int j = 0; j < nodes; ++j) {
        std::vector<int> col_children({j * 2, j * 2 + 1});

        bool admis_block = true;
        for (int c1 = 0; c1 < 2; ++c1) {
          for (int c2 = 0; c2 < 2; ++c2) {
            if (is_admissible.exists(row_children[c1], col_children[c2], child_level) &&
                !is_admissible(row_children[c1], col_children[c2], child_level)) {
              admis_block = false;
            }
          }
        }

        if (admis_block) {
          for (int c1 = 0; c1 < 2; ++c1) {
            for (int c2 = 0; c2 < 2; ++c2) {
              is_admissible.erase(row_children[c1], col_children[c2], child_level);
            }
          }
        }

        is_admissible.insert(i, j, level, std::move(admis_block));
      }
    }
  }

  compute_matrix_structure(level-1, admis);
}

void
SymmetricSharedBasisMatrix::generate_admissibility(const Hatrix::Domain& domain,
                                                   const bool use_nested_basis,
                                                   const Hatrix::ADMIS_ALGORITHM admis_algorithm,
                                                   const double admis) {
  assert(max_level != -1);
  if (admis_algorithm == Hatrix::ADMIS_ALGORITHM::DUAL_TREE_TRAVERSAL) {
    dual_tree_traversal(domain, 0, 0, use_nested_basis, admis);
    if (use_nested_basis) {
      min_level = 1;
    }
    else {
      min_level = max_level;
    }

    if (use_nested_basis && min_level == 1) {
      is_admissible.insert(0, 0, 0, false);
    }
  }
  else if (admis_algorithm == Hatrix::ADMIS_ALGORITHM::DIAGONAL_ADMIS) {
    if (use_nested_basis) {
      compute_matrix_structure(max_level, admis);
    }
    else {
      int64_t level = max_level;
      int64_t nodes = pow(2, level);
      for (int i = 0; i < nodes; ++i) {
        for (int j = 0; j < nodes; ++j) {
          is_admissible.insert(i, j, level, std::abs(i - j) > admis);
        }
      }

      // dense level for BLR2 matrix.
      level--;
      nodes = pow(2, level);
      for (int i = 0; i < nodes; ++i) {
        for (int j = 0; j < nodes; ++j) {
          is_admissible.insert(i, j, level, false);
        }
      }
    }
    min_level = -1;

    for (int64_t l = max_level; l > 0; --l) {
      int64_t nblocks = pow(2, l);
      bool all_dense = true;
      for (int64_t i = 0; i < nblocks; ++i) {
        for (int64_t j = 0; j < nblocks; ++j) {
          if ((is_admissible.exists(i, j, l) &&
               is_admissible(i, j, l)) || !is_admissible.exists(i, j, l)) {
            all_dense = false;
          }
        }
      }

      if (all_dense) {
        min_level = l;
        break;
      }
    }

    if ((max_level != min_level) && min_level != -1) { min_level++; }
    if (min_level == -1) {
      min_level = 1; // HSS matrix detected.
    }
    if (use_nested_basis && min_level == 1) {
      is_admissible.insert(0, 0, 0, false);
    }
  }
  else {
    abort();
  }
}
