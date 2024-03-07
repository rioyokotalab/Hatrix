#pragma once

#include <cassert>
#include <cmath>
#include <vector>

#include "Hatrix/Hatrix.hpp"
#include "distributed/distributed.hpp"
#include "factorize_noparsec.hpp"

using namespace Hatrix;

void
factorize_level_noparsec(SymmetricSharedBasisMatrix& A,
                         const Domain& domain,
                         const Args& opts, const int64_t level) {
  int64_t nblocks = std::pow(2, level);

  for (int64_t node = 0; node < nblocks; ++node) {
    auto U_F = make_complement(A.U(node, level));
    A.D(node, node, level) = matmul(matmul(U_F, A.D(node, node, level), true), U_F);

    Matrix& D_node = A.D(node, node, level);
    const int64_t node_c_size = D_node.rows - opts.max_rank;
    auto D_node_splits = D_node.split(std::vector<int64_t>{node_c_size},
                                      std::vector<int64_t>{node_c_size});

    Matrix& D_node_cc = D_node_splits[0];
    ldl(D_node_cc);

    Matrix& D_node_oc = D_node_splits[2];
    solve_triangular(D_node_cc, D_node_oc, Hatrix::Right, Hatrix::Lower, true, true);
    solve_diagonal(D_node_cc, D_node_oc, Hatrix::Right);

    Matrix& D_node_co = D_node_splits[1];
    solve_triangular(D_node_cc, D_node_co, Hatrix::Left, Hatrix::Lower, true, false);
    solve_diagonal(D_node_cc, D_node_co, Hatrix::Left);

    Matrix D_node_oc_copy(D_node_oc, true);  // Deep-copy
    Matrix& D_node_oo = D_node_splits[3];
    column_scale(D_node_oc_copy, D_node_cc);  // LD
    matmul(D_node_oc_copy, D_node_co, D_node_oo, false, false, -1, 1);  // LDL^T
  }
}


// This routine is not designed for multi-process.
void
factorize_noparsec(SymmetricSharedBasisMatrix& A,
                   const Domain& domain,
                   const Args& opts) {
  assert(MPISIZE == 1);
  for (int64_t level = A.max_level; level >= A.min_level; --level) {
    factorize_level_noparsec(A, domain, opts, level);

    // Merge unfactorized parts
    const int64_t parent_level = level - 1;
    const int64_t parent_nblocks = pow(2, parent_level);

    for (int64_t i = 0; i < parent_nblocks; ++i) {
      for (int64_t j = 0; j < parent_nblocks; ++j) {
        if (A.is_admissible.exists(i, j, parent_level) &&
            !A.is_admissible(i, j, parent_level)) {

          std::vector<int64_t> i_children, j_children;
          std::vector<int64_t> row_split, col_split;
          int64_t nrows=0, ncols=0;

          for (int64_t n = 0; n < 2; ++n) {
            int64_t ic = i * 2 + n;
            int64_t jc = j * 2 + n;
            i_children.push_back(ic);
            j_children.push_back(jc);

            nrows += opts.max_rank;
            ncols += opts.max_rank;
            if (n < 1) {
              row_split.push_back(nrows);
              col_split.push_back(ncols);
            }
          }

          Matrix& D_unelim = A.D(i, j, parent_level);
          auto D_unelim_splits = D_unelim.split(row_split, col_split);

          for (int64_t ic1 = 0; ic1 < i_children.size(); ++ic1) {
            for (int64_t jc2 = 0; jc2 < j_children.size(); ++jc2) {
              int64_t c1 = i_children[ic1], c2 = j_children[jc2];
              if (!A.U.exists(c1, level)) { continue; }

              if (A.is_admissible.exists(c1, c2, level) &&
                  !A.is_admissible(c1, c2, level)) {
                auto D_splits =
                  A.D(c1, c2, level).split(
                                           std::vector<int64_t>{A.D(c1, c2, level).rows -
                                             opts.max_rank},
                                           std::vector<int64_t>{A.D(c1, c2, level).cols -
                                             opts.max_rank});
                D_unelim_splits[ic1 * j_children.size() + jc2] = D_splits[3];
              }
              else {
                D_unelim_splits[ic1 * j_children.size() + jc2] = A.S(c1, c2, level);
              }
            }
          }
        }
      }
    }
  }

  // Factorize the remaining blocks as a block dense matrix.
  int64_t last_level = A.min_level - 1;
  int64_t nblocks = pow(2, last_level);
  for (int64_t k = 0; k < nblocks; ++k) {
    ldl(A.D(k, k, last_level));

    for (int64_t i = k + 1; i < nblocks; ++i) {
      solve_triangular(A.D(k, k, last_level),
                       A.D(i, k, last_level), Hatrix::Right, Hatrix::Lower, true, true);
      solve_diagonal(A.D(k, k, last_level), A.D(i, k, last_level), Hatrix::Right);
    }

    for (int64_t j = k + 1; j < nblocks; ++j) {
      solve_triangular(A.D(k, k, last_level),
                       A.D(k, j, last_level), Hatrix::Left, Hatrix::Lower, true, false);
      solve_diagonal(A.D(k, k, last_level), A.D(k, j, last_level), Hatrix::Left);
    }

    for (int64_t i = k + 1; i < nblocks; ++i) {
      for (int64_t j = k + 1; j < nblocks; ++j) {
        Matrix Dik(A.D(i, k, last_level), true);  // Deep-copy
        column_scale(Dik, A.D(k, k, last_level));  // LD
        matmul(Dik, A.D(k, j, last_level), A.D(i, j, last_level), false, false, -1, 1);
      }
    }
  }
}
