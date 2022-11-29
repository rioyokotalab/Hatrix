#include <algorithm>
#include <exception>
#include <cmath>
#include <random>

#include "Hatrix/Hatrix.h"
#include "franklin/franklin.hpp"

#include "h2_operations.hpp"

using namespace Hatrix;

Hatrix::RowLevelMap US;

void
factorize_diagonal(SymmetricSharedBasisMatrix& A, int64_t block, int64_t level) {
  Matrix& diagonal = A.D(block, block, level);
  auto diagonal_splits = split_dense(diagonal,
                                     diagonal.rows - A.ranks(block, level),
                                     diagonal.cols - A.ranks(block, level));
  cholesky(diagonal_splits[0], Hatrix::Lower);
}

void right_lower_triangle_reduce(SymmetricSharedBasisMatrix& A,
                                 const Matrix& Dcc,
                                 const int64_t i,
                                 const int64_t block,
                                 const int64_t level,
                                 const int64_t split_index) {
  if (exists_and_inadmissible(A, i, block, level)) {
    auto D_i_block_splits = split_dense(A.D(i, block, level),
                                        A.D(i, block, level).rows - A.ranks(i, level),
                                        A.D(i, block, level).cols - A.ranks(block, level));

    solve_triangular(Dcc, D_i_block_splits[split_index], Hatrix::Right, Hatrix::Lower,
                     false, true, 1.0);
  }
}

void triangle_reduction(SymmetricSharedBasisMatrix& A, int64_t block, int64_t level) {
  Matrix& diagonal = A.D(block, block, level);
  auto diagonal_splits = split_dense(diagonal,
                                     diagonal.rows - A.ranks(block, level),
                                     diagonal.cols - A.ranks(block, level));
  Matrix& Dcc = diagonal_splits[0];

  int64_t nblocks = pow(2, level);
  // TRSM with oc blocks along the 'block' column.
  for (int64_t i = block; i < nblocks; ++i) {
    right_lower_triangle_reduce(A, Dcc, i, block, level, 2);
  }

  // TRSM with cc blocks along the 'block' column after the diagonal block.
  for (int64_t i = block+1; i < nblocks; ++i) {
    right_lower_triangle_reduce(A, Dcc, i, block, level, 0);
  }

  // TRSM with co blocks behind the diagonal on the 'block' row.
  for (int64_t j = 0; j < block; ++j) {
    if (exists_and_inadmissible(A, block, j, level)) {
      auto D_block_j_splits = split_dense(A.D(block, j, level),
                                          A.D(block, j, level).rows - A.ranks(block, level),
                                          A.D(block, j, level).cols - A.ranks(j, level));

      solve_triangular(Dcc, D_block_j_splits[1], Hatrix::Left, Hatrix::Lower,
                       false, false, 1.0);
    }
  }
}

// 1. Schur's complement for cc blocks. cc = cc - cc * cc.T. All these blocks are in the
// 'block' column. This is not useful for diagonal strong admis.
template<typename T> void
reduction_loop1(SymmetricSharedBasisMatrix& A, int64_t block, int64_t level, T&& body) {
  int64_t nblocks = pow(2, level);
  for (int64_t i = block+1; i < nblocks; ++i) {
    if (exists_and_inadmissible(A, i, block, level)) {
      Matrix& D_i_block = A.D(i, block, level);
      auto D_i_block_splits = split_dense(D_i_block,
                                          D_i_block.rows - A.ranks(i, level),
                                          D_i_block.cols - A.ranks(block, level));

      for (int64_t j = block+1; j <= i; ++j) {
        if (exists_and_inadmissible(A, j, block, level)) {
          Matrix& D_j_block = A.D(j, block, level);
          auto D_j_block_splits = split_dense(D_j_block,
                                              D_j_block.rows - A.ranks(j, level),
                                              D_j_block.cols - A.ranks(block, level));

          body(i, j, D_i_block_splits, D_j_block_splits);
        }
      }
    }
  }
}

// 2. Schur's complements within the oc blocks and into oo
template<typename T> void
reduction_loop2(SymmetricSharedBasisMatrix& A, int64_t block, int64_t level, T&& body) {
  int64_t nblocks = pow(2, level);

  for (int64_t i = block; i < nblocks; ++i) {
    if (exists_and_inadmissible(A, i, block, level)) {
      Matrix& D_i_block = A.D(i, block, level);
      auto D_i_block_splits = split_dense(D_i_block,
                                          D_i_block.rows - A.ranks(i, level),
                                          D_i_block.cols - A.ranks(block, level));
      for (int64_t j = block; j <= i; ++j) {
        if (exists_and_inadmissible(A, j, block, level)) {
          Matrix& D_j_block = A.D(j, block, level);
          auto D_j_block_splits = split_dense(D_j_block,
                                              D_j_block.rows - A.ranks(j, level),
                                              D_j_block.cols - A.ranks(block, level));

          body(i, j, D_i_block_splits, D_j_block_splits);
        }
      }
    }
  }
}

// 3. Schur's complements between cc and oc blocks. oc = oc - oc * cc.T
// This only considers the blocks below the diagonal block.
template<typename T> void
reduction_loop3(SymmetricSharedBasisMatrix& A, int64_t block, int64_t level, T&& body) {
  int64_t nblocks = pow(2, level);
  for (int64_t i = block+1; i < nblocks; ++i) {
    if (exists_and_inadmissible(A, i, block, level)) {
      Matrix& D_i_block = A.D(i, block, level);
      auto D_i_block_splits = split_dense(D_i_block,
                                          D_i_block.rows - A.ranks(i, level),
                                          D_i_block.cols - A.ranks(block, level));
      for (int64_t j = block+1; j < nblocks; ++j) {
        if (exists_and_inadmissible(A, j, block, level)) {
          Matrix& D_j_block = A.D(j, block, level);
          auto D_j_block_splits = split_dense(D_j_block,
                                              D_j_block.rows - A.ranks(j, level),
                                              D_j_block.cols - A.ranks(block, level));

          body(i, j, D_i_block_splits, D_j_block_splits);
        }
      }
    }
  }
}

// 4. Between cc and co blocks. The product is expressed as a transposed co block.
template<typename T> void
reduction_loop4(SymmetricSharedBasisMatrix& A, int64_t block, int64_t level, T&& body) {
  int64_t nblocks = pow(2, level);
  for (int64_t i = block+1; i < nblocks; ++i) {
    if (exists_and_inadmissible(A, i, block, level)) {
      Matrix& D_i_block = A.D(i, block, level);
      auto D_i_block_splits = split_dense(D_i_block,
                                          D_i_block.rows - A.ranks(i, level),
                                          D_i_block.cols - A.ranks(block, level));

      for (int64_t j = 0; j <= block; ++j) {
        if (exists_and_inadmissible(A, block, j, level)) {
          Matrix& D_block_j = A.D(block, j, level);
          auto D_block_j_splits = split_dense(D_block_j,
                                              D_block_j.rows - A.ranks(block, level),
                                              D_block_j.cols - A.ranks(j, level));
          if (exists_and_inadmissible(A, i, j, level)) {
            Matrix& D_ij = A.D(i, j, level);
            auto D_ij_splits = split_dense(D_ij,
                                           D_ij.rows - A.ranks(i, level),
                                           D_ij.cols - A.ranks(j, level));
            matmul(D_i_block_splits[0], D_block_j_splits[1], D_ij_splits[1],
                   false, false, -1.0, 1.0);
          }
        }
      }
    }
  }
}

// 5. Between oc & co -> oo.
template<typename T> void
reduction_loop5(SymmetricSharedBasisMatrix& A, int64_t block, int64_t level, T&& body) {
  int64_t nblocks = pow(2, level);
  for (int64_t i = block; i < nblocks; ++i) {
    if (exists_and_inadmissible(A, i, block, level)) {
      Matrix& D_i_block = A.D(i, block, level);
      auto D_i_block_splits = split_dense(D_i_block,
                                          D_i_block.rows - A.ranks(i, level),
                                          D_i_block.cols - A.ranks(block, level));
      for (int64_t j = 0; j < block; ++j) {
        if (exists_and_inadmissible(A, block, j, level)) {
          Matrix& D_block_j = A.D(block, j, level);
          auto D_block_j_splits = split_dense(D_block_j,
                                              D_block_j.rows - A.ranks(block, level),
                                              D_block_j.cols - A.ranks(j, level));

          body(i, j, D_i_block_splits, D_block_j_splits);

        }
      }
    }
  }
}

void compute_schurs_complement(SymmetricSharedBasisMatrix& A, int64_t block, int64_t level) {
  reduction_loop1(A, block, level,
                  [&](int64_t i, int64_t j, std::vector<Matrix>& D_i_block_splits,
                      std::vector<Matrix>& D_j_block_splits) {
                    if (exists_and_inadmissible(A, i, j, level)) {
                      Matrix& D_ij = A.D(i, j, level);
                      auto D_ij_splits = split_dense(D_ij,
                                                     D_ij.rows - A.ranks(i, level),
                                                     D_ij.cols - A.ranks(j, level));

                      if (i == j) {
                        syrk(D_i_block_splits[0], D_ij_splits[0], Hatrix::Lower, false, -1.0, 1.0);
                      }
                      else {
                        matmul(D_i_block_splits[0], D_j_block_splits[0], D_ij_splits[0],
                               false, true, -1.0, 1.0);
                      }
                    }
                  });

  reduction_loop2(A, block, level,
                  [&](int64_t i, int64_t j, std::vector<Matrix>& D_i_block_splits,
                      std::vector<Matrix>& D_j_block_splits) {
                    if (exists_and_inadmissible(A, i, j, level)) {
                      Matrix& D_ij = A.D(i, j, level);
                      auto D_ij_splits = split_dense(D_ij,
                                                     D_ij.rows - A.ranks(i, level),
                                                     D_ij.cols - A.ranks(j, level));

                      if (i == j) {
                        syrk(D_i_block_splits[2], D_ij_splits[3], Hatrix::Lower, false, -1.0, 1.0);
                      }
                      else {
                        matmul(D_i_block_splits[2], D_j_block_splits[2], D_ij_splits[3], false, true,
                               -1.0, 1.0);
                      }
                    }
                  });

  reduction_loop3(A, block, level,
                  [&](int64_t i, int64_t j, std::vector<Matrix>& D_i_block_splits,
                      std::vector<Matrix>& D_j_block_splits) {
                    if (i < j) {  // update oc block in the true lower triangle
                      if (exists_and_inadmissible(A, j, i, level)) {
                        Matrix& D_ji = A.D(j, i, level);
                        auto D_ji_splits = split_dense(D_ji,
                                                       D_ji.rows - A.ranks(j, level),
                                                       D_ji.cols - A.ranks(i, level));

                        matmul(D_j_block_splits[2], D_i_block_splits[0], D_ji_splits[2], false, true,
                               -1, 1);

                      }
                    }
                    else {              // update co block in the transposed upper triangle
                      if (exists_and_inadmissible(A, i, j, level)) {
                        Matrix& D_ij = A.D(i, j, level);
                        auto D_ij_splits = split_dense(D_ij,
                                                       D_ij.rows - A.ranks(i, level),
                                                       D_ij.cols - A.ranks(j, level));
                        matmul(D_i_block_splits[0], D_j_block_splits[2], D_ij_splits[1],
                               false, true, 1, 1);
                      }
                    }
                  });

  reduction_loop4(A, block, level,
                  [&](int64_t i, int64_t j, std::vector<Matrix>& D_i_block_splits,
                      std::vector<Matrix>& D_block_j_splits) {
                    if (exists_and_inadmissible(A, i, j, level)) {
                      Matrix& D_ij = A.D(i, j, level);
                      auto D_ij_splits = split_dense(D_ij,
                                                     D_ij.rows - A.ranks(i, level),
                                                     D_ij.cols - A.ranks(j, level));
                      matmul(D_i_block_splits[0], D_block_j_splits[1], D_ij_splits[1],
                             false, false, -1.0, 1.0);
                    }
                  });


  reduction_loop5(A, block, level, [&](int64_t i, int64_t j, std::vector<Matrix>& D_i_block_splits,
                                       std::vector<Matrix>& D_block_j_splits) {
    if (exists_and_inadmissible(A, i, j, level)) {
      Matrix& D_ij = A.D(i, j, level);
      auto D_ij_splits = split_dense(D_ij,
                                     D_ij.rows - A.ranks(i, level),
                                     D_ij.cols - A.ranks(j, level));

      matmul(D_i_block_splits[2], D_block_j_splits[1], D_ij_splits[3],
             false, false, -1.0, 1.0);
    }
  });

  // 6. Between co and oo blocks.
  for (int64_t i = 0; i < block; ++i) {
    if (exists_and_inadmissible(A, block, i, level) && exists_and_inadmissible(A, i, i, level)) {
      Matrix& A_i_block = A.D(block, i, level);
      auto A_i_block_splits = split_dense(A_i_block,
                                          A_i_block.rows - A.ranks(block, level),
                                          A_i_block.cols - A.ranks(i, level));

      Matrix& A_ii = A.D(i, i, level);
      auto A_ii_splits = split_dense(A_ii,
                                     A_ii.rows - A.ranks(i, level),
                                     A_ii.cols - A.ranks(i, level));

      syrk(A_i_block_splits[1], A_ii_splits[3], Hatrix::Lower, true, -1, 1);
    }
  }
}

void
compute_fill_ins(SymmetricSharedBasisMatrix& A, int64_t block,
                 int64_t level, RowColLevelMap<Matrix>& F) {
  reduction_loop1(A, block, level, [&](int64_t i, int64_t j, std::vector<Matrix>& D_i_block_splits,
                                       std::vector<Matrix>& D_j_block_splits) {
    if (exists_and_admissible(A, i, j, level)) {
      Matrix fill_in =
        F.exists(i, j, level) ? F(i, j, level) : Matrix(A.U(i, level).rows,
                                                        A.U(j, level).rows);

      auto fill_in_splits = split_dense(fill_in,
                                        A.U(i, level).rows - A.ranks(i, level),
                                        A.U(j, level).rows - A.ranks(j, level));
      matmul(D_i_block_splits[0], D_j_block_splits[0], fill_in_splits[0], false, true,
             -1.0, 1.0);

      F.insert(i, j, level, std::move(fill_in));
    }
  });

  reduction_loop2(A, block, level, [&](int64_t i, int64_t j, std::vector<Matrix>& D_i_block_splits,
                                       std::vector<Matrix>& D_j_block_splits) {
    if (exists_and_admissible(A, i, j, level)) {
      Matrix fill_in =
        F.exists(i, j, level) ? F(i, j, level) : Matrix(A.U(i, level).rows,
                                                        A.U(j, level).rows);

      auto fill_in_splits = split_dense(fill_in,
                                        A.U(i, level).rows - A.ranks(i, level),
                                        A.U(j, level).rows - A.ranks(j, level));
      matmul(D_i_block_splits[2], D_j_block_splits[2], fill_in_splits[3], false, true,
             -1.0, 1.0);

      F.insert(i, j, level, std::move(fill_in));
    }
  });

  reduction_loop3(A, block, level, [&](int64_t i, int64_t j, std::vector<Matrix>& D_i_block_splits,
                                       std::vector<Matrix>& D_j_block_splits) {
    if (exists_and_admissible(A, i, j, level)) {
      Matrix fill_in =
        F.exists(i, j, level) ? F(i, j, level) : Matrix(A.U(i, level).rows,
                                                        A.U(j, level).rows);
      auto fill_in_splits = split_dense(fill_in,
                                        A.U(i, level).rows - A.ranks(i, level),
                                        A.U(j, level).rows - A.ranks(j, level));
      matmul(D_i_block_splits[2], D_j_block_splits[0], fill_in_splits[2], false, true,
             -1.0, 1.0);

      F.insert(i, j, level, std::move(fill_in));
    }
  });

  reduction_loop4(A, block, level, [&](int64_t i, int64_t j, std::vector<Matrix>& D_i_block_splits,
                                       std::vector<Matrix>& D_block_j_splits) {
    if (exists_and_admissible(A, i, j, level)) {
      Matrix fill_in =
        F.exists(i, j, level) ? F(i, j, level) :
        Matrix(A.U(i, level).rows, A.U(j, level).rows);

      auto fill_in_splits = split_dense(fill_in,
                                        A.U(i, level).rows - A.ranks(i, level),
                                        A.U(j, level).cols - A.ranks(j, level));
      matmul(D_i_block_splits[0], D_block_j_splits[2], fill_in_splits[1], false, true,
             -1.0, 1.0);
    }
  });

  reduction_loop5(A, block, level, [&](int64_t i, int64_t j, std::vector<Matrix>& D_i_block_splits,
                                       std::vector<Matrix>& D_block_j_splits) {
    if (exists_and_admissible(A, i, j, level)) {
      Matrix fill_in =
        F.exists(i, j, level) ? F(i, j, level) : Matrix(A.U(i, level).rows,
                                                        A.U(j, level).rows);

      auto fill_in_splits = split_dense(fill_in,
                                        fill_in.rows - A.ranks(i, level),
                                        fill_in.cols - A.ranks(j, level));
      matmul(D_i_block_splits[2], D_block_j_splits[1], fill_in_splits[3],
             false, false, -1.0, 1.0);
      F.insert(i, j, level, std::move(fill_in));
    }
  });
}

void
merge_unfactorized_blocks(SymmetricSharedBasisMatrix& A, int64_t level) {
  int64_t parent_level = level - 1;
  // Merge the unfactorized parts and prepare for the next level.
  for (int i = 0; i < pow(2, parent_level); ++i) {
    for (int j = 0; j <= i; ++j) {
      if (exists_and_inadmissible(A, i, j, parent_level)) {
        std::vector<int64_t> i_children({i * 2, i * 2 + 1}), j_children({j * 2, j * 2 + 1});
        const int64_t D_unelim_rows = A.ranks(i_children[0], level) + A.ranks(i_children[1], level);
        const int64_t D_unelim_cols = A.ranks(j_children[0], level) + A.ranks(j_children[1], level);
        Matrix D_unelim(D_unelim_rows, D_unelim_cols);
        auto D_unelim_splits = split_dense(D_unelim,
                                           A.ranks(i_children[0], level),
                                           A.ranks(j_children[0], level));

        for (int ic1 = 0; ic1 < 2; ++ic1) {
          for (int jc2 = 0; jc2 < ((i == j) ? (ic1+1) : 2); ++jc2) {
            int64_t c1 = i_children[ic1], c2 = j_children[jc2];

            if (A.is_admissible.exists(c1, c2, level)) {
              if (!A.is_admissible(c1, c2, level)) {
                Matrix& D_c1c2 = A.D(c1, c2, level);
                auto D_c1c2_splits = split_dense(D_c1c2,
                                            D_c1c2.rows - A.ranks(c1, level),
                                            D_c1c2.cols - A.ranks(c2, level));
                D_unelim_splits[ic1 * 2 + jc2] = D_c1c2_splits[3];
              }
              else {
                D_unelim_splits[ic1 * 2 + jc2] = A.S(c1, c2, level);
              }
            }
          }
        }

        A.D.insert(i, j, parent_level, std::move(D_unelim));
      }
    }
  }
}

static bool
col_has_admissible_blocks(SymmetricSharedBasisMatrix& A, const int64_t block,
                          const int64_t level) {
  bool has_admis = false;
  const int64_t nblocks = pow(2, level);
  for (int64_t i = block+1; i < nblocks; i++) {
    if (!A.is_admissible.exists(i, block, level) || // part of upper level admissible block
        exists_and_admissible(A, i, block, level)) {
      has_admis = true;
      break;
    }
  }
 return has_admis;
}

static bool
row_has_admissible_blocks(SymmetricSharedBasisMatrix& A, const int64_t block,
                          const int64_t level) {
  bool has_admis = false;
  for (int64_t j = 0; j < block; j++) {
    if (!A.is_admissible.exists(block, j, level) || // part of upper level admissible block
        exists_and_admissible(A, block, j, level)) {
      has_admis = true;
      break;
    }
  }
  return has_admis;
}

void
update_col_cluster_basis(SymmetricSharedBasisMatrix& A,
                         const int64_t block,
                         const int64_t level,
                         const RowColLevelMap<Matrix>& F,
                         RowMap<Matrix>& t,
                         const Args& opts) {
  const int64_t nblocks = pow(2, level);
  const int64_t block_size = A.D(block, block, level).cols;

  Matrix col_concat(0, block_size);
  col_concat = concat(col_concat,
                      matmul(US(block, level), A.U(block, level), false, true), 0);

  for (int64_t i = block+1; i < nblocks; ++i) {
    if (exists_and_admissible(A, i, block, level)) {
      if (F.exists(i, block, level)) {
        col_concat = concat(col_concat, F(i, block, level), 0);
      }
    }
  }

  Matrix col_concat_T = transpose(col_concat);
  Matrix Q,R;
  std::tie(Q, R) = pivoted_qr_nopiv_return(col_concat_T, A.ranks(block, level));

  Matrix Si(R.rows, R.rows), Vi(R.rows, R.cols);
  rq(R, Si, Vi);

  US.erase(block, level);
  US.insert(block, level, std::move(Si));

  // update the projection of the new basis on the old.
  Matrix t_row = matmul(Q, A.U(block, level), true, false);
  if (t.exists(block)) { t.erase(block); }
  t.insert(block, std::move(t_row));

  // update the transfer matrix on this row.
  A.U.erase(block, level);
  A.U.insert(block, level, std::move(Q));

}

void
update_row_cluster_basis(SymmetricSharedBasisMatrix& A,
                         const int64_t block,
                         const int64_t level,
                         const RowColLevelMap<Matrix>& F,
                         RowMap<Matrix>& r,
                         const Args& opts) {
  const int64_t block_size = A.D(block, block, level).rows;

  // Algorithm 1 from Ma2019a.
  // Update the row basis and obtain the projection of the old basis on the new
  // in order to incorporate it into the S block.

  // This is a temporary way to verify the cluster basis update.
  Matrix row_concat(block_size, 0);
  row_concat = concat(row_concat, matmul(A.U(block, level), US(block, level)), 1);

  for (int64_t j = 0; j < block; ++j) {
    if (exists_and_admissible(A, block, j, level)) {
      // row_concat = concat(row_concat,
      //                     matmul(A.U(block, level), A.S(block, j, level)), 1);

      if (F.exists(block, j, level)) {
        row_concat = concat(row_concat, F(block, j, level), 1);
      }
    }
  }

  Matrix Q,R;
  std::tie(Q, R) = pivoted_qr_nopiv_return(row_concat, A.ranks(block, level));

  Matrix Si(R.rows, R.rows), Vi(R.rows, R.cols);
  rq(R, Si, Vi);

  US.erase(block, level);
  US.insert(block, level, std::move(Si));

  // update the projection of the new basis on the old.
  Matrix r_row = matmul(Q, A.U(block, level), true, false);
  if (r.exists(block)) { r.erase(block); }
  r.insert(block, std::move(r_row));

  // update the transfer matrix on this row.
  A.U.erase(block, level);
  A.U.insert(block, level, std::move(Q));
}

void
multiply_complements(SymmetricSharedBasisMatrix& A, const int64_t block,
                     const int64_t level) {
  int64_t nblocks = pow(2, level);
  // capture the pre-matrix for verification of factorization.
  auto U_F = make_complement(A.U(block, level));

  // left multiply with the complement along the (symmetric) row.
  A.D(block, block, level) = matmul(U_F, A.D(block, block, level), true);
  for (int64_t j = 0; j < block; ++j) {
    if (exists_and_inadmissible(A, block, j, level)) {
      auto D_splits =
        A.D(block, j, level).split({},
                                   std::vector<int64_t>(1,
                                                        A.D(block, j, level).cols -
                                                        A.ranks(j, level)));
      D_splits[1] = matmul(U_F, D_splits[1], true);
    }
  }

  // right multiply with the transpose of the complement
  A.D(block, block, level) = matmul(A.D(block, block, level), U_F);
  for (int64_t i = block+1; i < nblocks; ++i) {
    if (exists_and_inadmissible(A, i, block, level)) {
      auto D_splits =
        A.D(i, block, level).split(std::vector<int64_t>(1, A.D(i, block, level).rows -
                                                        A.ranks(i, level)),
                                   {});

      D_splits[1] = matmul(D_splits[1], U_F);
    }
  }
}

void
update_row_S_blocks(Hatrix::SymmetricSharedBasisMatrix& A,
                    int64_t block, int64_t level,
                    const Hatrix::RowMap<Hatrix::Matrix>& r) {
  // update the S blocks with the new projected basis.
  for (int64_t j = 0; j < block; ++j) {
    if (exists_and_admissible(A, block, j, level)) {
      A.S(block, j, level) = matmul(r(block), A.S(block, j, level));
    }
  }
}

void
update_col_S_blocks(Hatrix::SymmetricSharedBasisMatrix& A,
                    int64_t block, int64_t level,
                    const Hatrix::RowMap<Hatrix::Matrix>& t) {
  int64_t nblocks = pow(2, level);
  // update the S blocks in this column.
  for (int64_t i = block+1; i < nblocks; ++i) {
    if (exists_and_admissible(A, i, block, level)) {
      A.S(i, block, level) = matmul(A.S(i, block, level), t(block), false, true);
    }
  }
}

void
update_col_transfer_basis(Hatrix::SymmetricSharedBasisMatrix& A,
                          const int64_t block, const int64_t level,
                          Hatrix::RowMap<Hatrix::Matrix>& t) {
  // update the transfer matrices one level higher
  const int64_t parent_level = level - 1;
  const int64_t parent_block = block / 2;
  if (parent_level > 0 && col_has_admissible_blocks(A, parent_block, parent_level)) {
    const int64_t c1 = parent_block * 2;
    const int64_t c2 = parent_block * 2 + 1;

    Matrix& Utransfer = A.U(parent_block, parent_level);
    Matrix Utransfer_new(A.U(c1, level).cols + A.U(c2, level).cols,
                         Utransfer.cols);

    auto Utransfer_splits = Utransfer.split(std::vector<int64_t>(1, A.U(c1, level).cols),
                                            {});
    auto Utransfer_new_splits = Utransfer_new.split(std::vector<int64_t>(1,
                                                                         A.U(c1, level).cols),
                                                    {});
    if (block == c1) {
      matmul(Utransfer_splits[0], t(c1), Utransfer_new_splits[0], false, true, 1, 0);
      Utransfer_new_splits[1] = Utransfer_splits[1];
      t.erase(c1);
    }
    else {
      matmul(Utransfer_splits[1], t(c2), Utransfer_new_splits[1], false, true, 1, 0);
      Utransfer_new_splits[0] = Utransfer_splits[0];
      t.erase(c2);
    }

    A.U.erase(parent_block, parent_level);
    A.U.insert(parent_block, parent_level, std::move(Utransfer_new));
  }
}

void
update_row_transfer_basis(Hatrix::SymmetricSharedBasisMatrix& A,
                          const int64_t block, const int64_t level,
                          Hatrix::RowMap<Hatrix::Matrix>& r) {
  // update the transfer matrices one level higher.
  const int64_t parent_block = block / 2;
  const int64_t parent_level = level - 1;
  if (parent_level > 0 && row_has_admissible_blocks(A, parent_block, parent_level)) {
    const int64_t c1 = parent_block * 2;
    const int64_t c2 = parent_block * 2 + 1;

    Matrix& Utransfer = A.U(parent_block, parent_level);
    Matrix Utransfer_new(A.U(c1, level).cols + A.U(c2, level).cols,
                         Utransfer.cols);

    auto Utransfer_splits = Utransfer.split(std::vector<int64_t>(1, A.U(c1, level).cols),
                                            {});
    auto Utransfer_new_splits = Utransfer_new.split(std::vector<int64_t>(1,
                                                                         A.U(c1, level).cols),
                                                    {});
    if (block == c1) {
      matmul(r(c1), Utransfer_splits[0], Utransfer_new_splits[0], false, false, 1, 0);
      Utransfer_new_splits[1] = Utransfer_splits[1];
      r.erase(c1);
    }
    else {
      matmul(r(c2), Utransfer_splits[1], Utransfer_new_splits[1], false, false, 1, 0);
      Utransfer_new_splits[0] = Utransfer_splits[0];
      r.erase(c2);
    }

    A.U.erase(parent_block, parent_level);
    A.U.insert(parent_block, parent_level, std::move(Utransfer_new));
  }
}

void
update_row_cluster_basis_and_S_blocks(Hatrix::SymmetricSharedBasisMatrix& A,
                                      Hatrix::RowColLevelMap<Hatrix::Matrix>& F,
                                      Hatrix::RowMap<Hatrix::Matrix>& r,
                                      const Hatrix::Args& opts,
                                      const int64_t block,
                                      const int64_t level) {
  bool found_row_fill_in = false;
  for (int64_t j = 0; j < block; ++j) {
    if (F.exists(block, j, level)) {
      found_row_fill_in = true;
      break;
    }
  }

  if (found_row_fill_in) {    // update row cluster bases
    // recompress fill-ins on this row so that they dont generate further fill-ins.
    update_row_cluster_basis(A, block, level, F, r, opts);
    update_row_S_blocks(A, block, level, r);
    update_row_transfer_basis(A, block, level, r);
  }
}

void
update_col_cluster_basis_and_S_blocks(Hatrix::SymmetricSharedBasisMatrix& A,
                                      Hatrix::RowColLevelMap<Hatrix::Matrix>& F,
                                      Hatrix::RowMap<Hatrix::Matrix>& t,
                                      const Hatrix::Args& opts,
                                      const int64_t block,
                                      const int64_t level) {
  bool found_col_fill_in = false;
  int64_t nblocks = pow(2, level);

  for (int64_t i = block+1; i < nblocks; ++i) {
    if (F.exists(i, block, level)) {
      found_col_fill_in = true;
      break;
    }
  }

  if (found_col_fill_in) {
    update_col_cluster_basis(A, block, level, F, t, opts);
    update_col_S_blocks(A, block, level, t);
    update_col_transfer_basis(A, block, level, t);
  }
}

void
factorize_level(SymmetricSharedBasisMatrix& A,
                int64_t level, RowColLevelMap<Matrix>& F,
                RowMap<Matrix>& r, RowMap<Matrix>& t,
                const Hatrix::Args& opts) {
  const int64_t nblocks = pow(2, level);

  for (int64_t block = 0; block < nblocks; ++block) {
    update_row_cluster_basis_and_S_blocks(A, F, r, opts, block, level);
    update_col_cluster_basis_and_S_blocks(A, F, t, opts, block, level);

    multiply_complements(A, block, level);
    factorize_diagonal(A, block, level);
    triangle_reduction(A, block, level);
    compute_schurs_complement(A, block, level);
    compute_fill_ins(A, block, level, F);
  } // for (int block = 0; block < nblocks; ++block)
}

void
factorize(Hatrix::SymmetricSharedBasisMatrix& A, const Hatrix::Args& opts) {
  RowColLevelMap<Matrix> F;
  int64_t level;

  for (int level = A.max_level; level >= A.min_level; --level) {
    int nblocks = pow(2, level);

    for (int i = 0; i < nblocks; ++i) {
      for (int j = i + 1; j < nblocks; ++j) {
        if (A.is_admissible.exists(i, j, level)) {
          A.is_admissible.erase(i, j, level);
        }
      }
    }
  }

  for (level = A.max_level; level >= A.min_level; --level) {
    int64_t nblocks = pow(2, level);
    RowMap<Matrix> r, t;
    // PLU of one level of the H2 matrix.

    factorize_level(A, level, F, r, t, opts);

    const int64_t parent_level = level-1;
    const int64_t parent_nblocks = pow(2, parent_level);

    // Update coupling matrices of each admissible block to add fill in contributions.
    for (int64_t i = 0; i < nblocks; ++i) {
      for (int64_t j = 0; j <= i; ++j) {
        if (exists_and_admissible(A, i, j, level)) {
          if (F.exists(i, j, level)) {
            Matrix projected_fill_in = matmul(matmul(A.U(i, level), F(i, j, level),
                                                     true),
                                              A.U(j, level));
            A.S(i, j, level) += projected_fill_in;
          }
        }
      }
    }

    // Propagate fill-in to upper level admissible blocks
    if (parent_level > 0) {
      for (int64_t i = 0; i < parent_nblocks; ++i) {
        for (int64_t j = 0; j <= i; ++j) {
          if (exists_and_admissible(A, i, j, parent_level) ||
              !A.is_admissible.exists(i, j, parent_level)) {
            int64_t i1 = i * 2;
            int64_t i2 = i * 2 + 1;
            int64_t j1 = j * 2;
            int64_t j2 = j * 2 + 1;

            if (F.exists(i1, j1, level) || F.exists(i1, j2, level) ||
                F.exists(i2, j1, level) || F.exists(i2, j2, level)) {
              int64_t nrows = A.U(i1, level).cols + A.U(i2, level).cols;
              int64_t ncols = A.U(j1, level).cols + A.U(j2, level).cols;
              Matrix fill_in(nrows, ncols);

              auto fill_in_splits = split_dense(fill_in,
                                                A.U(i1, level).cols,
                                                A.U(j1, level).cols);
              if (F.exists(i1, j1, level)) {
                matmul(matmul(A.U(i1, level),
                              F(i1, j1, level), true, false),
                       A.U(j1, level),
                       fill_in_splits[0], false, false, 1, 0);
              }
              if (F.exists(i1, j2, level)) {
                matmul(matmul(A.U(i1, level),
                              F(i1, j2, level), true, false),
                       A.U(j2, level),
                       fill_in_splits[1], false, false, 1, 0);
              }
              if (F.exists(i2, j1, level)) {
                matmul(matmul(A.U(i2, level),
                              F(i2, j1, level), true, false),
                       A.U(j1, level),
                       fill_in_splits[2], false, false, 1, 0);
              }
              if (F.exists(i2, j2, level)) {
                matmul(matmul(A.U(i2, level),
                              F(i2, j2, level), true, false),
                       A.U(j2, level),
                       fill_in_splits[3], false, false, 1, 0);
              }

              F.insert(i, j, parent_level, std::move(fill_in));
            }
          }
        }
      }
    }

    r.erase_all();
    t.erase_all();

    merge_unfactorized_blocks(A, level);
  } // level loop

  F.erase_all();

  int64_t last_nodes = pow(2, level);
  for (int d = 0; d < last_nodes; ++d) {
    cholesky(A.D(d, d, level), Hatrix::Lower);
    for (int i = d+1; i < last_nodes; ++i) {
      solve_triangular(A.D(d, d, level), A.D(i, d, level), Hatrix::Right, Hatrix::Lower,
                       false, true, 1.0);
    }

    for (int i = d+1; i < last_nodes; ++i) {
      for (int j = d+1; j <= i; ++j) {
        if (i == j) {
          syrk(A.D(i, d, level), A.D(i, j, level), Hatrix::Lower, false, -1.0, 1.0);
        }
        else {
          matmul(A.D(i, d, level), A.D(j, d, level), A.D(i, j, level), false, true, -1.0, 1.0);
        }
      }
    }
  }
}

void
solve_forward_level(const SymmetricSharedBasisMatrix& A, Matrix& x_level,
                    const int64_t level) {
  int64_t nblocks = pow(2, level);
  std::vector<int64_t> row_offsets;
  int64_t nrows = 0;
  for (int64_t i = 0; i < nblocks; ++i) {
    row_offsets.push_back(nrows + A.D(i, i, level).rows);
    nrows += A.D(i, i, level).rows;
  }
  std::vector<Matrix> x_level_split = x_level.split(row_offsets, {});

  for (int64_t block = 0; block < nblocks; ++block) {
    Matrix U_F = make_complement(A.U(block, level));
    Matrix prod = matmul(U_F, x_level_split[block]);
    x_level_split[block] = prod;

    int64_t rank = A.ranks(block, level);
    const int64_t row_split = A.D(block, block, level).rows - rank;
    auto block_splits = split_dense(A.D(block, block, level),
                                    row_split,
                                    A.D(block, block, level).cols - rank);

    Matrix x_block(x_level_split[block], true);
    auto x_block_splits = x_block.split(std::vector<int64_t>(1, row_split), {});

    // forward substitution with cc and oc blocks on the diagonal dense block.
    solve_triangular(block_splits[0], x_block_splits[0], Hatrix::Left, Hatrix::Lower, false,
                     false, 1.0);
    matmul(block_splits[2], x_block_splits[0], x_block_splits[1], false, false, -1.0, 1.0);
    x_level_split[block] = x_block;

    // apply the oc blocks that are actually in the upper triangular matrix.
    for (int64_t irow = 0; irow < block; ++irow) {
      if (exists_and_inadmissible(A, block, irow, level)) { // need to take the symmetric block
        const Matrix& D_block_irow = A.D(block, irow, level);
        const int64_t row_split = D_block_irow.rows - A.ranks(block, level);
        const int64_t col_split = D_block_irow.cols - A.ranks(irow, level);
        auto D_block_irow_splits = split_dense(D_block_irow, row_split, col_split);

        Matrix x_block(x_level_split[block], true), x_irow(x_level_split[irow], true);
        auto x_block_splits = x_block.split(std::vector<int64_t>(1, row_split), {});
        auto x_irow_splits = x_irow.split(std::vector<int64_t>(1, col_split), {});

        matmul(D_block_irow_splits[1], x_block_splits[0], x_irow_splits[1],
               true, false, -1.0, 1.0);
        x_level_split[irow] = x_irow;
      }
    }

    // forward subsitute with (cc;oc) blocks below the diagonal.
    for (int64_t irow = block+1; irow < nblocks; ++irow) {
      if (exists_and_inadmissible(A, irow, block, level)) {
        const Matrix& D_irow_block = A.D(irow, block, level);
        const int64_t col_split = D_irow_block.cols - A.ranks(block, level);
        auto lower_splits = D_irow_block.split({},
                                               std::vector<int64_t>(1, col_split));

        Matrix x_block(x_level_split[block], true), x_irow(x_level_split[irow], true);
        auto x_block_splits = x_block.split(std::vector<int64_t>(1, col_split), {});

        matmul(lower_splits[0], x_block_splits[0], x_irow, false, false, -1.0, 1.0);
        x_level_split[irow] = x_irow;
      }
    }
  }
}

void
solve_backward_level(const SymmetricSharedBasisMatrix& A, Matrix& x_level,
                     const int64_t level) {
  int64_t nblocks = pow(2, level);
  std::vector<int64_t> col_offsets;
  int64_t nrows = 0;
  for (int64_t i = 0; i < nblocks; ++i) {
    col_offsets.push_back(nrows + A.D(i, i, level).cols);
    nrows += A.D(i, i, level).cols;
  }
  std::vector<Matrix> x_level_split = x_level.split(col_offsets, {});

  for (int64_t block = nblocks-1; block >= 0; --block) {
    // backward substitution

    // apply the tranpose of the oc block that is actually in the lower triangle.
    for (int64_t icol = 0; icol < block; ++icol) {
      if (exists_and_inadmissible(A, block, icol, level)) {
        const Matrix& D_icol_block = A.D(block, icol, level);
        const int64_t row_split = D_icol_block.rows - A.ranks(block, level);
        const int64_t col_split = D_icol_block.cols - A.ranks(icol, level);

        auto D_icol_block_splits = split_dense(D_icol_block, row_split, col_split);

        Matrix x_block(x_level_split[block], true), x_icol(x_level_split[icol], true);
        auto x_block_splits = x_block.split(std::vector<int64_t>(1, row_split),
                                            {});
        auto x_icol_splits = x_icol.split(std::vector<int64_t>(1, col_split),
                                          {});
        matmul(D_icol_block_splits[2], x_block_splits[1], x_icol_splits[0],
               true, false, -1.0, 1.0);
        x_level_split[icol] = x_icol;
      }
    }

    // apply the cc and oc blocks (transposed) to the respective slice
    // of the vector.
    for (int64_t icol = nblocks-1; icol > block; --icol) {
      if (exists_and_inadmissible(A, icol, block, level)) {
        const Matrix& D_icol_block = A.D(icol, block, level);
        const int64_t col_split = D_icol_block.cols - A.ranks(block, level);

        auto D_icol_block_splits = D_icol_block.split({},
                                                      std::vector<int64_t>(1, col_split));

        Matrix x_block(x_level_split[block], true);
        auto x_block_splits = x_block.split(std::vector<int64_t>(1, col_split), {});

        matmul(D_icol_block_splits[0], x_level_split[icol], x_block_splits[0],
               true, false, -1.0, 1.0);
        x_level_split[block] = x_block;
      }
    }

    // backward substition using the diagonal block.
    int64_t rank = A.ranks(block, level);
    int64_t row_split = A.D(block, block, level).rows - rank;
    int64_t col_split = A.D(block, block, level).cols - rank;
    auto block_splits = split_dense(A.D(block, block, level), row_split, col_split);

    Matrix x_block(x_level_split[block], true);
    auto x_block_splits = x_block.split(std::vector<int64_t>(1, row_split), {});
    matmul(block_splits[2], x_block_splits[1], x_block_splits[0], true, false, -1.0, 1.0);
    solve_triangular(block_splits[0], x_block_splits[0],
                     Hatrix::Left, Hatrix::Lower, false, true, 1.0);
    x_level_split[block] = x_block;

    auto V_F = make_complement(A.U(block, level));
    Matrix prod = matmul(V_F, x_level_split[block], true);
    x_level_split[block] = prod;
  }
}

int64_t
permute_forward(const SymmetricSharedBasisMatrix& A,
                Matrix& x, int64_t level, int64_t permute_offset) {
  Matrix copy(x);
  int64_t num_nodes = pow(2, level);
  int64_t c_offset = permute_offset;
  for (int64_t block = 0; block < num_nodes; ++block) {
    permute_offset += A.D(block, block, level).rows - A.ranks(block, level);
  }

  int64_t csize_offset = 0, bsize_offset = 0, rsize_offset = 0;
  for (int64_t block = 0; block < num_nodes; ++block) {
    int64_t rows = A.D(block, block, level).rows;
    int64_t c_size = rows - A.ranks(block, level);

    // copy the complement part of the vector into the temporary vector
    for (int64_t i = 0; i < c_size; ++i) {
      copy(c_offset + csize_offset + i, 0) = x(c_offset + bsize_offset + i, 0);
    }
    // copy the rank part of the vector into the temporary vector
    for (int64_t i = 0; i < A.ranks(block, level); ++i) {
      copy(permute_offset + rsize_offset + i, 0) =
        x(c_offset + bsize_offset + c_size + i, 0);
    }

    csize_offset += c_size;
    bsize_offset += rows;
    rsize_offset += A.ranks(block, level);
  }

  x = copy;
  return permute_offset;
}

int64_t
permute_backward(const SymmetricSharedBasisMatrix& A,
                 Matrix& x, const int64_t level, int64_t rank_offset) {
  Matrix copy(x);
  int64_t num_nodes = pow(2, level);
  int64_t c_offset = rank_offset;
  for (int64_t block = 0; block < num_nodes; ++block) {
    c_offset -= A.D(block, block, level).cols - A.ranks(block, level);
  }

  int64_t csize_offset = 0, bsize_offset = 0, rsize_offset = 0;
  for (int64_t block = 0; block < num_nodes; ++block) {
    int64_t cols = A.D(block, block, level).cols;
    int64_t c_size = cols - A.ranks(block, level);

    for (int64_t i = 0; i < c_size; ++i) {
      copy(c_offset + bsize_offset + i, 0) = x(c_offset + csize_offset + i, 0);
    }

    for (int64_t i = 0; i < A.ranks(block, level); ++i) {
      copy(c_offset + bsize_offset + c_size + i, 0) = x(rank_offset + rsize_offset + i, 0);
    }

    csize_offset += c_size;
    bsize_offset += cols;
    rsize_offset += A.ranks(block, level);
  }

  x = copy;

  return c_offset;
}

Hatrix::Matrix
solve(const Hatrix::SymmetricSharedBasisMatrix& A,
      const Hatrix::Matrix& b) {
  Matrix x(b);
  int64_t level_offset = 0;
  std::vector<Matrix> x_splits;
  int64_t level;

  // forward substitution.
  for (level = A.max_level; level >= A.min_level; --level) {
    int nblocks = pow(2, level);
    int64_t n = 0;              // total vector length due to variable ranks.
    for (int64_t i = 0; i < nblocks; ++i) { n += A.D(i, i, level).rows; }

    Matrix x_level(n, 1);
    for (int64_t i = 0; i < x_level.rows; ++i) {
      x_level(i, 0) = x(level_offset + i, 0);
    }

    solve_forward_level(A, x_level, level);
    for (int64_t i = 0; i < x_level.rows; ++i) {
      x(level_offset + i, 0) = x_level(i, 0);
    }

    level_offset = permute_forward(A, x, level, level_offset);
  }

  x_splits = x.split(std::vector<int64_t>(1, level_offset), {});
  Matrix x_last(x_splits[1]);

  int64_t last_nodes = pow(2, level);
  std::vector<int64_t> vector_splits;
  int64_t nrows = 0;
  for (int64_t i = 0; i < last_nodes; ++i) {
    vector_splits.push_back(nrows + A.D(i, i, level).rows);
    nrows += A.D(i, i, level).rows;
  }
  auto x_last_splits = x_last.split(vector_splits, {});

  // forward for the last blocks
  for (int i = 0; i < last_nodes; ++i) {
    for (int j = 0; j < i; ++j) {
      matmul(A.D(i, j, level), x_last_splits[j], x_last_splits[i], false, false, -1.0, 1.0);
    }
    solve_triangular(A.D(i, i, level), x_last_splits[i], Hatrix::Left, Hatrix::Lower,
                     false, false, 1.0);
  }

  // backward for the last blocks.
  for (int j = last_nodes-1; j >= 0; --j) {
    for (int i = last_nodes-1; i > j; --i) {
      matmul(A.D(i, j, level), x_last_splits[i], x_last_splits[j], true, false, -1.0, 1.0);
    }
    solve_triangular(A.D(j, j, level), x_last_splits[j], Hatrix::Left, Hatrix::Lower,
                     false, true, 1.0);
  }

  x_splits[1] = x_last;
  ++level;

  // backward substitution.
  for (; level <= A.max_level; ++level) {
    int64_t nblocks = pow(2, level);

    int64_t n = 0;
    for (int64_t i = 0; i < nblocks; ++i) { n += A.D(i, i, level).cols; }
    Matrix x_level(n, 1);

    level_offset = permute_backward(A, x, level, level_offset);
    for (int64_t i = 0; i < x_level.rows; ++i) {
      x_level(i, 0) = x(level_offset + i, 0);
    }

    solve_backward_level(A, x_level, level);
    for (int64_t i = 0; i < x_level.rows; ++i) {
      x(level_offset + i, 0) = x_level(i, 0);
    }
  }

  return x;
}

static void
multiply_S(const Hatrix::SymmetricSharedBasisMatrix& A,
           std::vector<Matrix>& x_hat, std::vector<Matrix>& b_hat,
           int x_hat_offset, int b_hat_offset, int level) {
  int64_t nblocks = pow(2, level);

  std::mt19937 gen(0);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < i; ++j) {
      if (A.is_admissible.exists(i, j, level) && A.is_admissible(i, j, level)) {
        matmul(A.S(i, j, level), x_hat[x_hat_offset + j], b_hat[b_hat_offset + i]);
        matmul(A.S(i, j, level), x_hat[x_hat_offset + i],
               b_hat[b_hat_offset + j], true, false);
      }
    }
  }
}

Matrix
matmul(const Hatrix::SymmetricSharedBasisMatrix& A, const Matrix& x) {
  int leaf_nblocks = pow(2, A.max_level);
  std::vector<Matrix> x_hat;
  auto x_splits = x.split(leaf_nblocks, 1);

  std::mt19937 gen(0);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  // V leaf nodes
  for (int i = 0; i < leaf_nblocks; ++i) {
    x_hat.push_back(matmul(A.U(i, A.max_level), x_splits[i], true, false, 1.0));
  }

  int64_t x_hat_offset = 0;     // index offset for the x_hat array.
  for (int64_t level = A.max_level - 1; level >= A.min_level; --level) {
    int64_t nblocks = pow(2, level);
    int64_t child_level = level + 1;
    for (int64_t i = 0; i < nblocks; ++i) {
      int64_t c1 = i * 2;
      int64_t c2 = i * 2 + 1;

      Matrix xtemp = Matrix(A.U(i, level).rows, 1);
      auto xtemp_splits = xtemp.split(std::vector<int64_t>(1, A.ranks(c1, child_level)),
                                      {});
      xtemp_splits[0] = x_hat[x_hat_offset + c1];
      xtemp_splits[1] = x_hat[x_hat_offset + c2];

      x_hat.push_back(matmul(A.U(i, level), xtemp, true, false, 1.0));
    }

    x_hat_offset += pow(2, child_level);
  }

  // b_hat does the product in reverse so matrices are pushed from the back.
  std::vector<Matrix> b_hat;
  int64_t nblocks = pow(2, A.min_level);
  for (int64_t i = 0; i < nblocks; ++i) {
    b_hat.push_back(Matrix(A.ranks(i, A.min_level), 1));
  }

  // Multiply the S blocks at the top-most level with the corresponding xhat.
  int64_t b_hat_offset = 0;
  multiply_S(A, x_hat, b_hat, x_hat_offset, b_hat_offset, A.min_level);

  // Multiply the S block with the col bases transfer matrices.
  for (int64_t level = A.min_level; level < A.max_level; ++level) {
    int64_t nblocks = pow(2, level);
    int64_t child_level = level + 1;
    x_hat_offset -= pow(2, child_level);

    for (int64_t row = 0; row < nblocks; ++row) {
      int c_r1 = row * 2, c_r2 = row * 2 + 1;
      Matrix Ub = matmul(A.U(row, level),
                         b_hat[b_hat_offset + row]);
      auto Ub_splits = Ub.split(std::vector<int64_t>(1, A.U(c_r1, child_level).cols),
                                {});

      b_hat.push_back(Matrix(Ub_splits[0], true));
      b_hat.push_back(Matrix(Ub_splits[1], true));
    }

    multiply_S(A, x_hat, b_hat, x_hat_offset, b_hat_offset + nblocks, child_level);

    b_hat_offset += nblocks;
  }

  // Multiply with the leaf level transfer matrices.
  Matrix b(x.rows, 1);
  auto b_splits = b.split(leaf_nblocks, 1);
  for (int64_t i = 0; i < leaf_nblocks; ++i) {
    matmul(A.U(i, A.max_level), b_hat[b_hat_offset + i], b_splits[i]);
  }


  // Multiply with the dense blocks to obtain the final product in b_splits.
  for (int i = 0; i < leaf_nblocks; ++i) {
    matmul(A.D(i, i, A.max_level), x_splits[i], b_splits[i]);
  }

  for (int64_t i = 0; i < leaf_nblocks; ++i) {
    for (int64_t j = 0; j < i; ++j) {
      if (A.is_admissible.exists(i, j, A.max_level) &&
          !A.is_admissible(i, j, A.max_level)) {
        // TODO: make the diagonal tringular and remove this.
        matmul(A.D(i, j, A.max_level), x_splits[j], b_splits[i]);
        if (i != j) {
          matmul(A.D(i, j, A.max_level), x_splits[i], b_splits[j], true, false);
        }
      }
    }
  }

  return b;
}
