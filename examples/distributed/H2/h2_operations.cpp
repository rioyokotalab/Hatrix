#include <algorithm>
#include <exception>
#include <cmath>
#include <random>
#include <vector>
#include <chrono>

#include "Hatrix/Hatrix.h"
#include "distributed/distributed.hpp"

#include "h2_construction.hpp"
#include "h2_operations.hpp"

#ifdef USE_MKL
#include "mkl_cblas.h"
#else
#include "cblas.h"
#endif

#ifdef USE_MKL
#include "mkl_cblas.h"
#include "mkl_lapacke.h"
#else
#include "cblas.h"
#include "lapacke.h"
#endif

using namespace Hatrix;

Hatrix::RowLevelMap US;

std::vector<double> timer;
std::vector<int64_t> counts;

Hatrix::RowColMap<std::vector<int> > pivot_map;

Matrix orig(32, 32);

static double cond_2norm(const Matrix& A) {
  Matrix copy(A, true);
  inverse(copy);

  double nrm = Hatrix::norm(A);
  double inv_nrm = Hatrix::norm(copy);

  return nrm / inv_nrm;
}

static double cond_svd(const Matrix& A) {
  Matrix copy(A, true);
  Matrix _U(A, true), _S(A, true), _V(A, true);
  double error;

  svd(copy, _U, _S, _V);

  return _S(0,0) / _S(_S.rows-1, _S.cols-1);
}

void
factorize_diagonal(SymmetricSharedBasisMatrix& A, int64_t block, int64_t level) {
  int nleaf = A.D(block, block, level).rows, rank = A.ranks(block,level);
  Matrix& diagonal = A.D(block, block, level);
  auto diagonal_splits = split_dense(diagonal,
                                     nleaf-rank,
                                     nleaf-rank);
  cholesky(diagonal_splits[0], Hatrix::Lower);
  solve_triangular(diagonal_splits[0], diagonal_splits[2], Hatrix::Right, Hatrix::Lower,
                   false, true, 1.0);
  syrk(diagonal_splits[2], diagonal_splits[3], Hatrix::Lower, false, -1, 1);

  // double *_D = &A.D(block, block, level);
  // LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', nleaf-rank, _D, nleaf);
  // cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans,
  //             CblasNonUnit,
  //             rank, nleaf-rank, 1.0,
  //             _D, nleaf, _D+(nleaf-rank), nleaf);
  // cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, rank, nleaf-rank, -1,
  //             _D+(nleaf-rank), nleaf, 1.0, _D + nleaf * (nleaf-rank) + (nleaf-rank),
  //             nleaf);

  std::cout << "POST SYRK: " << cond_svd(A.D(block, block, level)) << std::endl;
}

void partial_triangle_reduce(SymmetricSharedBasisMatrix& A,
                             const Matrix& Dcc,
                             const int64_t row,
                             const int64_t col,
                             const int64_t level,
                             const int64_t split_index,
                             Hatrix::Side side,
                             Hatrix::Mode uplo,
                             bool unit_diag,
                             bool transA) {
  if (exists_and_inadmissible(A, row, col, level)) {
    auto D_i_block_splits = split_dense(A.D(row, col, level),
                                        A.D(row, col, level).rows - A.ranks(row, level),
                                        A.D(row, col, level).cols - A.ranks(col, level));

    solve_triangular(Dcc, D_i_block_splits[split_index], side, uplo, unit_diag, transA, 1.0);
  }
}

void triangle_reduction(SymmetricSharedBasisMatrix& A, int64_t block, int64_t level) {
  Matrix& diagonal = A.D(block, block, level);
  auto diagonal_splits = split_dense(diagonal,
                                     diagonal.rows - A.ranks(block, level),
                                     diagonal.cols - A.ranks(block, level));
  const Matrix& Dcc = diagonal_splits[0];
  const int64_t nblocks = pow(2, level);

  // TRSM with oc blocks along the 'block' column.
  for (int64_t i = block; i < nblocks; ++i) {
    partial_triangle_reduce(A, Dcc, i, block, level, 2, Hatrix::Right, Hatrix::Lower, false, true);
  }

  // TRSM with cc blocks along the 'block' column after the diagonal block.
  for (int64_t i = block+1; i < nblocks; ++i) {
    partial_triangle_reduce(A, Dcc, i, block, level, 0, Hatrix::Right, Hatrix::Lower, false, true);
  }

  // TRSM with co blocks behind the diagonal on the 'block' row.
  for (int64_t j = 0; j < block; ++j) {
    partial_triangle_reduce(A, Dcc, block, j, level, 1, Hatrix::Left, Hatrix::Lower, false, false);
  }
}

void
compute_schurs_complement(SymmetricSharedBasisMatrix& A, int64_t block, int64_t level) {
  int64_t nblocks = pow(2, level);
  // schur's complement in front of diagonal block.

  // r * c x (b * c).T -> b * r. This one is only between the co of the diagonal and the
  // blocks below it. The schur's complement is applied only on the (i, block) dense block.
  for (int64_t i = block+1; i < nblocks; ++i) {
    if (exists_and_inadmissible(A, i, block, level)) {
      auto D_block_block_split = split_dense(A.D(block, block, level),
                                             A.D(block, block, level).rows - A.ranks(block, level),
                                             A.D(block, block, level).cols - A.ranks(block, level));
      auto D_i_block_split =
        A.D(i, block, level).split({},
                                   std::vector<int64_t>(1,
                                                        A.D(i, block, level).cols -
                                                        A.ranks(block, level)));
      matmul(D_i_block_split[0], D_block_block_split[2], D_i_block_split[1], false, true, -1, 1);
    }
  }

  // b*c x (b*c).T -> b*b.
  for (int64_t i : near_neighbours(block, level)) {
    for (int64_t j : near_neighbours(block, level)) {
      if (i >= block+1 && j >= block+1) {
        if (exists_and_inadmissible(A, i, j, level)) {
          auto D_i_block_split =
            A.D(i, block, level).split({},
                                       std::vector<int64_t>(1,
                                                        A.D(i, block, level).cols -
                                                            A.ranks(block, level)));
          auto D_j_block_split =
            A.D(j, block, level).split({},
                                       std::vector<int64_t>(1,
                                                            A.D(j, block, level).cols -
                                                            A.ranks(block, level)));

          if (i == j) {
            syrk(D_i_block_split[0], A.D(i, j, level), Hatrix::Lower, false, -1, 1);
          }
          else {
            matmul(D_i_block_split[0], D_j_block_split[0], A.D(i, j, level), false, true, -1, 1);
          }
        }
      }
    }
  }

  // schur's complement behind the diagonal block.

  // (r*c) x (c*r) = (r*r)
  for (int64_t j : near_neighbours(block, level)) {
    if (j < block) {
      auto D_block_block_split = split_dense(A.D(block, block, level),
                                             A.D(block, block, level).rows - A.ranks(block, level),
                                              A.D(block, block, level).cols - A.ranks(block, level));
      auto D_block_j_split = split_dense(A.D(block, j, level),
                                         A.D(block, j, level).rows - A.ranks(block, level),
                                         A.D(block, j, level).cols - A.ranks(j, level));

      matmul(D_block_block_split[2], D_block_j_split[1], D_block_j_split[3], false, false, -1, 1);
    }
  }

  // (nb*c) x (c*r) -> (nb*r)
  for (int64_t i : near_neighbours(block, level)) {
    for (int64_t j : near_neighbours(block, level)) {
      if (i >= block+1 && j < block) {
        if (exists_and_inadmissible(A, i, j, level)) {
          auto D_i_block_splits =
            A.D(i, block, level).split({},
                                       std::vector<int64_t>(1,
                                                            A.D(i, block, level).cols -
                                                            A.ranks(block, level)));
          auto D_block_j_splits =
            split_dense(A.D(block, j, level),
                        A.D(block, j, level).rows - A.ranks(block, level),
                        A.D(block, j, level).cols - A.ranks(j, level));

          auto D_ij_splits =
            A.D(i, j, level).split({},
                                   std::vector<int64_t>(1, A.D(i, j, level).cols -
                                                        A.ranks(j, level)));
          matmul(D_i_block_splits[0], D_block_j_splits[1], D_ij_splits[1], false, false, -1, 1);
        }
      }
    }
  }
}

void
compute_fill_ins(SymmetricSharedBasisMatrix& A, int64_t block,
                 int64_t level, RowColLevelMap<Matrix>& F) {
  int nblocks = pow(2, level);

  // b*b sized fill in
  for (int i = block+1; i < nblocks; ++i) {
    for (int j = block+1; j < i; ++j) {
      if (exists_and_inadmissible(A, i, block, level) &&
          exists_and_inadmissible(A, j, block, level)) {
        if (exists_and_admissible(A, i, j, level)) {
          Matrix fill_in = Matrix(A.U(i, level).rows, A.U(j, level).rows);

          auto fill_in_splits = split_dense(fill_in,
                                            fill_in.rows - A.ranks(i, level),
                                            fill_in.cols - A.ranks(j, level));

          auto D_i_block_splits = split_dense(A.D(i, block, level),
                                              A.D(i, block, level).rows - A.ranks(i, level),
                                              A.D(i, block, level).cols - A.ranks(block, level));

          auto D_j_block_splits = split_dense(A.D(j, block, level),
                                              A.D(j, block, level).rows - A.ranks(j, level),
                                              A.D(j, block, level).cols - A.ranks(block, level));

          matmul(D_i_block_splits[0], D_j_block_splits[0], fill_in_splits[0],
                 false, true, -1, 1); // cc
          matmul(D_i_block_splits[2], D_j_block_splits[0], fill_in_splits[2],
                 false, true, -1, 1); // oc
          matmul(D_i_block_splits[2], D_j_block_splits[2], fill_in_splits[3],
                 false, true, -1, 1); // oo

          if (F.exists(i, j, level)) {
            F(i, j, level) += fill_in;
          }
          else {
            F.insert(i, j, level, std::move(fill_in));
          }
        }
      }
    }
  }

  // b * rank sized fill-in
  for (int i = block+1; i < nblocks; ++i) {
    for (int j = 0; j < block; ++j) {
      if (exists_and_inadmissible(A, i, block, level) &&
          exists_and_inadmissible(A, block, j, level))  {
        if (exists_and_admissible(A, i, j, level)) {
          Matrix fill_in(A.U(i, level).rows, A.ranks(j, level));

          auto D_i_block_splits = A.D(i, block, level).split(
                                              {},
                                              std::vector<int64_t>(1,
                                                                   A.D(i, block, level).cols -
                                                                   A.ranks(block, level)));

          auto D_block_j_splits = split_dense(A.D(block, j, level),
                                              A.D(block, j, level).rows - A.ranks(block, level),
                                              A.D(block, j, level).cols - A.ranks(j, level));

          matmul(D_i_block_splits[0], D_block_j_splits[1], fill_in, false, false, -1, 0);

          Matrix projected_fill_in = matmul(fill_in, A.U(j, level), false, true);

          if (F.exists(i, j, level)) {
            F(i, j, level) += projected_fill_in;
          }
          else {
            F.insert(i, j, level, std::move(projected_fill_in));
          }
        }
      }
    }
  }
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
        Matrix D_unelim = generate_identity_matrix(D_unelim_rows, D_unelim_cols);
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
                std::cout << "c1: " << c1 << " c2: " << c2 << " lvl: " << level << std::endl;
                D_unelim_splits[ic1 * 2 + jc2] = A.S(c1, c2, level);
              }
            }
          }
        }

        std::cout << "i : " << i << " j: " << j << " parent level: " << parent_level << std::endl;

        if (i == j) {
          for (int ii = 0; ii < D_unelim.rows; ++ii) {
            for (int jj = ii+1; jj < D_unelim.cols; ++jj) {
              D_unelim(ii, jj) = 0;
            }
          }
        }

        A.D.insert(i, j, parent_level, std::move(D_unelim));
      }
    }
  }
}

void
update_col_cluster_basis(SymmetricSharedBasisMatrix& A,
                         const int64_t block,
                         const int64_t level,
                         RowColLevelMap<Matrix>& F,
                         RowMap<Matrix>& t,
                         const Args& opts) {
  const int64_t nblocks = pow(2, level);
  const int64_t block_size = A.D(block, block, level).cols;

  Matrix fill_in(block_size, block_size);
  for (int64_t i : far_neighbours(block, level)) {
    if (i >= block+1) {
      if (F.exists(i, block, level)) {
        fill_in += matmul(F(i, block, level), F(i, block, level), true, false);
        // F.erase(i, block, level);
      }
    }
  }

  fill_in += matmul(A.U(block, level), matmul(US(block, level), A.U(block, level), false, true));
  // fill_in = concat(fill_in, matmul(A.U(block, level), US(block, level)), 1);

  Matrix col_concat_T = transpose(fill_in);
  Matrix Q,R;
  Matrix Si, Vi;

  switch(opts.kind_of_recompression) {
  case 0:                       // accuracy truncated
    int64_t rank;
    std::tie(Q, R, rank) = error_pivoted_qr(col_concat_T,
                                            opts.qr_accuracy * 1e-1,
                                            false, false);

    Q.shrink(A.U(block,level).rows, A.ranks(block, level));
    R.shrink(A.ranks(block, level), A.ranks(block, level));

    Vi.destructive_resize(R.rows, R.cols);
    Si.destructive_resize(R.rows, R.rows);

    rq(R, Si, Vi);
    break;
  case 1:                       // lapack constant rank QR
    std::tie(Q, R) = pivoted_qr_nopiv_return(col_concat_T, A.ranks(block, level));
    Vi.destructive_resize(R.rows, R.cols);
    Si.destructive_resize(R.rows, R.rows);

    rq(R, Si, Vi);
    break;
  case 2:                       // constant rank
    std::tie(Q, R) = truncated_pivoted_qr(col_concat_T, A.ranks(block, level));
    Vi.destructive_resize(R.rows, R.cols);
    Si.destructive_resize(R.rows, R.rows);

    rq(R, Si, Vi);
    break;
  case 3:                       // fixed rank svd
    double err;
    std::tie(Q, Si, Vi, err) = truncated_svd(col_concat_T, A.ranks(block, level));
    break;
  default:
    throw std::runtime_error("wrong option for kind_of_recompression");
  }

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
                         RowColLevelMap<Matrix>& F,
                         RowMap<Matrix>& r,
                         const Args& opts) {
  const int64_t block_size = A.D(block, block, level).rows;

  // Algorithm 1 from Ma2019a.
  // Update the row basis and obtain the projection of the old basis on the new
  // in order to incorporate it into the S block.

  // This is a temporary way to verify the cluster basis update.
  Matrix fill_in(block_size, block_size);

  for (int64_t j : far_neighbours(block, level)) {
    if (j < block) {
      if (F.exists(block, j, level)) {
        fill_in += matmul(F(block, j, level), F(block, j, level), false, true);
        // F.erase(block, j, level);
      }
    }
  }

  fill_in += matmul(matmul(A.U(block, level), US(block, level)), A.U(block, level), false, true);

  // fill_in = concat(fill_in, matmul(A.U(block, level), US(block, level)), 0);

  Matrix Q,R;
  Matrix Si, Vi;

  switch(opts.kind_of_recompression) {
  case 0:                       // accuracy truncated
    int64_t rank;
    std::tie(Q, R, rank) = error_pivoted_qr(fill_in,
                                            opts.qr_accuracy * 1e-1,
                                            false, false);

    Q.shrink(A.U(block,level).rows, A.ranks(block, level));
    R.shrink(A.ranks(block, level), A.ranks(block, level));

    Vi.destructive_resize(R.rows, R.cols);
    Si.destructive_resize(R.rows, R.rows);

    rq(R, Si, Vi);
    break;
  case 1:                       // lapack constant rank QR
    std::tie(Q, R) = pivoted_qr_nopiv_return(fill_in, A.ranks(block, level));
    Vi.destructive_resize(R.rows, R.cols);
    Si.destructive_resize(R.rows, R.rows);

    rq(R, Si, Vi);
    break;
  case 2:                       // constant rank
    std::tie(Q, R) = truncated_pivoted_qr(fill_in, A.ranks(block, level));
    Vi.destructive_resize(R.rows, R.cols);
    Si.destructive_resize(R.rows, R.rows);

    rq(R, Si, Vi);
    break;
  case 3:                       // fixed rank svd
    double err;
    std::tie(Q, Si, Vi, err) = truncated_svd(fill_in, A.ranks(block, level));
    break;
  default:
    throw std::runtime_error("wrong option for kind_of_recompression");
  }

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

  // left multiply with the complement along the (symmetric) row.
  auto diagonal_splits = split_dense(A.D(block, block, level),
                                     A.D(block, block, level).rows - A.ranks(block, level),
                                     A.D(block, block, level).cols - A.ranks(block, level));

  std::cout << "@@@ PRE-PRODUCT @@@ " << cond_svd(A.D(block, block, level))
            << " " << cond_svd(diagonal_splits[3]) << std::endl;
    // A.D(block, block, level).print();



  auto U_F = make_complement(A.U(block, level));

  A.D(block, block, level) = matmul(matmul(U_F, A.D(block, block, level), true), U_F);


  // std::cout << "@@@ PRODUCT @@@ "  << cond_svd(post_diagonal_splits[0]) << std::endl;
  std::cout << "@@@ PRODUCT @@@ "  << cond_svd(A.D(block, block, level)) << std::endl;
  // A.D(block, block, level).print();

  // for (int64_t j : near_neighbours(block, level)) {
  //   if (j < block) {
  //     auto D_splits =
  //       A.D(block, j, level).split({},
  //                                  std::vector<int64_t>(1,
  //                                                       A.D(block, j, level).cols -
  //                                                       A.ranks(j, level)));
  //     D_splits[1] = matmul(U_F, D_splits[1], true);
  //   }
  // }

  // for (int64_t i : near_neighbours(block, level)) {
  //   if (i > block) {
  //     A.D(i, block, level) = matmul(A.D(i, block, level), U_F);
  //   }
  // }
}

void
update_row_S_blocks(Hatrix::SymmetricSharedBasisMatrix& A,
                    int64_t block, int64_t level,
                    const Hatrix::RowMap<Hatrix::Matrix>& r) {
  // update the S blocks with the new projected basis.
  for (int64_t j : far_neighbours(block, level)) {
    if (j < block) {
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
  for (int64_t i : far_neighbours(block, level)) {
    if (i > block) {
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
  if (parent_level > 0) {
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
      matmul(t(c1), Utransfer_splits[0], Utransfer_new_splits[0], false, false, 1, 0);
      Utransfer_new_splits[1] = Utransfer_splits[1];
      t.erase(c1);
    }
    else {
      matmul(t(c2), Utransfer_splits[1], Utransfer_new_splits[1], false, false, 1, 0);
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
  if (parent_level > 0) {
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
    counts[1] += 1;
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
    counts[0] += 1;
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
    // auto start_cluster_update = std::chrono::system_clock::now();
    // update_row_cluster_basis_and_S_blocks(A, F, r, opts, block, level);
    // update_col_cluster_basis_and_S_blocks(A, F, t, opts, block, level);
    // auto stop_cluster_update = std::chrono::system_clock::now();
    // timer[0] += std::chrono::duration_cast<
    //   std::chrono::milliseconds>(stop_cluster_update - start_cluster_update).count();

    // auto start_multiply_complements = std::chrono::system_clock::now();
    multiply_complements(A, block, level);
    // auto stop_multiply_complements = std::chrono::system_clock::now();
    // timer[1] += std::chrono::duration_cast<
    //   std::chrono::milliseconds>(stop_multiply_complements - start_multiply_complements).count();

    // auto start_factorize_diag = std::chrono::system_clock::now();
    factorize_diagonal(A, block, level);
    // auto stop_factorize_diag = std::chrono::system_clock::now();
    // timer[2] += std::chrono::duration_cast<
    //   std::chrono::milliseconds>(stop_factorize_diag - start_factorize_diag).count();

    // auto start_triangle = std::chrono::system_clock::now();
    // triangle_reduction(A, block, level);
    // auto stop_triangle = std::chrono::system_clock::now();
    // timer[3] += std::chrono::duration_cast<
    //   std::chrono::milliseconds>(stop_triangle - start_triangle).count();

    // auto start_schurs = std::chrono::system_clock::now();
    // compute_schurs_complement(A, block, level);
    // auto stop_schurs = std::chrono::system_clock::now();
    // timer[4] += std::chrono::duration_cast<
    //   std::chrono::milliseconds>(stop_schurs - start_schurs).count();

    // auto start_fill_ins = std::chrono::system_clock::now();
    // compute_fill_ins(A, block, level, F);
    // auto stop_fill_ins = std::chrono::system_clock::now();
    // timer[5] += std::chrono::duration_cast<
    //   std::chrono::milliseconds>(stop_fill_ins - start_fill_ins).count();
  } // for (int block = 0; block < nblocks; ++block)
}

long long int
factorize(Hatrix::SymmetricSharedBasisMatrix& A, const Hatrix::Args& opts) {
  Hatrix::profiling::PAPI papi;
  papi.add_fp_ops(0);
  papi.start();
  RowColLevelMap<Matrix> F;
  int64_t level;
  timer.resize(8, 0);
  counts.resize(10, 0);
  // Matrix pre_fac(30, 30);

  // orig = A.D(0, 0, 2);

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

    auto start_merge = std::chrono::system_clock::now();
    const int64_t parent_level = level-1;
    const int64_t parent_nblocks = pow(2, parent_level);

    // Update coupling matrices of each admissible block to add fill in contributions.
    for (int64_t i = 0; i < nblocks; ++i) {
      for (int64_t j : far_neighbours(i, level)) {
        if (F.exists(i, j, level)) {
          Matrix projected_fill_in = matmul(matmul(A.U(i, level), F(i, j, level),
                                                   true),
                                            A.U(j, level));
          A.S(i, j, level) += projected_fill_in;
          F.erase(i, j, level);
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

    auto stop_merge = std::chrono::system_clock::now();
    timer[6] += std::chrono::duration_cast<
      std::chrono::milliseconds>(stop_merge - start_merge).count();
  } // level loop

  F.erase_all();

  // Matrix dense(opts.max_rank * 4, opts.max_rank * 4);
  // auto d_splits = split_dense(dense,
  //                             opts.max_rank * 2,
  //                             opts.max_rank * 2);

  // d_splits[0] = A.D(0, 0, 1);
  // d_splits[2] = A.D(1, 0, 1);
  // d_splits[1] = transpose(A.D(1, 0, 1));
  // d_splits[3] = A.D(1, 1, 1);

  // for (int i = 0; i < dense.rows; ++i) {
  //   for (int j = i+1; j < dense.cols; ++j) {
  //     dense(i, j) = 0;
  //   }
  // }


  // std::cout << "full dense cond: " << cond_svd(dense) << std::endl;
  // cholesky(dense, Hatrix::Lower);
  // std::cout << "full dense post factor xcond: " << cond_svd(dense) << std::endl;

  // A.D(0, 0, 1) = d_splits[0];
  // A.D(1, 0, 1) = d_splits[2];
  // A.D(1, 1, 1) = d_splits[3];

  auto start_last = std::chrono::system_clock::now();
  int64_t last_nodes = pow(2, level);
  for (int d = 0; d < last_nodes; ++d) {
    std::cout << "pre d: " << d << " lvl: " << level << " "
              << cond_svd(A.D(d, d, level)) << std::endl;

    cholesky(A.D(d, d, level), Hatrix::Lower);
    for (int i = d+1; i < last_nodes; ++i) {
      solve_triangular(A.D(d, d, level), A.D(i, d, level), Hatrix::Right, Hatrix::Lower,
                       false, true, 1.0);

      std::cout << "post d: " << d << " lvl: " << level << " "
                << cond_svd(A.D(d, d, level)) << " "
                << cond_svd(A.D(i, d, level))
                << std::endl;
    }

    for (int i = d+1; i < last_nodes; ++i) {
      for (int j = d+1; j <= i; ++j) {
        if (i == j) {
          syrk(A.D(i, d, level), A.D(i, j, level), Hatrix::Lower, false, -1.0, 1.0);
          std::cout << "post i: " << i << " j: " << j << " lvl: " << level << " "
                    << cond_svd(A.D(i, j, level))
                    << std::endl;
        }
        else {
          matmul(A.D(i, d, level), A.D(j, d, level),
                 A.D(i, j, level), false, true, -1.0, 1.0);
        }
      }
    }
  }

  auto stop_last = std::chrono::system_clock::now();
  timer[7] += std::chrono::duration_cast<
    std::chrono::milliseconds>(stop_last - start_last).count();

  auto fp_ops = papi.fp_ops();
  return fp_ops;
}

void
factorize_raw(SymmetricSharedBasisMatrix& A, Hatrix::Args& opts) {
  int nleaf = 16; int rank = 10;
  double *D002 = new double[nleaf * nleaf];
  auto UF0 = make_complement(A.U(0, 2));
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
              nleaf, nleaf, nleaf, 1.0,
              &UF0, UF0.stride,
              &A.D(0,0,2), A.D(0,0,2).stride,
              0.0,
              D002, nleaf);

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
              nleaf, nleaf, nleaf, 1.0,
              D002, nleaf,
              &UF0, UF0.stride,
              0.0,
              &A.D(0,0,2), A.D(0,0,2).stride);

  delete[] D002;

  D002 = &A.D(0,0,2);

  LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', (nleaf-rank), D002, nleaf);
  cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans,
              CblasNonUnit,
              rank, (nleaf-rank), 1.0,
              D002, nleaf, D002+(nleaf-rank), nleaf);

  cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, rank, (nleaf-rank), -1,
              D002+(nleaf-rank), nleaf, 1.0, D002 + nleaf * (nleaf-rank) + (nleaf-rank), nleaf);

  double *D112 = new double[nleaf * nleaf];
  auto UF1 = make_complement(A.U(1, 2));
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
              nleaf, nleaf, nleaf, 1.0,
              &UF1, UF1.stride,
              &A.D(1,1,2), A.D(1,1,2).stride,
              0.0,
              D112, nleaf);

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
              nleaf, nleaf, nleaf, 1.0,
              D112, nleaf,
              &UF1, UF1.stride,
              0.0,
              &A.D(1,1,2), A.D(1,1,2).stride);
  delete[] D112;

  D112 = &A.D(1,1,2);

  LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', (nleaf-rank), D112, nleaf);
  cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans,
              CblasNonUnit,
              rank, (nleaf-rank), 1.0,
              D112, nleaf, D112+(nleaf-rank), nleaf);
  cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, rank, (nleaf-rank), -1,
              D112+(nleaf-rank), nleaf, 1.0, D112 + nleaf * (nleaf-rank) + (nleaf-rank), nleaf);


  double *D222= new double[nleaf * nleaf];
  auto UF2 = make_complement(A.U(2, 2));
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
              nleaf, nleaf, nleaf, 1.0,
              &UF2, UF2.stride,
              &A.D(2,2,2), A.D(2,2,2).stride,
              0.0,
              D222, nleaf);

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
              nleaf, nleaf, nleaf, 1.0,
              D222, nleaf,
              &UF2, UF2.stride,
              0.0,
              &A.D(2,2,2), A.D(2,2,2).stride);
  delete[] D222;

  D222 = &A.D(2,2,2);

  LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', (nleaf-rank), D222, nleaf);
  cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans,
              CblasNonUnit,
              rank, (nleaf-rank), 1.0,
              D222, nleaf, D222+(nleaf-rank), nleaf);

  cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, rank, (nleaf-rank), -1,
              D222+(nleaf-rank), nleaf, 1.0, D222 + nleaf * (nleaf-rank) + (nleaf-rank), nleaf);

  double *D332= new double[nleaf * nleaf];
  auto UF3 = make_complement(A.U(3, 2));
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
              nleaf, nleaf, nleaf, 1.0,
              &UF3, UF3.stride,
              &A.D(3,3,2), A.D(3,3,2).stride,
              0.0,
              D332, nleaf);

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
              nleaf, nleaf, nleaf, 1.0,
              D332, nleaf,
              &UF3, UF3.stride,
              0.0,
              &A.D(3,3,2), A.D(3,3,2).stride);
  delete[] D332;

  D332 = &A.D(3, 3, 2);

  LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', (nleaf-rank), D332, nleaf);
  cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans,
              CblasNonUnit,
              rank, (nleaf-rank), 1.0,
              D332, nleaf, D332+(nleaf-rank), nleaf);
  cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, rank, (nleaf-rank), -1,
              D332+(nleaf-rank), nleaf, 1.0, D332 + nleaf * (nleaf-rank) + (nleaf-rank), nleaf);


  double *merge = new double[(rank*4) * (rank*4)]();
  int LDM = rank * 4;
  for (int i = 0; i < rank; ++i) { // 0,0
    for (int j = 0; j < rank; ++j) {
      // merge[i + j * (rank*2)] = A.D(0, 0, 1)(i+(nleaf-rank), j+(nleaf-rank));
      merge[i + j * LDM] = D002[(i+(nleaf-rank)) + (j+(nleaf-rank)) * nleaf];
    }
  }
  for (int i = 0; i < rank; ++i) { // 1,0
    for (int j = 0; j < rank; ++j) {
      merge[(i+rank) + j * LDM] = A.S(1, 0, 2)(i, j);
    }
  }
  for (int i = 0; i < rank; ++i) { // 1,1
    for (int j = 0; j < rank; ++j) {
      merge[(i+rank) + (j+rank) * LDM] = D112[(i+(nleaf-rank)) + (j+(nleaf-rank)) * nleaf];
    }
  }


  for (int i = 0; i < rank; ++i) { // 2,0
    for (int j = 0; j < rank; ++j) {
      merge[(i+rank*2) + j * LDM] = A.S(2,0,2)(i, j);
    }
  }

  for (int i = 0; i < rank; ++i) { // 2,1
    for (int j = 0; j < rank; ++j) {
      merge[(i+rank*2) + (j+rank) * LDM] = A.S(2,1,2)(i, j);
    }
  }


  for (int i = 0; i < rank; ++i) { // 2,2
    for (int j = 0; j < rank; ++j) {
      merge[(i+rank*2) + (j+rank*2) * LDM] = D222[(i+(nleaf-rank)) + (j+(nleaf-rank)) * nleaf];
    }
  }

 for (int i = 0; i < rank; ++i) { // 3,0
    for (int j = 0; j < rank; ++j) {
      merge[(i+rank*3) + j * LDM] = A.S(3,0,2)(i, j);
    }
  }

  for (int i = 0; i < rank; ++i) { // 3,1
    for (int j = 0; j < rank; ++j) {
      merge[(i+rank*3) + (j+rank) * LDM] = A.S(3,1,2)(i, j);
    }
  }

  for (int i = 0; i < rank; ++i) { // 3,2
    for (int j = 0; j < rank; ++j) {
      merge[(i+rank*3) + (j+rank*2) * LDM] = A.S(3,2,2)(i, j);
    }
  }

  for (int i = 0; i < rank; ++i) { // 3,3
    for (int j = 0; j < rank; ++j) {
      merge[(i+rank*3) + (j+rank*3) * LDM] = D332[(i+(nleaf-rank)) + (j+(nleaf-rank)) * nleaf];
    }
  }

  LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', (rank*4), merge, (rank*4));
  Matrix d_merge((rank*4), (rank*4));
  for (int i = 0; i < (rank*4); ++i) {
    for (int j = 0; j < (rank*4) ; ++j) {
      d_merge(i, j) = merge[i + j * LDM];
    }
  }
  auto d_splits = split_dense(d_merge, rank*2, rank*2);
  std::cout << "cond last: " << cond_svd(d_merge) << std::endl;

  Matrix d0(d_splits[0], true);
  Matrix d1(d_splits[2], true);
  Matrix d3(d_splits[3], true);

  A.D.insert(0,0,1, std::move(d0));
  A.D.insert(1,0,1, std::move(d1));
  A.D.insert(1,1,1, std::move(d3));

  delete[] merge;


}

void
solve_forward_level(const SymmetricSharedBasisMatrix& A,
                    Matrix& x_level,
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
    Matrix prod = matmul(U_F, x_level_split[block], true);
    x_level_split[block] = prod;

    int64_t rank = A.ranks(block, level);
    const int64_t row_split = A.D(block, block, level).rows - rank;
    auto block_splits = split_dense(A.D(block, block, level),
                                    row_split,
                                    A.D(block, block, level).cols - rank);

    Matrix x_block(x_level_split[block], true);
    auto x_block_splits =
      x_block.split(std::vector<int64_t>(1, row_split), {});

    // Forward substitution with cc and oc blocks on the
    // diagonal dense block.
    solve_triangular(block_splits[0], x_block_splits[0],
                     Hatrix::Left, Hatrix::Lower, false,
                     false, 1.0);
    matmul(block_splits[2], x_block_splits[0], x_block_splits[1],
           false, false, -1.0, 1.0);
    x_level_split[block] = x_block;

    // // apply the oc blocks that are actually in the upper triangular matrix.
    // for (int64_t irow = 0; irow < block; ++irow) {
    //   // need to take the symmetric block
    //   if (exists_and_inadmissible(A, block, irow, level)) {
    //     const Matrix& D_block_irow = A.D(block, irow, level);
    //     const int64_t row_split =
    //       D_block_irow.rows - A.ranks(block, level);
    //     const int64_t col_split =
    //       D_block_irow.cols - A.ranks(irow, level);
    //     auto D_block_irow_splits = split_dense(D_block_irow, row_split, col_split);

    //     Matrix x_block(x_level_split[block], true),
    //       x_irow(x_level_split[irow], true);
    //     auto x_block_splits = x_block.split(std::vector<int64_t>(1, row_split), {});
    //     auto x_irow_splits = x_irow.split(std::vector<int64_t>(1, col_split), {});

    //     matmul(D_block_irow_splits[1], x_block_splits[0], x_irow_splits[1],
    //            true, false, -1.0, 1.0);
    //     x_level_split[irow] = x_irow;
    //   }
    // }

    // // forward subsitute with (cc;oc) blocks below the diagonal.
    // for (int64_t irow = block+1; irow < nblocks; ++irow) {
    //   if (exists_and_inadmissible(A, irow, block, level)) {
    //     const Matrix& D_irow_block = A.D(irow, block, level);
    //     const int64_t col_split = D_irow_block.cols - A.ranks(block, level);
    //     auto lower_splits = D_irow_block.split({},
    //                                            std::vector<int64_t>(1, col_split));

    //     Matrix x_block(x_level_split[block], true), x_irow(x_level_split[irow], true);
    //     auto x_block_splits = x_block.split(std::vector<int64_t>(1, col_split), {});

    //     matmul(lower_splits[0], x_block_splits[0], x_irow, false, false, -1.0, 1.0);
    //     x_level_split[irow] = x_irow;
    //   }
    // }
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
    // for (int64_t icol = 0; icol < block; ++icol) {
    //   if (exists_and_inadmissible(A, block, icol, level)) {
    //     const Matrix& D_block_icol = A.D(block, icol, level);
    //     const int64_t row_split = D_block_icol.rows - A.ranks(block, level);
    //     const int64_t col_split = D_block_icol.cols - A.ranks(icol, level);

    //     auto D_block_icol_splits = split_dense(D_block_icol, row_split, col_split);

    //     Matrix x_block(x_level_split[block], true), x_icol(x_level_split[icol], true);
    //     auto x_block_splits = x_block.split(std::vector<int64_t>(1, row_split),
    //                                         {});
    //     auto x_icol_splits = x_icol.split(std::vector<int64_t>(1, col_split),
    //                                       {});
    //     matmul(D_block_icol_splits[2], x_block_splits[1], x_icol_splits[0],
    //            true, false, -1.0, 1.0);
    //     x_level_split[icol] = x_icol;
    //   }
    // }

    // apply the cc and oc blocks (transposed) to the respective slice
    // of the vector.
    // for (int64_t icol = nblocks-1; icol > block; --icol) {
    //   if (exists_and_inadmissible(A, icol, block, level)) {
    //     const Matrix& D_icol_block = A.D(icol, block, level);
    //     const int64_t col_split =
    //       D_icol_block.cols - A.ranks(block, level);

    //     auto D_icol_block_splits = D_icol_block.split({},
    //                                                   std::vector<int64_t>(1, col_split));

    //     Matrix x_block(x_level_split[block], true);
    //     auto x_block_splits = x_block.split(std::vector<int64_t>(1,
    //                                                              col_split),
    //                                         {});

    //     matmul(D_icol_block_splits[0], x_level_split[icol], x_block_splits[0],
    //            true, false, -1.0, 1.0);
    //     x_level_split[block] = x_block;
    //   }
    // }

    // backward substition using the diagonal block.
    int64_t rank = A.ranks(block, level);
    int64_t row_split = A.D(block, block, level).rows - rank;
    int64_t col_split = A.D(block, block, level).cols - rank;
    auto block_splits = split_dense(A.D(block, block, level),
                                    row_split, col_split);

    Matrix x_block(x_level_split[block], true);
    auto x_block_splits = x_block.split(std::vector<int64_t>(1,
                                                             row_split),
                                        {});
    matmul(block_splits[2], x_block_splits[1], x_block_splits[0],
           true, false, -1.0, 1.0);
    solve_triangular(block_splits[0], x_block_splits[0],
                     Hatrix::Left, Hatrix::Lower, false, true, 1.0);
    x_level_split[block] = x_block;

    auto V_F = make_complement(A.U(block, level));
    Matrix prod = matmul(V_F, x_level_split[block]);
    x_level_split[block] = prod;
  }
}

int64_t
permute_forward(const SymmetricSharedBasisMatrix& A,
                Matrix& x, int64_t level,
                int64_t permute_offset) {
  Matrix copy(x);
  int64_t num_nodes = pow(2, level);
  int64_t c_offset = permute_offset;
  for (int64_t block = 0; block < num_nodes; ++block) {
    permute_offset +=
      A.D(block, block, level).rows - A.ranks(block, level);
  }

  int64_t csize_offset = 0, bsize_offset = 0, rsize_offset = 0;
  for (int64_t block = 0; block < num_nodes; ++block) {
    int64_t rows = A.D(block, block, level).rows;
    int64_t c_size = rows - A.ranks(block, level);

    // copy the complement part of the vector into the temporary vector
    for (int64_t i = 0; i < c_size; ++i) {
      copy(c_offset + csize_offset + i, 0) =
        x(c_offset + bsize_offset + i, 0);
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
                 Matrix& x, const int64_t level,
                 int64_t rank_offset) {
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
solve_raw(const Hatrix::SymmetricSharedBasisMatrix& A,
          const Hatrix::Matrix& b) {
  Matrix x(b, true);
  int64_t level_offset = 0;
  int N = 64;
  int nleaf = 16;
  int rank = 10;

  double res[N];
  double *x_ptr = &x;

  // Forward.
  Matrix UF0 = make_complement(A.U(0, 2));
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, nleaf, 1, nleaf,
              1, &UF0, UF0.stride, x_ptr, N, 0, res, N);

  cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
              CblasNonUnit, (nleaf-rank), 1, 1.0,
              &A.D(0,0,2), A.D(0,0,2).stride,
              res, N);

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, rank, 1, (nleaf-rank),
              -1, &A.D(0,0,2) + (nleaf-rank), A.D(0,0,2).stride,
              res, N, 1, res+(nleaf-rank), N);

  Matrix UF1 = make_complement(A.U(1, 2));
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, nleaf, 1, nleaf,
              1, &UF1, UF1.stride, x_ptr+nleaf, x.stride, 0, res+nleaf, N);

  cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
              CblasNonUnit, (nleaf-rank), 1, 1.0,
              &A.D(1,1,2), A.D(1,1,2).stride,
              res+nleaf, N);

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, rank, 1, (nleaf-rank),
              -1, &A.D(1,1,2) + (nleaf-rank), A.D(1,1,2).stride,
              res+nleaf, N, 1, res+nleaf+(nleaf-rank), N);

  double copy[rank*2];
  for (int i = 0; i < rank; ++i) {
    copy[i] = res[(nleaf-rank) + i];
    copy[i+rank] = res[nleaf+(nleaf-rank)+i];
  }

  cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
              CblasNonUnit, 20, 1, 1.0,
              &A.D(0,0,1), A.D(0,0,1).stride,
              copy, rank*2);
  cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans,
              CblasNonUnit, 20, 1, 1.0,
              &A.D(0,0,1), A.D(0,0,1).stride,
              copy, rank*2);


  for (int i = 0; i < rank; ++i) {
    res[(nleaf-rank) + i] = copy[i];
    res[nleaf+(nleaf-rank)+i] = copy[i + rank];
  }

  // Backward. (1,1,2)
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, (nleaf-rank), 1, rank,
              -1, &A.D(1,1,2) + (nleaf-rank), A.D(1,1,2).stride,
              res+nleaf+(nleaf-rank), N, 1, res+nleaf, N);

  cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans,
              CblasNonUnit, (nleaf-rank), 1, 1.0,
              &A.D(1,1,2), A.D(1,1,2).stride,
              res+nleaf, N);

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nleaf, 1, nleaf,
              1, &UF1, UF1.stride, res+nleaf, N, 0, x_ptr+nleaf, N);

  // Backward. (0,0,2)
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, (nleaf-rank), 1, rank,
              -1, &A.D(0,0,2) + (nleaf-rank), A.D(0,0,2).stride,
              res+(nleaf-rank), N, 1, res, N);

  cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans,
              CblasNonUnit, (nleaf-rank), 1, 1.0,
              &A.D(0,0,2), A.D(0,0,2).stride,
              res, N);

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nleaf, 1, nleaf,
              1, &UF0, UF0.stride, res, x.stride, 0, x_ptr, N);

  return x;
}


Hatrix::Matrix
solve(const Hatrix::SymmetricSharedBasisMatrix& A,
      const Hatrix::Matrix& b) {
  Matrix x(b, true);
  int64_t level_offset = 0;
  std::vector<Matrix> x_splits;
  int64_t level;




  // forward substitution.
  for (level = A.max_level; level >= A.min_level; --level) {
    int nblocks = pow(2, level);
    int64_t n = 0;
    // total vector length due to variable ranks.
    for (int64_t i = 0; i < nblocks; ++i) {
      n += A.D(i, i, level).rows;
    }

    // copy from x into x_level
    Matrix x_level(n, 1);
    for (int64_t i = 0; i < x_level.rows; ++i) {
      x_level(i, 0) = x(level_offset + i, 0);
    }

    std::cout << "X(1) BEFORE\n";
    for (int i = 0; i < 16; ++i) {
      std::cout << x_level(i+16, 0) << std::endl;
    }

    solve_forward_level(A, x_level, level);
    // copy back into x from x_level
    for (int64_t i = 0; i < x_level.rows; ++i) {
      x(level_offset + i, 0) = x_level(i, 0);
    }

    std::cout << "X(1) AFTER\n";
    for (int i = 0; i < 16; ++i) {
      std::cout << x_level(i+16, 0) << std::endl;
    }

    level_offset = permute_forward(A, x, level, level_offset);
  }

  x_splits = x.split(std::vector<int64_t>(1, level_offset),
                     {});
  Matrix x_last(x_splits[1]);

  // std::cout << "x last:\n";
  // x_last.print();

  int64_t last_nodes = pow(2, level);
  std::vector<int64_t> vector_splits;
  int64_t nrows = 0;
  for (int64_t i = 0; i < last_nodes; ++i) {
    vector_splits.push_back(nrows + A.D(i, i, level).rows);
    nrows += A.D(i, i, level).rows;
  }
  auto x_last_splits = x_last.split(vector_splits, {});

  // LAPACKE_dpotrs(LAPACK_COL_MAJOR, 'L', 20, 1, &A.D(0,0,0), A.D(0,0,0).stride, &x + 6 * 2, 64);

  // forward for the last blocks
  for (int i = 0; i < last_nodes; ++i) {
    solve_triangular(A.D(i, i, level), x_last_splits[i],
                     Hatrix::Left, Hatrix::Lower,
                     false, false, 1.0);
    for (int j = 0; j < i; ++j) {
      matmul(A.D(i, j, level), x_last_splits[j],
             x_last_splits[i],
             false, false, -1.0, 1.0);
    }
  }

  // backward for the last blocks.
  for (int j = last_nodes-1; j >= 0; --j) {
    for (int i = last_nodes-1; i > j; --i) {
      matmul(A.D(i, j, level), x_last_splits[i],
             x_last_splits[j],
             true, false, -1.0, 1.0);
    }
    solve_triangular(A.D(j, j, level), x_last_splits[j],
                     Hatrix::Left, Hatrix::Lower,
                     false, true, 1.0);
  }

  x_splits[1] = x_last;
  ++level;

  // backward substitution.
  for (; level <= A.max_level; ++level) {
    int64_t nblocks = pow(2, level);

    int64_t n = 0;
    for (int64_t i = 0; i < nblocks; ++i) {
      n += A.D(i, i, level).cols;
    }
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

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < i; ++j) {
      if (A.is_admissible.exists(i, j, level) && A.is_admissible(i, j, level)) {
        matmul(A.S(i, j, level), x_hat[x_hat_offset + j], b_hat[b_hat_offset + i]);
        matmul(A.S(i, j, level), x_hat[x_hat_offset + i], b_hat[b_hat_offset + j],
               true, false);
      }
    }
  }
}

Matrix
matmul(const Hatrix::SymmetricSharedBasisMatrix& A, const Matrix& x) {
  int leaf_nblocks = pow(2, A.max_level);
  std::vector<Matrix> x_hat;
  auto x_splits = x.split(leaf_nblocks, 1);

  // // V leaf nodes
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

  for (int64_t i = 0; i < leaf_nblocks; ++i) {
    for (int64_t j = 0; j <= i; ++j) {
      if (A.is_admissible.exists(i, j, A.max_level) &&
          !A.is_admissible(i, j, A.max_level)) {
        // TODO: make the diagonal tringular and remove this.
        matmul(A.D(i, j, A.max_level), x_splits[j], b_splits[i]);
      }
    }
  }

  return b;
}
