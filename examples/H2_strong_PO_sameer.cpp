#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <tuple>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cassert>
#include <random>
#include <string>
#include <iomanip>
#include <functional>
#include <fstream>
#include <chrono>
#include <stdexcept>

#include "Hatrix/Hatrix.hpp"

using namespace Hatrix;
static Hatrix::greens_functions::kernel_function_t kernel;

Hatrix::RowLevelMap US;

// Level-wise lists generated from the is_admissible map of the matrix for
// traversal of the inadmissible and admissible blocks denoted by
// near_neighbours and far_neighbours, respectively.
RowColMap<std::vector<int64_t>> near_neighbours, far_neighbours;

static std::vector<Hatrix::Matrix>
split_dense(const Hatrix::Matrix& dense, int64_t row_split, int64_t col_split) {
  return dense.split(std::vector<int64_t>(1, row_split),
                     std::vector<int64_t>(1, col_split));
}

static bool
exists_and_inadmissible(const Hatrix::SymmetricSharedBasisMatrix& A,
                        const int64_t i, const int64_t j, const int64_t level) {
  return A.is_admissible.exists(i, j, level) && !A.is_admissible(i, j, level);
}

static bool
exists_and_admissible(const Hatrix::SymmetricSharedBasisMatrix& A,
                      const int64_t i, const int64_t j, const int64_t level) {
  return A.is_admissible.exists(i, j, level) && A.is_admissible(i, j, level);
}

static std::tuple<Matrix, Matrix, Matrix, int64_t>
svd_like_compression(Matrix& matrix,
                     const int64_t max_rank,
                     const double accuracy) {
  Matrix Ui, Si, Vi;
  int64_t rank;
  std::tie(Ui, Si, Vi, rank) = error_svd(matrix, accuracy, true, false);

  // Assume fixed rank if accuracy==0.
  rank = accuracy == 0. ? max_rank : std::min(max_rank, rank);

  return std::make_tuple(std::move(Ui), std::move(Si), std::move(Vi), std::move(rank));
}

static void
populate_near_far_lists(SymmetricSharedBasisMatrix& A) {
  // populate near and far lists. comment out when doing H2.
  for (int64_t level = A.max_level; level >= A.min_level; --level) {
    int64_t nblocks = pow(2, level);

    for (int64_t i = 0; i < nblocks; ++i) {
      far_neighbours.insert(i, level, std::vector<int64_t>());
      near_neighbours.insert(i, level, std::vector<int64_t>());
      for (int64_t j = 0; j <= i; ++j) {
        if (A.is_admissible.exists(i, j, level)) {
          if (A.is_admissible(i, j, level)) {
            far_neighbours(i, level).push_back(j);
          }
          else {
            near_neighbours(i, level).push_back(j);
          }
        }
      }
    }
  }
}

static bool
row_has_admissible_blocks(const Hatrix::SymmetricSharedBasisMatrix& A,
                          const int64_t row, const int64_t level) {
  bool has_admis = false;
  for (int64_t col = 0; col < pow(2, level); col++) {
    if ((!A.is_admissible.exists(row, col, level)) || // part of upper level admissible block
        (A.is_admissible.exists(row, col, level) && A.is_admissible(row, col, level))) {
      has_admis = true;
      break;
    }
  }

  return has_admis;
}

static RowLevelMap
generate_H2_strong_transfer_matrices(Hatrix::SymmetricSharedBasisMatrix& A,
                                     RowLevelMap Uchild,
                                     const Hatrix::Domain& domain,
                                     const int64_t N, const int64_t nleaf,
                                     const int64_t max_rank,
                                     const int64_t level, const double accuracy) {
  Matrix Ui, Si, _Vi; double error; int64_t rank;
  const int64_t nblocks = pow(2, level);
  const int64_t block_size = N / nblocks;
  Matrix AY(block_size, block_size);
  RowLevelMap Ubig_parent;
  const int64_t child_level = level + 1;

  for (int64_t row = 0; row < nblocks; ++row) {
    const int64_t child1 = row * 2;
    const int64_t child2 = row * 2 + 1;

    // Generate U transfer matrix.
    const Matrix& Ubig_child1 = Uchild(child1, child_level);
    const Matrix& Ubig_child2 = Uchild(child2, child_level);

    if (row_has_admissible_blocks(A, row, level)) {
      for (int64_t col = 0; col < nblocks; ++col) {
        if (!A.is_admissible.exists(row, col, level) ||
            (A.is_admissible.exists(row, col, level) && A.is_admissible(row, col, level))) {
          Hatrix::Matrix dense = generate_p2p_interactions(domain,
                                                           row * block_size, block_size,
                                                           col * block_size, block_size,
                                                           kernel);
          AY += dense;
        }
      }

      Matrix temp(Ubig_child1.cols + Ubig_child2.cols, AY.cols);
      std::vector<Matrix> temp_splits = temp.split(std::vector<int64_t>{Ubig_child1.cols},
                                                   std::vector<int64_t>{});
      std::vector<Matrix> AY_splits = AY.split(2, 1);

      matmul(Ubig_child1, AY_splits[0], temp_splits[0], true, false, 1, 0);
      matmul(Ubig_child2, AY_splits[1], temp_splits[1], true, false, 1, 0);

      std::tie(Ui, Si, _Vi, rank) = svd_like_compression(temp, max_rank, accuracy);
      Ui.shrink(Ui.rows, rank);
      A.U.insert(row, level, std::move(Ui));

      // Generate the full basis to pass to the next level.
      auto Utransfer_splits = A.U(row, level).split(std::vector<int64_t>{Ubig_child1.cols}, {});

      Matrix Ubig(block_size, rank);
      auto Ubig_splits = Ubig.split(std::vector<int64_t>{Ubig_child1.rows},
                                    {});
      matmul(Ubig_child1, Utransfer_splits[0], Ubig_splits[0]);
      matmul(Ubig_child2, Utransfer_splits[1], Ubig_splits[1]);

      Ubig_parent.insert(row, level, std::move(Ubig));
    }
    else {                      // insert an identity block if there are no admissible blocks.
      A.U.insert(row, level, generate_identity_matrix(max_rank * 2, max_rank));
      Ubig_parent.insert(row, level, generate_identity_matrix(block_size, max_rank));
    }
  }

  for (int64_t row = 0; row < nblocks; ++row) {
    for (int64_t col = 0; col < row; ++col) {
      if (A.is_admissible.exists(row, col, level) && A.is_admissible(row, col, level)) {
        Matrix& Urow_actual = Ubig_parent(row, level);
        Matrix& Ucol_actual = Ubig_parent(col, level);

        Matrix dense = generate_p2p_interactions(domain,
                                                 row * block_size, block_size,
                                                 col * block_size, block_size,
                                                 kernel);
        Matrix S_block = matmul(matmul(Urow_actual, dense, true, false), Ucol_actual);
        A.S.insert(row, col, level, std::move(S_block));
      }
    }
  }

  return std::move(Ubig_parent);
}

static void
construct_H2_strong_leaf_nodes(Hatrix::SymmetricSharedBasisMatrix& A,
                               const Hatrix::Domain& domain,
                               const int64_t N, const int64_t nleaf,
                               const int64_t max_rank, const double accuracy) {
  Hatrix::Matrix Utemp, Stemp, Vtemp;
  int64_t rank;
  const int64_t nblocks = pow(2, A.max_level);


  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < nblocks; ++j) {
      if (A.is_admissible.exists(i, j, A.max_level) && !A.is_admissible(i, j, A.max_level)) {
        Hatrix::Matrix Aij = generate_p2p_interactions(domain,
                                                       i * nleaf, nleaf,
                                                       j * nleaf, nleaf,
                                                       kernel);
        A.D.insert(i, j, A.max_level, std::move(Aij));
      }
    }
  }

  for (int64_t i = 0; i < nblocks; ++i) {
    Hatrix::Matrix AY(nleaf, nleaf);
    for (int64_t j = 0; j < nblocks; ++j) {
      if (!A.is_admissible.exists(i, j, A.max_level) ||
          (A.is_admissible.exists(i, j, A.max_level) && A.is_admissible(i, j, A.max_level))) {
        Hatrix::Matrix Aij = generate_p2p_interactions(domain,
                                                       i * nleaf, nleaf,
                                                       j * nleaf, nleaf,
                                                       kernel);
        AY += Aij;
      }
    }

    std::tie(Utemp, Stemp, Vtemp, rank) = svd_like_compression(AY, max_rank, accuracy);
    Utemp.shrink(Utemp.rows, rank);
    A.U.insert(i, A.max_level, std::move(Utemp));
  }

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < i; ++j) {
      Hatrix::Matrix dense = generate_p2p_interactions(domain,
                                                       i * nleaf, nleaf,
                                                       j * nleaf, nleaf,
                                                       kernel);
      A.S.insert(i, j, A.max_level,
                   Hatrix::matmul(Hatrix::matmul(A.U(i, A.max_level), dense, true),
                                  A.U(j, A.max_level)));
    }
  }
}

static void
construct_H2_strong(Hatrix::SymmetricSharedBasisMatrix& A, const Hatrix::Domain& domain,
                    const int64_t N, const int64_t nleaf, const int64_t max_rank,
                    const double accuracy) {
  construct_H2_strong_leaf_nodes(A, domain, N, nleaf, max_rank, accuracy);
  RowLevelMap Uchild = A.U;

  for (int64_t level = A.max_level - 1; level >= A.min_level; --level) {
    Uchild = generate_H2_strong_transfer_matrices(A, Uchild, domain, N, nleaf,
                                                  max_rank, level, accuracy);
  }
}

static Matrix
make_complement(const Matrix& Q) {
  Hatrix::Matrix Q_F(Q.rows, Q.rows);
  Hatrix::Matrix Q_full, R;
  std::tie(Q_full, R) = qr(Q,
                           Hatrix::Lapack::QR_mode::Full,
                           Hatrix::Lapack::QR_ret::OnlyQ);

  for (int64_t i = 0; i < Q_F.rows; ++i) {
    for (int64_t j = 0; j < Q_F.cols - Q.cols; ++j) {
      Q_F(i, j) = Q_full(i, j + Q.cols);
    }
  }

  for (int64_t i = 0; i < Q_F.rows; ++i) {
    for (int64_t j = 0; j < Q.cols; ++j) {
      Q_F(i, j + (Q_F.cols - Q.cols)) = Q(i, j);
    }
  }
  return Q_F;
}

static void
update_row_cluster_basis(SymmetricSharedBasisMatrix& A,
                         const int64_t block,
                         const int64_t level,
                         const int64_t max_rank,
                         double accuracy,
                         RowColLevelMap<Matrix>& F,
                         RowMap<Matrix>& r) {
  const int64_t block_size = A.D(block, block, level).rows;
  Matrix fill_in(block_size, block_size);

  for (int64_t j = 0; j < block; ++j) {
    if (exists_and_admissible(A, block, j, level) && F.exists(block, j, level)) {
      fill_in += matmul(F(block, j, level), F(block, j, level), false, true);
      // F.erase(block, j, level);
    }
  }

  fill_in += matmul(matmul(A.U(block, level), US(block, level)),
                    A.U(block, level), false, true);
  Matrix Q,R;
  Matrix Si, Vi;
  int64_t rank;

  std::tie(Q, R, rank) = error_pivoted_qr(fill_in,
                                          accuracy * 1e-1,
                                          false, false);

  Q.shrink(A.U(block,level).rows, max_rank);
  R.shrink(max_rank, max_rank);

  Vi.destructive_resize(R.rows, R.cols);
  Si.destructive_resize(R.rows, R.rows);

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

static void
update_row_S_blocks(SymmetricSharedBasisMatrix& A,
                    int64_t block, int64_t level,
                    const Hatrix::RowMap<Hatrix::Matrix>& r) {
  // update the S blocks with the new projected basis.
  for (int64_t j : far_neighbours(block, level)) {
    if (j < block) {
      A.S(block, j, level) = matmul(r(block), A.S(block, j, level));
    }
  }
}

static void
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

static void
update_row_cluster_basis_and_S_blocks(Hatrix::SymmetricSharedBasisMatrix& A,
                                      Hatrix::RowColLevelMap<Hatrix::Matrix>& F,
                                      Hatrix::RowMap<Hatrix::Matrix>& r,
                                      const int64_t block,
                                      const int64_t level,
                                      const int64_t max_rank,
                                      const double accuracy) {
  bool found_row_fill_in = false;
  for (int64_t j = 0; j < block; ++j) {
    if (F.exists(block, j, level)) {
      found_row_fill_in = true;
      break;
    }
  }

  if (found_row_fill_in) {    // update row cluster bases
    update_row_cluster_basis(A, block, level, max_rank, accuracy, F, r);
    update_row_S_blocks(A, block, level, r);
    update_row_transfer_basis(A, block, level, r);
  }
}

static void
update_col_cluster_basis(SymmetricSharedBasisMatrix& A,
                         const int64_t block,
                         const int64_t level,
                         const int64_t max_rank,
                         const double accuracy,
                         RowColLevelMap<Matrix>& F,
                         RowMap<Matrix>& t) {
  const int64_t nblocks = pow(2, level);
  const int64_t block_size = A.D(block, block, level).cols;

  Matrix fill_in(block_size, block_size);
  for (int64_t i = block+1; i < nblocks; ++i) {
    if (exists_and_admissible(A, i, block, level) && F.exists(i, block, level)) {
      fill_in += matmul(F(i, block, level), F(i, block, level), true, false);
      // F.erase(i, block, level);
    }
  }

  fill_in += matmul(A.U(block, level),
                    matmul(US(block, level), A.U(block, level), false, true));

  Matrix col_concat_T = transpose(fill_in);
  Matrix Q,R;
  Matrix Si, Vi;

  int64_t rank;
  std::tie(Q, R, rank) = error_pivoted_qr(col_concat_T,
                                          accuracy * 1e-1,
                                          false, false);

  Q.shrink(A.U(block,level).rows, max_rank);
  R.shrink(max_rank, max_rank);

  Vi.destructive_resize(R.rows, R.cols);
  Si.destructive_resize(R.rows, R.rows);

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

static void
update_col_S_blocks(Hatrix::SymmetricSharedBasisMatrix& A,
                    const int64_t block,
                    const int64_t level,
                    const Hatrix::RowMap<Hatrix::Matrix>& t) {
  int64_t nblocks = pow(2, level);
  // update the S blocks in this column.
  for (int64_t i : far_neighbours(block, level)) {
    if (i > block) {
      A.S(i, block, level) = matmul(A.S(i, block, level), t(block), false, true);
    }
  }
}

static void
update_col_transfer_basis(Hatrix::SymmetricSharedBasisMatrix& A,
                          const int64_t block,
                          const int64_t level,
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


static void
update_col_cluster_basis_and_S_blocks(Hatrix::SymmetricSharedBasisMatrix& A,
                                      Hatrix::RowColLevelMap<Hatrix::Matrix>& F,
                                      Hatrix::RowMap<Hatrix::Matrix>& t,
                                      const int64_t block,
                                      const int64_t level,
                                      const int64_t max_rank,
                                      const double accuracy) {
  bool found_col_fill_in = false;
  int64_t nblocks = pow(2, level);

  for (int64_t i = block+1; i < nblocks; ++i) {
    if (F.exists(i, block, level)) {
      found_col_fill_in = true;
      break;
    }
  }

  if (found_col_fill_in) {
    update_col_cluster_basis(A, block, level, max_rank, accuracy, F, t);
    update_col_S_blocks(A, block, level, t);
    update_col_transfer_basis(A, block, level, t);
  }
}

static void
multiply_complements(SymmetricSharedBasisMatrix& A,
                     const int64_t block,
                     const int64_t level,
                     const int64_t max_rank) {

  // left multiply with the complement along the (symmetric) row.
  auto diagonal_splits = split_dense(A.D(block, block, level),
                                     A.D(block, block, level).rows - max_rank,
                                     A.D(block, block, level).cols - max_rank);

  // copy the upper triangle into the lower triangle.
  for (int i = 0; i < A.D(block, block, level).rows; ++i) {
    for (int j = i+1; j < A.D(block, block, level).cols; ++j) {
      A.D(block, block, level)(i,j) = A.D(block, block, level)(j,i);
    }
  }

  auto U_F = make_complement(A.U(block, level));
  A.D(block, block, level) = matmul(matmul(U_F, A.D(block, block, level), true), U_F);

  for (int64_t j = 0; j < block; ++j) {
    if (exists_and_inadmissible(A, block, j, level)) {
      auto D_splits =
        A.D(block, j, level).split({},
                                   std::vector<int64_t>{A.D(block, j, level).cols -
                                                        max_rank});
      D_splits[1] = matmul(U_F, D_splits[1], true);
    }
  }

  for (int64_t i = block+1; i < pow(2, level); ++i) {
    if (exists_and_inadmissible(A, i, block, level)) {
      A.D(i, block, level) = matmul(A.D(i, block, level), U_F);
    }
  }
}

static void
factorize_diagonal(SymmetricSharedBasisMatrix& A,
                   const int64_t block,
                   const int64_t level,
                   const int64_t max_rank) {
  int nleaf = A.D(block, block, level).rows, rank = max_rank;
  Matrix& diagonal = A.D(block, block, level);
  auto diagonal_splits = split_dense(diagonal,
                                     nleaf-rank,
                                     nleaf-rank);
  cholesky(diagonal_splits[0], Hatrix::Lower);
  solve_triangular(diagonal_splits[0], diagonal_splits[2], Hatrix::Right, Hatrix::Lower,
                   false, true, 1.0);
  syrk(diagonal_splits[2], diagonal_splits[3], Hatrix::Lower, false, -1, 1);
}

static void
triangle_reduction(SymmetricSharedBasisMatrix& A,
                   const int64_t block,
                   const int64_t level,
                   const int64_t max_rank) {
  Matrix& diagonal = A.D(block, block, level);
  auto diagonal_splits = split_dense(diagonal,
                                     diagonal.rows - max_rank,
                                     diagonal.cols - max_rank);
  const Matrix& Dcc = diagonal_splits[0];
  const int64_t nblocks = pow(2, level);

  for (int64_t i = block+1; i < nblocks; ++i) {
    if (exists_and_inadmissible(A, i, block, level)) {
      auto D_i_block_splits =
        A.D(i, block, level).split(
                                   {},
                                   std::vector<int64_t>{
                                     A.D(i, block, level).cols - max_rank});
      solve_triangular(Dcc, D_i_block_splits[0], Hatrix::Right, Hatrix::Lower, false, true, 1);
    }
  }

  // TRSM with co blocks behind the diagonal on the 'block' row.
  for (int64_t j = 0; j < block; ++j) {
    if (exists_and_inadmissible(A, block, j, level)) {
      auto D_block_j_splits =
        A.D(block, j, level).split(std::vector<int64_t>{A.D(block, j, level).rows - max_rank},
                                   std::vector<int64_t>{A.D(block, j, level).cols - max_rank});
      solve_triangular(Dcc, D_block_j_splits[1], Hatrix::Left, Hatrix::Lower, false, false, 1.0);
    }
  }
}


static void
factorize_level(SymmetricSharedBasisMatrix& A,
                const int64_t level,
                const int64_t max_rank,
                const double accuracy,
                RowColLevelMap<Matrix>& F,
                RowMap<Matrix>& r,
                RowMap<Matrix>& t) {
  const int64_t parent_level = level - 1;
  const int64_t nblocks = pow(2, level);

  for (int64_t block = 0; block < nblocks; ++block) {
    update_row_cluster_basis_and_S_blocks(A, F, r, block,
                                          level, max_rank, accuracy);
    update_col_cluster_basis_and_S_blocks(A, F, t, block,
                                          level, max_rank, accuracy);

    multiply_complements(A, block, level, max_rank);
    factorize_diagonal(A, block, level, max_rank);
    triangle_reduction(A, block, level, max_rank);
    // compute_schurs_complement(A, block, level);
    // compute_fill_ins(A, block, level, F);
  } // for (int block = 0; block < nblocks; ++block)

}

static void
factorize_H2_strong(SymmetricSharedBasisMatrix& A,
                    const int64_t N, const int64_t nleaf, const int64_t max_rank,
                    const double accuracy) {
  int64_t level = A.max_level;
  RowColLevelMap<Matrix> F;

  for (; level >= A.min_level; --level) {
    RowMap<Matrix> r, t;

    factorize_level(A, level, max_rank, accuracy, F, r, t);

    const int64_t parent_level = level-1;
    const int64_t parent_nblocks = pow(2, parent_level);
  }
}

static Matrix
solve_H2_strong(SymmetricSharedBasisMatrix& A, Matrix& b) {
  Matrix x(b);

  return x;
}

int main(int argc, char ** argv) {
  const int64_t N = argc > 1 ? atol(argv[1]) : 256;
  const int64_t leaf_size = argc > 2 ? atol(argv[2]) : 32;
  const double accuracy = argc > 3 ? atof(argv[3]) : 1.e-5;
  const int64_t max_rank = argc > 4 ? atol(argv[4]) : 30;
  const int64_t random_matrix_size = argc > 5 ? atol(argv[5]) : 100;
  const double admis = argc > 6 ? atof(argv[6]) : 1.0;

  // Specify kernel function
  // 0: Laplace Kernel
  // 1: Yukawa Kernel
  // 2: Matern kernel
  const int64_t kernel_type = argc > 7 ? atol(argv[7]) : 0;

  // Specify underlying geometry
  // 0: Unit Circular
  // 1: Unit Cubical
  // 2: StarsH Uniform Grid
  const int64_t geom_type = argc > 8 ? atol(argv[8]) : 0;
  const int64_t ndim  = argc > 9 ? atol(argv[9]) : 2;

  // Specify compressed representation
  // 0: BLR2
  // 1: H2
  const int64_t matrix_type = argc > 10 ? atol(argv[10]) : 1;

  const double add_diag = 1e-4;
  const double alpha = 1;
  const double sigma = 1.0;
  const double nu = 0.03;
  const double smoothness = 0.5;

  switch(kernel_type) {
  case 0:                       // laplace
    kernel = [&](const std::vector<double>& c_row,
                 const std::vector<double>& c_col) {
      if (ndim == 1) {
        return Hatrix::greens_functions::laplace_1d_kernel(c_row, c_col, add_diag);
      }
      else if (ndim == 2) {
        return Hatrix::greens_functions::laplace_2d_kernel(c_row, c_col, add_diag);
      }
      else {
        return Hatrix::greens_functions::laplace_3d_kernel(c_row, c_col, add_diag);
      }
    };
    break;
  case 1:                       // yukawa
    kernel = [&](const std::vector<double>& c_row,
                 const std::vector<double>& c_col) {
      return Hatrix::greens_functions::yukawa_kernel(c_row, c_col, alpha, add_diag);
    };
    break;
  case 2:                       // matern
    kernel = [&](const std::vector<double>& c_row,
                 const std::vector<double>& c_col) {
      return Hatrix::greens_functions::matern_kernel(c_row, c_col, sigma, nu, smoothness);
    };
    break;
  }

  Hatrix::Domain domain(N, ndim);
  switch(geom_type) {
  case 1:                       // cube mesh
    domain.generate_grid_particles();
    break;
  default:                      // circle / sphere mesh
    domain.generate_circular_particles();
  }
  domain.cardinal_sort_and_cell_generation(leaf_size);

  // Initialize H2 matrix class.
  Hatrix::SymmetricSharedBasisMatrix A;
  A.max_level = log2(N / leaf_size);
  A.generate_admissibility(domain, matrix_type == 1,
                           Hatrix::ADMIS_ALGORITHM::DUAL_TREE_TRAVERSAL, admis);
  populate_near_far_lists(A);
  // Construct H2 strong admis matrix.
  construct_H2_strong(A, domain, N, leaf_size, max_rank, accuracy);

  // Factorize the strong admissiblity H2 matrix.
  factorize_H2_strong(A, N, leaf_size, max_rank, accuracy);

  // Generate verfication vector from a full-accuracy dense matrix.
  Matrix A_dense = Hatrix::generate_p2p_interactions(domain, kernel);
  Matrix x = Hatrix::generate_random_matrix(N, 1);
  Matrix b = Hatrix::matmul(A_dense, x);
  Matrix x_solve = solve_H2_strong(A, b);

  double rel_error = Hatrix::norm(x_solve - x) / Hatrix::norm(x);

  return 0;
}
