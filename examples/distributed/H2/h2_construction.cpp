#include <algorithm>
#include <exception>
#include <cmath>
#include <random>
#include <iomanip>

#include "Hatrix/Hatrix.h"
#include "distributed/distributed.hpp"

#include "h2_operations.hpp"
#include "h2_construction.hpp"

using namespace Hatrix;

static Matrix
generate_column_block(const Domain& domain,
                      int64_t block, int64_t block_size,
                      int64_t level,
                      const SymmetricSharedBasisMatrix& A,
                      const Matrix& dense,
                      const Args& opts) {
  int64_t nblocks = pow(2, level);
  auto dense_splits = dense.split(nblocks, nblocks);
  Matrix AY(block_size, block_size);

  for (int64_t j = 0; j < nblocks; ++j) {
    if (A.is_admissible.exists(block, j, level) &&
        !A.is_admissible(block, j, level)) { continue; }
    for (int ii = 0; ii < block_size; ++ii) {
      for (int jj = 0; jj < block_size; ++jj) {

        AY(ii, jj) += opts.kernel(domain.particles[block * block_size + ii].coords,
                                  domain.particles[j * block_size + jj].coords);
      }
    }
  }

  return AY;
}

static Matrix
generate_column_bases(const Domain& domain,
                      int64_t block, int64_t block_size, int64_t level,
                      SymmetricSharedBasisMatrix& A,
                      const Matrix& dense,
                      const Args& opts) {
  Matrix AY = generate_column_block(domain, block, block_size, level, A, dense, opts);
  Matrix Ui;
  int64_t rank;

  if (opts.accuracy == -1) {        // constant rank compression
    rank = opts.max_rank;
    Matrix _U, _S, _V;
    double error;
    std::tie(Ui, _S, _V, error) = truncated_svd(AY, rank);
    US.insert(block, level, std::move(_S));
  }
  else {
    Matrix _S, _V;
    std::tie(Ui, _S, _V, rank) = error_svd(AY, opts.accuracy * 1e-2, false, true);
    if (rank == Ui.rows) {
      std::cout << "@@@ COMPRESSION FAILED! @@@\n";
      abort();
    }
    if (rank > opts.max_rank) {
      Ui.shrink(Ui.rows, opts.max_rank);
      _S.shrink(opts.max_rank, opts.max_rank);
      rank = opts.max_rank;
    }
    US.insert(block, level, std::move(_S));
  }

  A.ranks.insert(block, level, std::move(rank));

  return Ui;
}

static double cond_svd(const Matrix& A) {
  Matrix copy(A, true);
  Matrix _U(A, true), _S(A, true), _V(A, true);
  double error;

  svd(copy, _U, _S, _V);

  return _S(0,0) / _S(_S.rows-1, _S.cols-1);
}


static void
generate_leaf_nodes(const Domain& domain,
                    SymmetricSharedBasisMatrix& A,
                    const Matrix& dense,
                    const Args& opts) {
  int64_t nblocks = pow(2, A.max_level);
  auto dense_splits = dense.split(nblocks, nblocks);

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j <= i; ++j) {
      if (exists_and_inadmissible(A, i, j, A.max_level)) {
        Matrix Aij(dense_splits[i * nblocks + j], true);
        A.D.insert(i, j, A.max_level, std::move(Aij));
      }
    }
  }

  for (int64_t i = 0; i < nblocks; ++i) {
    A.U.insert(i,
               A.max_level,
               generate_column_bases(domain,
                                     i,
                                     domain.cell_size(i, A.max_level),
                                     A.max_level,
                                     A,
                                     dense,
                                     opts));
  }

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < i; ++j) {
      if (exists_and_admissible(A, i, j, A.max_level)) {
        Matrix Sblock = matmul(matmul(A.U(i, A.max_level),
                                      dense_splits[i * nblocks + j], true, false),
                               A.U(j, A.max_level));
        A.S.insert(i, j, A.max_level, std::move(Sblock));
      }
    }
  }
}

static Matrix
generate_U_transfer_matrix(const Domain& domain,
                           const Matrix& Ubig_c1,
                           const Matrix& Ubig_c2,
                           const int64_t node,
                           const int64_t block_size,
                           const int64_t level,
                           SymmetricSharedBasisMatrix& A,
                           const Matrix& dense,
                           const Args& opts) {
  Matrix col_block = generate_column_block(domain, node, block_size, level, A, dense, opts);
  auto col_block_splits = col_block.split(2, 1);

  int64_t c1 = node * 2;
  int64_t c2 = node * 2 + 1;
  int64_t child_level = level + 1;
  int64_t r_c1 = A.ranks(c1, child_level);
  int64_t r_c2 = A.ranks(c2, child_level);

  Matrix temp(r_c1 + r_c2, col_block.cols);
  auto temp_splits = temp.split(std::vector<int64_t>(1, r_c1), {});

  matmul(Ubig_c1, col_block_splits[0], temp_splits[0], true, false, 1, 0);
  matmul(Ubig_c2, col_block_splits[1], temp_splits[1], true, false, 1, 0);

  Matrix Utransfer;
  std::vector<int64_t> pivots;
  int64_t rank;
  if (opts.accuracy == -1) {      // constant rank factorization
    rank = opts.max_rank;
    Matrix Si, Vi; double error;
    std::tie(Utransfer, Si, Vi, error) = truncated_svd(temp, rank);
  }
  else {
    std::tie(Utransfer, pivots, rank) =
      error_pivoted_qr_max_rank(temp, opts.accuracy, opts.max_rank);
  }

  Matrix _U, _S, _V; double _error;
  std::tie(_U, _S, _V, _error) = truncated_svd(temp, rank);
  US.insert(node, level, std::move(_S));

  A.ranks.insert(node, level, std::move(rank));

  return Utransfer;
}

static bool
row_has_admissible_blocks(const SymmetricSharedBasisMatrix& A, int64_t row,
                          int64_t level) {
  bool has_admis = false;

  for (int64_t i = 0; i < pow(2, level); ++i) {
    if (!A.is_admissible.exists(row, i, level) ||
        exists_and_admissible(A, row, i, level)) {
      has_admis = true;
      break;
    }
  }

  return has_admis;
}

static RowLevelMap
generate_transfer_matrices(const Domain& domain,
                           const int64_t level,
                           const RowLevelMap& Uchild,
                           SymmetricSharedBasisMatrix& A,
                           const Matrix& dense,
                           const Args& opts) {
  int64_t nblocks = pow(2, level);
  auto dense_splits = dense.split(nblocks, nblocks);

  RowLevelMap Ubig_parent;
  for (int64_t node = 0; node < nblocks; ++node) {
    int64_t c1 = node * 2;
    int64_t c2 = node * 2 + 1;
    int64_t child_level = level + 1;

    const Matrix& Ubig_c1 = Uchild(c1, child_level);
    const Matrix& Ubig_c2 = Uchild(c2, child_level);
    int64_t block_size = Ubig_c1.rows + Ubig_c2.rows;

    if (row_has_admissible_blocks(A, node, level) && A.max_level != 1) {
      Matrix Utransfer = generate_U_transfer_matrix(domain,
                                                    Ubig_c1,
                                                    Ubig_c2,
                                                    node,
                                                    block_size,
                                                    level,
                                                    A,
                                                    dense,
                                                    opts);
      auto Utransfer_splits =
        Utransfer.split(std::vector<int64_t>{A.ranks(c1, child_level)},
                        {});

      Matrix Ubig(block_size, A.ranks(node, level));
      auto Ubig_splits = Ubig.split(2, 1);

      matmul(Ubig_c1, Utransfer_splits[0], Ubig_splits[0]);
      matmul(Ubig_c2, Utransfer_splits[1], Ubig_splits[1]);

      A.U.insert(node, level, std::move(Utransfer));
      Ubig_parent.insert(node, level, std::move(Ubig));
    }
    else {                      // add identity transfer matrix
      int64_t rank = std::max(A.ranks(c1, child_level), A.ranks(c2, child_level));
      Ubig_parent.insert(node, level,
                         generate_identity_matrix(block_size, rank));
      A.U.insert(node, level,
                 generate_identity_matrix(A.ranks(c1, child_level) +
                                          A.ranks(c2, child_level),
                                          rank));

      A.ranks.insert(node, level, std::move(rank));
    }
  }

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < i; ++j) {
      if (A.is_admissible.exists(i, j, level) &&
          A.is_admissible(i, j, level)) {
        Matrix Sdense = matmul(matmul(Ubig_parent(i, level),
                                      dense_splits[i * nblocks + j], true, false),
                               Ubig_parent(j, level));
        A.S.insert(i, j, level, std::move(Sdense));
      }
    }
  }

  return Ubig_parent;
}

void
construct_h2_matrix_miro(SymmetricSharedBasisMatrix& A, const Domain& domain, const Args& opts) {
  int64_t P = opts.max_rank;
  Matrix dense = generate_p2p_matrix(domain, opts.kernel);
  generate_leaf_nodes(domain, A, dense, opts);

  RowLevelMap Uchild = A.U;

  for (int64_t level = A.max_level-1; level > 0; --level) {
    Uchild = generate_transfer_matrices(domain, level, Uchild, A, dense, opts);
  }
}
