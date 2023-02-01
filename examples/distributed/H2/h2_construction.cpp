#include <algorithm>
#include <exception>
#include <cmath>
#include <random>

#include "Hatrix/Hatrix.h"
#include "distributed/distributed.hpp"

#include "h2_operations.hpp"
#include "h2_construction.hpp"

using namespace Hatrix;

// This is actually RowLevelMap
Hatrix::RowColMap<std::vector<int64_t>> near_neighbours, far_neighbours;

static void
dual_tree_traversal(SymmetricSharedBasisMatrix& A, const Cell& Ci, const Cell& Cj,
                    const Domain& domain, const Args& opts) {
  int64_t i_level = Ci.level;
  int64_t j_level = Cj.level;

  bool well_separated = false;
  if (i_level == j_level &&
      ((!opts.use_nested_basis && i_level == A.max_level) ||
       opts.use_nested_basis)) {
    double distance = 0;
    for (int64_t k = 0; k < opts.ndim; ++k) {
      distance += pow(Ci.center[k] - Cj.center[k], 2);
    }
    distance = sqrt(distance);

    if (distance * opts.admis > (Ci.radius + Cj.radius)) {
      // well-separated blocks.
      well_separated = true;
    }

    bool val = well_separated;
    A.is_admissible.insert(Ci.level_index, Cj.level_index, i_level, std::move(val));
  }

  // Only descend down the tree if you are currently at a higher level and the blocks
  // at the current level are inadmissible. You then want to refine the tree further
  // since it has been found that the higher blocks are inadmissible.
  //
  // Alternatively, to create a BLR2 matrix you want to down to the finest level of granularity
  // anyway and populate the blocks at that level. So that puts another OR condition to check
  // if the use of nested basis is enabled.
  if (i_level <= j_level && Ci.cells.size() > 0 && (!well_separated || !opts.use_nested_basis)) {
    // j is at a higher level and i is not leaf.
    dual_tree_traversal(A, Ci.cells[0], Cj, domain, opts);
    dual_tree_traversal(A, Ci.cells[1], Cj, domain, opts);
  }
  else if (j_level <= i_level && Cj.cells.size() > 0 && (!well_separated || !opts.use_nested_basis)) {
    // i is at a higheer level and j is not leaf.
    dual_tree_traversal(A, Ci, Cj.cells[0], domain, opts);
    dual_tree_traversal(A, Ci, Cj.cells[1], domain, opts);
  }
}

static void
build_dense_level(SymmetricSharedBasisMatrix& A) {
  int64_t level = A.max_level - 1;
  int64_t nblocks = pow(2, level);

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < nblocks; ++j) {
      A.is_admissible.insert(i, j, level, false);
    }
  }
}

void init_geometry_admis(SymmetricSharedBasisMatrix& A,
                         const Domain& domain, const Args& opts) {
  A.max_level = domain.tree.height() - 1;
  dual_tree_traversal(A, domain.tree, domain.tree, domain, opts);
  // Using BLR2 so need an 'artificial' dense matrix level at max_level-1
  // for accumulation of the partial factorization.
  if (!opts.use_nested_basis) {
    build_dense_level(A);
  }
  A.min_level = 0;
  for (int64_t l = A.max_level; l > 0; --l) {
    int64_t nblocks = pow(2, l);
    bool all_dense = true;
    for (int64_t i = 0; i < nblocks; ++i) {
      for (int64_t j = 0; j < nblocks; ++j) {
        if (A.is_admissible.exists(i, j, l) && A.is_admissible(i, j, l)) {
          all_dense = false;
        }
      }
    }

    if (all_dense) {
      A.min_level = l;
      break;
    }
  }

  if (A.max_level != A.min_level) { A.min_level++; }

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

static Matrix
generate_column_block(int64_t block, int64_t block_size,
                      int64_t level,
                      const SymmetricSharedBasisMatrix& A,
                      const Matrix& dense,
                      const Matrix& rand) {
  int64_t nblocks = pow(2, level);
  auto dense_splits = dense.split(nblocks, nblocks);
  auto rand_splits = rand.split(nblocks, 1);
  Matrix AY(block_size, rand.cols);

  for (int64_t j = 0; j < nblocks; ++j) {
    if (A.is_admissible.exists(block, j, level) &&
        !A.is_admissible(block, j, level)) { continue; }
    matmul(dense_splits[block * nblocks + j], rand_splits[j], AY, false, false, 1.0, 1.0);
  }

  return AY;
}

static Matrix
generate_column_bases(int64_t block, int64_t block_size, int64_t level,
                      SymmetricSharedBasisMatrix& A,
                      const Matrix& dense,
                      const Matrix&rand,
                      const Args& opts) {
  Matrix AY = generate_column_block(block, block_size, level, A, dense, rand);
  Matrix Ui;
  std::vector<int64_t> pivots;
  int64_t rank;

  if (opts.accuracy == -1) {        // constant rank compression
    rank = opts.max_rank;
    std::tie(Ui, pivots) = pivoted_qr(AY, rank);
  }
  else {
    std::tie(Ui, pivots, rank) = error_pivoted_qr_max_rank(AY, opts.accuracy, (int64_t)opts.max_rank);
  }

  Matrix _U, _S, _V; double _error;
  std::tie(_U, _S, _V, _error) = truncated_svd(AY, rank);
  US.insert(block, level, std::move(_S));

  A.ranks.insert(block, level, std::move(rank));

  return Ui;
}

static void
generate_leaf_nodes(const Domain& domain,
                    SymmetricSharedBasisMatrix& A,
                    const Matrix& dense,
                    const Matrix& rand,
                    const Args& opts) {
  int64_t nblocks = pow(2, A.max_level);
  auto dense_splits = dense.split(nblocks, nblocks);

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j : near_neighbours(i, A.max_level)) {
      // TODO: Make this only a lower triangular matrix with the diagonal.
      // Will need a special type.
      Matrix Aij(dense_splits[i * nblocks + j], true);
      A.D.insert(i, j, A.max_level, std::move(Aij));
    }
  }

  for (int64_t i = 0; i < nblocks; ++i) {
    A.U.insert(i,
               A.max_level,
               generate_column_bases(i,
                                     domain.cell_size(i, A.max_level),
                                     A.max_level,
                                     A,
                                     dense,
                                     rand,
                                     opts));
  }

  for (int64_t i = 0; i < nblocks; ++i) {
    // for (int64_t j : far_neighbours(i, A.max_level)) {
    for (int64_t j = 0; j < i; ++j) {
      if (A.is_admissible.exists(i, j, A.max_level) &&
          A.is_admissible(i, j, A.max_level)) {
        Matrix Sblock = matmul(matmul(A.U(i, A.max_level),
                                      dense_splits[i * nblocks + j], true, false),
                               A.U(j, A.max_level));
        A.S.insert(i, j, A.max_level, std::move(Sblock));
      }
    }
  }
}

static Matrix
generate_U_transfer_matrix(const Matrix& Ubig_c1,
                           const Matrix& Ubig_c2,
                           const int64_t node,
                           const int64_t block_size,
                           const int64_t level,
                           SymmetricSharedBasisMatrix& A,
                           const Matrix& dense,
                           const Matrix& rand,
                           const Args& opts) {
  Matrix col_block = generate_column_block(node, block_size, level, A, dense, rand);
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
    std::tie(Utransfer, pivots) = pivoted_qr(temp, rank);
  }
  else {
    std::tie(Utransfer, pivots, rank) = error_pivoted_qr_max_rank(temp, opts.accuracy, opts.max_rank);
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
    if (!A.is_admissible.exists(row, i, level) || exists_and_admissible(A, row, i, level)) {
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
                           const Matrix& rand,
                           const Args& opts) {
  int64_t nblocks = pow(2, level);
  auto dense_splits = dense.split(nblocks, nblocks);

  std::mt19937 gen(0);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  RowLevelMap Ubig_parent;
  for (int64_t node = 0; node < nblocks; ++node) {
    int64_t c1 = node * 2;
    int64_t c2 = node * 2 + 1;
    int64_t child_level = level + 1;

    const Matrix& Ubig_c1 = Uchild(c1, child_level);
    const Matrix& Ubig_c2 = Uchild(c2, child_level);
    int64_t block_size = Ubig_c1.rows + Ubig_c2.rows;

    if (row_has_admissible_blocks(A, node, level) && A.max_level != 1) {
      Matrix Utransfer = generate_U_transfer_matrix(Ubig_c1,
                                                    Ubig_c2,
                                                    node,
                                                    block_size,
                                                    level,
                                                    A,
                                                    dense,
                                                    rand,
                                                    opts);
      auto Utransfer_splits = Utransfer.split(std::vector<int64_t>(1, A.ranks(c1, child_level)),
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
                 generate_identity_matrix(A.ranks(c1, child_level) + A.ranks(c2, child_level),
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
  Matrix rand = generate_random_matrix(opts.N, P);
  generate_leaf_nodes(domain, A, dense, rand, opts);

  RowLevelMap Uchild = A.U;

  for (int64_t level = A.max_level-1; level > 0; --level) {
    Uchild = generate_transfer_matrices(domain, level, Uchild, A, dense, rand, opts);
  }
}