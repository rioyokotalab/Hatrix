#include <exception>

#include "Hatrix/Hatrix.h"

#include "functions.hpp"
#include "matrix_construction.hpp"
#include "SymmetricSharedBasisMatrix.hpp"
#include "Domain.hpp"
#include "Args.hpp"

using namespace Hatrix;

static void coarsen_blocks(SymmetricSharedBasisMatrix& A, int64_t level) {
  int64_t child_level = level + 1;
  int64_t nblocks = pow(2, level);
  for (int64_t i = 0; i < nblocks; ++i) {
    std::vector<int64_t> row_children({i * 2, i * 2 + 1});
    for (int64_t j = 0; j <= i; ++j) {
      std::vector<int64_t> col_children({j * 2, j * 2 + 1});

      bool admis_block = true;
      for (int64_t c1 = 0; c1 < 2; ++c1) {
        for (int64_t c2 = 0; c2 < 2; ++c2) {
          if (A.is_admissible.exists(row_children[c1], col_children[c2], child_level) &&
              !A.is_admissible(row_children[c1], col_children[c2], child_level)) {
            admis_block = false;
          }
        }
      }

      if (admis_block) {
        for (int64_t c1 = 0; c1 < 2; ++c1) {
          for (int64_t c2 = 0; c2 < 2; ++c2) {
            A.is_admissible.erase(row_children[c1], col_children[c2], child_level);
          }
        }
      }

      A.is_admissible.insert(i, j, level, std::move(admis_block));
    }
  }
}

static void diagonal_admis_init(SymmetricSharedBasisMatrix& A, const Args& opts, int64_t level) {
  int64_t nblocks = pow(2, level); // pow since we are using diagonal based admis.
  A.level_blocks.push_back(nblocks);
  if (level == 0) { return; }
  if (level == A.height) {
    for (int64_t i = 0; i < nblocks; ++i) {
      for (int64_t j = 0; j <= i; ++j) {
        A.is_admissible.insert(i, j, level, std::abs(i - j) > opts.admis);
      }
    }
  }
  else {
    coarsen_blocks(A, level);
  }

  diagonal_admis_init(A, opts, level-1);
}

void init_diagonal_admis(SymmetricSharedBasisMatrix& A, const Args& opts) {
  A.height = int64_t(log2(opts.N / opts.nleaf));
  diagonal_admis_init(A, opts, A.height);
  std::reverse(std::begin(A.level_blocks), std::end(A.level_blocks));
  A.is_admissible.insert(0, 0, 0, false);
}

void init_geometry_admis(SymmetricSharedBasisMatrix& A, const Args& opts) {
  throw std::exception();
}

static Matrix
generate_column_block(int64_t block, int64_t block_size,
                      int64_t level,
                      const SymmetricSharedBasisMatrix& A,
                      const Matrix& dense,
                      const Matrix& rand) {
  int64_t nblocks = A.level_blocks[level];
  auto dense_splits = dense.split(nblocks, nblocks);
  auto rand_splits = rand.split(nblocks, 1);
  Matrix AY(block_size, rand.cols);

  int64_t index = 0;
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

  if (opts.accuracy > 1) {        // constant rank compression
    std::tie(Ui, pivots) = pivoted_qr(AY, opts.max_rank);
  }
  else {
    int64_t rank;
    std::tie(Ui, pivots, rank) = error_pivoted_qr(AY, opts.accuracy);
    A.ranks.insert(block, level, std::move(rank));
  }

  return std::move(Ui);
}

static void
generate_leaf_nodes(const Domain& domain,
                    SymmetricSharedBasisMatrix& A,
                    const Matrix& dense,
                    const Matrix& rand,
                    const Args& opts) {
  int64_t nblocks = A.level_blocks[A.height];
  auto dense_splits = dense.split(nblocks, nblocks);

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < nblocks; ++j) {
      if (A.is_admissible.exists(i, j, A.height) &&
          !A.is_admissible(i, j, A.height)) {
        // TODO: Make this only a lower triangular matrix with the diagonal.
        Matrix Aij(dense_splits[i * nblocks + j], true);
        A.D.insert(i, j, A.height, std::move(Aij));
      }
    }
  }

  for (int64_t i = 0; i < nblocks; ++i) {
    A.U.insert(i,
               A.height,
               generate_column_bases(i,
                                     domain.boxes[i].num_particles,
                                     A.height,
                                     A,
                                     dense,
                                     rand,
                                     opts));
  }

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < i; ++j) {
      if (A.is_admissible.exists(i, j, A.height) &&
          A.is_admissible(i, j, A.height)) {
        Matrix Sblock = matmul(matmul(A.U(i, A.height),
                                      dense_splits[i * nblocks + j], true, false),
                               A.U(j, A.height));
        A.S.insert(i, j, A.height, std::move(Sblock));
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
  if (opts.accuracy > 1) {      // constant rank factorization
    std::tie(Utransfer, pivots) = pivoted_qr(temp, opts.max_rank);
  }
  else {
    int64_t rank;
    std::tie(Utransfer, pivots, rank) = error_pivoted_qr(temp, opts.accuracy);
    A.ranks.insert(node, level, std::move(rank));
  }

  return std::move(Utransfer);
}

static bool
row_has_admissible_blocks(const SymmetricSharedBasisMatrix& A, int64_t row, int64_t level) {
  bool has_admis = false;
  for (int64_t i = 0; i < A.level_blocks[level]; ++i) {
    if (!A.is_admissible.exists(row, i, level) ||
        (A.is_admissible.exists(row, i, level) && A.is_admissible(row, i, level))) {
      has_admis = true;
      break;
    }
  }

  return has_admis;
}


static RowLevelMap
generate_transfer_matrices(const int64_t level,
                           const RowLevelMap& Uchild,
                           SymmetricSharedBasisMatrix& A,
                           const Matrix& dense,
                           const Matrix& rand,
                           const Args& opts) {
  int64_t nblocks = A.level_blocks[level];
  auto dense_splits = dense.split(nblocks, nblocks);

  RowLevelMap Ubig_parent;
  for (int64_t node = 0; node < nblocks; ++node) {
    int64_t c1 = node * 2;
    int64_t c2 = node * 2 + 1;
    int64_t child_level = level + 1;

    if (row_has_admissible_blocks(A, node, level) && A.height != 1) {
      const Matrix& Ubig_c1 = Uchild(c1, child_level);
      const Matrix& Ubig_c2 = Uchild(c2, child_level);
      int64_t block_size = Ubig_c1.rows + Ubig_c2.rows;

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
  }

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < i; ++j) {
      if (A.is_admissible.exists(i, j, level) &&
          A.is_admissible(i, j, level)) {
        Matrix Sdense = matmul(matmul(Ubig_parent(i, level), dense_splits[i * nblocks + j], true, false),
                               Ubig_parent(j, level));

        A.S.insert(i, j, level, std::move(Sdense));
      }
    }
  }

  return Ubig_parent;
}

void construct_h2_matrix_miro(SymmetricSharedBasisMatrix& A,
                              const Domain& domain,
                              const Args& opts) {
  int64_t P = 100;
  Matrix dense = generate_p2p_matrix(domain, opts.kernel);
  Matrix rand = generate_random_matrix(opts.N, P);
  generate_leaf_nodes(domain, A, dense, rand, opts);

  RowLevelMap Uchild = A.U;

  for (int64_t level = A.height-1; level > 0; --level) {
    Uchild = generate_transfer_matrices(level, Uchild, A, dense, rand, opts);
  }
}

static void
actually_print_h2_structure(const SymmetricSharedBasisMatrix& A, const int64_t level) {
  if (level == 0) { return; }
  int64_t nblocks = A.level_blocks[level];
  std::cout << "LEVEL: " << level << " NBLOCKS: " << nblocks << std::endl;
  for (int64_t i = 0; i < nblocks; ++i) {
    if (level == A.height) {
      std::cout << A.U(i, A.height).rows << " ";
    }
    std::cout << "| " ;
    for (int64_t j = 0; j < nblocks; ++j) {
      if (A.is_admissible.exists(i, j, level)) {
        std::cout << A.is_admissible(i, j, level) << " | " ;
      }
      else {
        std::cout << "  | ";
      }
    }
    std::cout << std::endl;
  }

  std::cout << std::endl;

  actually_print_h2_structure(A, level-1);
}

void print_h2_structure(const SymmetricSharedBasisMatrix& A) {
  actually_print_h2_structure(A, A.height);
}

double
reconstruct_accuracy(const SymmetricSharedBasisMatrix& A,
                     const Domain& domain,
                     const Matrix& dense,
                     const Args& opts) {
  double error = 0;
  double dense_norm = 0;
  int64_t nblocks = A.level_blocks[A.height];

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < nblocks; ++j) {
      if (A.is_admissible.exists(i, j, A.height) && !A.is_admissible(i, j, A.height)) {
        Matrix actual = generate_p2p_interactions(domain, i, j, opts.kernel);
        Matrix expected = A.D(i, j, A.height);
        error += pow(norm(actual - expected), 2);
        dense_norm += pow(norm(actual), 2);
      }
    }
  }

  for (int64_t level = A.height; level > 0; --level) {
    int64_t nblocks = A.level_blocks[level];

    for (int64_t row = 0; row < nblocks; ++row) {
      for (int64_t col = 0; col < row; ++col) {
        if (A.is_admissible.exists(row, col, level) && A.is_admissible(row, col, level)) {
          Matrix Ubig = A.Ubig(row, level);
          Matrix Vbig = A.Ubig(col, level);

          Matrix expected_matrix = matmul(matmul(Ubig, A.S(row, col, level)),
                                          Vbig, false, true);
          Matrix actual_matrix =
            Hatrix::generate_p2p_interactions(domain, row, col, level,
                                              A.height, opts.kernel);

          dense_norm += pow(norm(actual_matrix), 2);
          error += pow(norm(expected_matrix - actual_matrix), 2);
        }
      }
    }

  }

  return std::sqrt(error / dense_norm);
}
