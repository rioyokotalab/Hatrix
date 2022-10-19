#include <algorithm>
#include <exception>
#include <cmath>
#include <random>

#include "Hatrix/Hatrix.h"
#include "franklin/franklin.hpp"

#include "h2_operations.hpp"

using namespace Hatrix;

static std::vector<Matrix>
split_dense(const Matrix& dense, int64_t row_split, int64_t col_split) {
  return dense.split(std::vector<int64_t>(1, row_split),
                     std::vector<int64_t>(1, col_split));
}

static inline bool
exists_and_inadmissible(const Hatrix::SymmetricSharedBasisMatrix& A,
                        const int64_t i, const int64_t j, const int64_t level) {
  return A.is_admissible.exists(i, j, level) && !A.is_admissible(i, j, level);
}

Hatrix::RowColLevelMap<Matrix> D_pre, D_post;

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
factorize_level(SymmetricSharedBasisMatrix& A,
                int64_t level, RowColLevelMap<Matrix>& F,
                RowMap<Matrix>& r, RowMap<Matrix>& t) {
  int64_t nblocks = pow(2, level);

  for (int64_t block = 0; block < nblocks; ++block) {
    int64_t block_size = A.D(block, block, level).rows,
      rank = A.ranks(block, level);
    bool found_row_fill_in = false, found_col_fill_in = false;

    for (int64_t i = 0; i < nblocks; ++i) {
      if (F.exists(i, block, level)) {
        found_row_fill_in = true;
        break;
      }
    }

    for (int64_t j = 0; j < nblocks; ++j) {
      if (F.exists(block, j, level)) {
        found_col_fill_in = true;
        break;
      }
    }


    if (found_row_fill_in) {    // update row cluster bases
    }

    if (found_col_fill_in) {    // update col cluster bases
    }

    auto U_F = make_complement(A.U(block, level));

    // left multiply with the complement along the (symmetric) row.
    for (int64_t j = 0; j <= block; ++j) {
      if (exists_and_inadmissible(A, block, j, level)) {
        A.D(block, j, level) = matmul(U_F, A.D(block, j, level), true, false);
      }
    }

    // right multiply with the transpose of the complement
    for (int64_t i = block; i < nblocks; ++i) {
      if (exists_and_inadmissible(A, i, block, level)) {
        A.D(i, block, level) = matmul(A.D(i, block, level), U_F);
      }
    }

    // start with partial cholesky factorization.
    auto diagonal_splits = split_dense(A.D(block, block, level),
                                       block_size - rank,
                                       block_size - rank);
    Matrix& Dcc = diagonal_splits[0];
    cholesky(Dcc, Hatrix::Lower);

    // TRSM with oc blocks along the 'block' column.
    for (int64_t i = block; i < nblocks; ++i) {
      if (exists_and_inadmissible(A, i, block, level)) {
        const int64_t row_splits = A.D(i, block, level).rows - A.ranks(i, level);
        const int64_t col_splits = A.D(i, block, level).cols - A.ranks(block, level);
        auto D_i_block_splits = split_dense(A.D(i, block, level),
                                            row_splits, col_splits);

        solve_triangular(Dcc, D_i_block_splits[2], Hatrix::Right, Hatrix::Lower,
                         false, true, 1.0);
      }
    }

    // ------- Compute Schur's complements --------

    // Schur's complement for cc blocks. cc = cc - cc * cc
    for (int64_t i = block + 1; i < nblocks; ++i) {
      if (exists_and_inadmissible(A, i, block, level)) {
        Matrix& D_i_block = A.D(i, block, level);
        const int64_t row_splits = D_i_block.rows - A.ranks(i, level);
        const int64_t col_splits = D_i_block.cols - A.ranks(block, level);
        auto D_i_block_splits = split_dense(D_i_block, row_splits, col_splits);

        for (int64_t j = block + 1; j <= i; ++j) {
          if (exists_and_inadmissible(A, j, block, level)) {
            Matrix& D_j_block = A.D(j, block, level);
            const int64_t row_splits = D_j_block.rows - A.ranks(block, level);
            const int64_t col_splits = D_j_block.cols - A.ranks(j, level);
            auto D_j_block_splits = split_dense(D_j_block, row_splits, col_splits);

            if (exists_and_inadmissible(A, i, j, level)) {
              Matrix& D_ij = A.D(i, j, level);
              const int64_t row_splits = D_ij.rows - A.ranks(i, level);
              const int64_t col_splits = D_ij.cols - A.ranks(j, level);
              auto D_ij_splits = split_dense(D_ij, row_splits, col_splits);

              if (i == j) {
                syrk(D_i_block_splits[0], D_ij_splits[0], Hatrix::Lower, false, -1.0, 1.0);
              }
              else {
                matmul(D_i_block_splits[0], D_j_block_splits[0], D_ij_splits[0], false, true,
                       -1.0, 1.0);
              }
            }
          }
        }
      }
    }

    // Schur's complement for oc blocks. oc = oc - oc * cc
    for (int64_t i = 0; i < nblocks; ++i) {
      for (int64_t j = block + 1; j <= i; ++j) {
        if (i == j) {
        }
        else {
        }
      }
    }

    // Schur's complement for co blocks. co = co - cc * oc.T
    for (int64_t i = block+1; i < nblocks; ++i) {
      for (int64_t j = 0; j < nblocks; ++j) {
        if (i == j) {

        }
        else {

        }
      }
    }

    // Schur's complement for oo blocks. o
    // if i==j: oo = oo - oc * oc.T (syrk)
    // else: oo = oo - oc * co (gemm)
    for (int64_t i = block; i < nblocks; ++i) {
      for (int64_t j = block; j <= i; ++j) {
        if (exists_and_inadmissible(A, j, block, level) &&
            exists_and_inadmissible(A, i, block, level)) {
          auto D_i_block_splits = split_dense(A.D(i, block, level),
                                              A.D(i, block, level).rows - A.ranks(i, level),
                                              A.D(i, block, level).cols - A.ranks(block, level));
          auto D_j_block_splits = split_dense(A.D(j, block, level),
                                              A.D(j, block, level).rows - A.ranks(j, level),
                                              A.D(j, block, level).cols - A.ranks(block, level));

          if (exists_and_inadmissible(A, i, j, level)) {
            Matrix& D_block = A.D(i, j, level);
            const int64_t row_split = D_block.rows - A.ranks(i, level);
            const int64_t col_split = D_block.cols - A.ranks(j, level);
            auto D_splits = split_dense(D_block, row_split, col_split);

            if (i == j) {
              syrk(D_i_block_splits[2], D_splits[3], Hatrix::Lower, false, -1.0, 1.0);
            }
            else {
              // matmul(D_);
            }
          }
        }
      }
    }
  } // for (int block = 0; block < nblocks; ++block)
}

void
save_matrix_pre(Hatrix::SymmetricSharedBasisMatrix& A, int64_t level) {
  int64_t nblocks = pow(2, level);
}

void save_matrix_post(Hatrix::SymmetricSharedBasisMatrix& A, int64_t level) {
  int64_t nblocks = pow(2, level);
}

static void
check_product(Hatrix::SymmetricSharedBasisMatrix& A, int64_t level) {
  int64_t nblocks = pow(2, level);

}

void
factorize(Hatrix::SymmetricSharedBasisMatrix& A) {
  RowColLevelMap<Matrix> F;
  int64_t level;

  for (level = A.max_level; level >= A.min_level; --level) {
    int64_t nblocks = pow(2, level);
    RowMap<Matrix> r, t;
    // PLU of one level of the H2 matrix.
    save_matrix_pre(A, level);
    factorize_level(A, level, F, r, t);
    save_matrix_post(A, level);
    check_product(A, level);

    const int64_t parent_level = level-1;
    const int64_t parent_nblocks = pow(2, parent_level);

    // Update coupling matrices of each admissible block to add fill in contributions.
    for (int64_t i = 0; i < nblocks; ++i) {
      for (int64_t j = 0; j <= i; ++j) {

      }
    }

    // Propagate fill-in to upper level admissible blocks
    if (parent_level > 0) {

    }

    // Put identity bases when an all dense row/col is encountered.
    for (int64_t block = 0; block < parent_nblocks; ++block) {
      if (!A.U.exists(block, parent_level)) {
        int64_t c1 = block * 2;
        int64_t c2 = block * 2 + 1;
        int64_t rank_c1 = A.ranks(c1, level);
        int64_t rank_c2 = A.ranks(c2, level);
        int64_t rank_parent = std::max(rank_c1, rank_c2);
        Matrix Utransfer =
          generate_identity_matrix(rank_c1 + rank_c2, rank_parent);

        if (r.exists(c1)) r.erase(c1);
        if (r.exists(c2)) r.erase(c2);
        A.U.insert(block, parent_level, std::move(Utransfer));
      }
    }

    // Merge and permute to prepare for the next level.

    // Merge the unfactorized parts.
    for (int i = 0; i < pow(2, parent_level); ++i) {
      for (int j = 0; j <= i; ++j) {
        if (exists_and_inadmissible(A, i, j, parent_level)) {
          std::vector<int64_t> i_children({i * 2, i * 2 + 1}), j_children({j * 2, j * 2 + 1});
          const int64_t c_rows = A.ranks(i_children[0], level) + A.ranks(i_children[1], level);
          const int64_t c_cols = A.ranks(j_children[0], level) + A.ranks(j_children[1], level);
          Matrix D_unelim(c_rows, c_cols);
          auto D_unelim_splits = split_dense(D_unelim,
                                             A.ranks(i_children[0], level),
                                             A.ranks(j_children[0], level));

          for (int ic1 = 0; ic1 < 2; ++ic1) {
            for (int jc2 = 0; jc2 < ((i == j) ? (ic1+1) : 2); ++jc2) {
              int64_t c1 = i_children[ic1], c2 = j_children[jc2];

              if (A.is_admissible.exists(c1, c2, level)) {
                if (!A.is_admissible(c1, c2, level)) {
                  Matrix& D_c1c2 = A.D(c1, c2, level);
                  const int64_t row_splits = D_c1c2.rows - A.ranks(c1, level);
                  const int64_t col_splits = D_c1c2.cols - A.ranks(c2, level);

                  auto D_splits = split_dense(D_c1c2, row_splits, col_splits);
                  D_unelim_splits[ic1 * 2 + jc2] = D_splits[3];
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

static void
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
    Matrix prod = matmul(U_F, x_level_split[block], true);
    x_level_split[block] = prod;
  }

  // forward substitution with cc blocks
  for (int64_t block = 0; block < nblocks; ++block) {
    int64_t rank = A.ranks(block, level);
    int64_t row_split = A.D(block, block, level).rows - rank;
    int64_t col_split = A.D(block, block, level).cols - rank;
    auto block_splits = split_dense(A.D(block, block, level), row_split, col_split);

    Matrix x_block(x_level_split[block]);
    auto x_block_splits = x_block.split(std::vector<int64_t>(1, row_split), {});

    solve_triangular(block_splits[0], x_block_splits[0], Hatrix::Left, Hatrix::Lower, false,
                     false, 1.0);
    matmul(block_splits[2], x_block_splits[0], x_block_splits[1], false, false, -1.0, 1.0);
    x_level_split[block] = x_block;
  }
}

static void
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

  // backward substition using cc blocks
  for (int64_t block = nblocks-1; block >= 0; --block) {
    int64_t rank = A.ranks(block, level);
    int64_t row_split = A.D(block, block, level).rows - rank;
    int64_t col_split = A.D(block, block, level).cols - rank;
    auto block_splits = split_dense(A.D(block, block, level), row_split, col_split);

    Matrix x_block(x_level_split[block]);
    auto x_block_splits = x_block.split(std::vector<int64_t>(1, row_split), {});
    matmul(block_splits[2], x_block_splits[1], x_block_splits[0], true, false, -1.0, 1.0);
    solve_triangular(block_splits[0], x_block_splits[0],
                     Hatrix::Left, Hatrix::Lower, false, true, 1.0);
    x_level_split[block] = x_block;
  }

  for (int64_t block = nblocks-1; block >= 0; --block) {
    auto V_F = make_complement(A.U(block, level));
    Matrix prod = matmul(V_F, x_level_split[block]);
    x_level_split[block] = prod;
  }
}

static int64_t
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

static int64_t
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

  // forward substitution.
  for (int64_t level = A.max_level; level >= A.min_level; --level) {
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


  x_splits[1] = x_last;

  // backward substitution.
  for (int64_t level = A.min_level; level <= A.max_level; ++level) {
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
multiply_S(Hatrix::SymmetricSharedBasisMatrix& A,
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
matmul(SymmetricSharedBasisMatrix& A, const Matrix& x) {
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
