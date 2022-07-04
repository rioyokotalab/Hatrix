#include "franklin/franklin.hpp"

#include "SymmetricSharedBasisMatrix.hpp"
#include "operations.hpp"

#define SPLIT_DENSE(dense, row_split, col_split)        \
  dense.split(std::vector<int64_t>(1, row_split),       \
              std::vector<int64_t>(1, col_split));

using namespace Hatrix;

static Matrix
make_complement(const Matrix& Q) {
  Hatrix::Matrix Q_F(Q.rows, Q.rows);
  Hatrix::Matrix Q_full, R;
  std::tie(Q_full, R) = qr(Q, Hatrix::Lapack::QR_mode::Full, Hatrix::Lapack::QR_ret::OnlyQ);

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
factorize_level(const int64_t level,
                SymmetricSharedBasisMatrix& A) {
  int64_t nblocks = pow(2, level);
  for (int64_t block = 0; block < nblocks; ++block) {
    int64_t block_rank = A.ranks(block, level);
    int64_t block_size = A.D(block, block, level).rows;
    Matrix U_F = make_complement(A.U(block, level));

    // TODO: use triangular matrix and use a trinangular matrix multiplication.
    A.D(block, block, level) = matmul((matmul(U_F, A.D(block, block, level), true, false)),
                                    U_F);
    int64_t split_size = block_size - block_rank;

    auto D_splits = SPLIT_DENSE(A.D(block, block, level), split_size, split_size);
    Matrix& Dcc = D_splits[0];
    Matrix& Dco = D_splits[1];
    Matrix& Doc = D_splits[2];
    Matrix& Doo = D_splits[3];

    cholesky(Dcc, Hatrix::Lower);
    solve_triangular(Dcc, Doc, Hatrix::Right, Hatrix::Lower, false, true, 1.0);
    syrk(Doc, Doo, Hatrix::Lower, false, -1.0, 1.0);
  }
}

void factorize(SymmetricSharedBasisMatrix& A) {
  for (int64_t level = A.max_level; level > A.min_level; --level) {
    factorize_level(level, A);

    int64_t parent_level = level-1;
    int64_t parent_nblocks = pow(2, parent_level);
    for (int64_t i = 0; i < parent_nblocks; ++i) {
      int64_t nrows = 0, row_split = A.ranks(i * 2, level);
      for (int64_t ic1 = 0; ic1 < 2; ++ic1) { nrows += A.ranks(i * 2 + ic1, level); }

      for (int64_t j = 0; j <= i; ++j) {
        int64_t ncols = 0, col_split = A.ranks(j * 2, level);
        for (int64_t jc2 = 0; jc2 < 2; ++jc2) { ncols += A.ranks(j * 2 + jc2, level); }

        if (A.is_admissible.exists(i, j, parent_level) &&
            !A.is_admissible(i, j, parent_level)) {
          Matrix D_unelim(nrows, ncols);
          auto D_unelim_splits = SPLIT_DENSE(D_unelim, row_split, col_split);

          for (int64_t ic1 = 0; ic1 < 2; ++ic1) {
            for (int64_t jc2 = 0; jc2 <= ic1; ++jc2) {
              int64_t c1 = i * 2 + ic1, c2 = j * 2 + jc2;
              if (!A.U.exists(c1, level)) { continue; }

              if (A.is_admissible.exists(c1, c2, level) && !A.is_admissible(c1, c2, level)) {
                auto D_splits = SPLIT_DENSE(A.D(c1, c2, level),
                                            A.D(c1, c2, level).rows - A.ranks(c1, level),
                                            A.D(c1, c2, level).cols - A.ranks(c2, level));
                D_unelim_splits[ic1 * 2 + jc2] = D_splits[3];
              }
              else {
                D_unelim_splits[ic1 * 2 + jc2] = A.S(c1, c2, level);
              }
            }
          }

          A.D.insert(i, j, parent_level, std::move(D_unelim));
        }
      }
    }
  }

  cholesky(A.D(0, 0, 0), Hatrix::Lower);
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
    int64_t row_split = A.D(block, block, level).rows - A.ranks(block, level);
    int64_t col_split = A.D(block, block, level).cols - A.ranks(block, level);
    auto block_splits = SPLIT_DENSE(A.D(block, block, level), row_split, col_split);

    Matrix x_block(x_level_split[block]);
    auto x_block_splits = x_block.split(std::vector<int64_t>(1, row_split), {});

    solve_triangular(block_splits[0], x_block_splits[0], Hatrix::Left, Hatrix::Lower, true);
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
    auto block_splits = SPLIT_DENSE(A.D(block, block, level), row_split, col_split);

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


Matrix
solve(const SymmetricSharedBasisMatrix& A, const Matrix& b) {
  Matrix x(b);
  int64_t level_offset = 0;
  std::vector<Matrix> x_splits;

  // forward substitution.
  for (int64_t level = A.max_level; level > A.min_level; --level) {
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

  // last block forward
  solve_triangular(A.D(0, 0, A.min_level), x_last, Hatrix::Left, Hatrix::Lower, true, false, 1.0);
  // last block backward
  solve_triangular(A.D(0, 0, A.min_level), x_last, Hatrix::Left, Hatrix::Lower, false, true, 1.0);

  x_splits[1] = x_last;

  // backward substitution.
  for (int64_t level = A.min_level+1; level <= A.max_level; ++level) {
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

Matrix
matmul(const SymmetricSharedBasisMatrix& A, const Matrix& x) {
  int leaf_nblocks = pow(2, A.max_level);
  std::vector<Matrix> x_hat;
  auto x_splits = x.split(leaf_nblocks, 1);

  // V leaf nodes
  for (int i = 0; i < leaf_nblocks; ++i) {
    x_hat.push_back(matmul(A.U(i, A.max_level), x_splits[i], true, false, 1.0));
  }

  int x_hat_offset = 0;
  for (int64_t level = A.max_level - 1; level > 0; --level) {
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
  int64_t level = 1;

  // b_hat does the product in reverse so matrices are pushed from the back.
  std::vector<Matrix> b_hat;

  // Multiply the S blocks at the top-most level with the corresponding xhat.
  Matrix b1_2 = matmul(A.S(1, 0, level), x_hat[x_hat_offset]);
  Matrix b1_1 = matmul(A.S(1, 0, level), x_hat[x_hat_offset+1], true, false);
  b_hat.push_back(b1_1);
  b_hat.push_back(b1_2);
  int b_hat_offset = 0;

  for (int64_t level = 1; level < A.max_level; ++level) {
    int64_t nblocks = pow(2, level);
    int64_t child_level = level + 1;
    x_hat_offset -= pow(2, child_level);

    for (int64_t row = 0; row < nblocks; ++row) {
      int c_r1 = row * 2, c_r2 = row * 2 + 1;

      Matrix Ub = matmul(A.U(row, level),
                         b_hat[b_hat_offset + row]);
      auto Ub_splits = Ub.split(std::vector<int64_t>(1, A.U(c_r1, child_level).cols),
                                {});

      Matrix b_r1_cl = matmul(A.S(c_r2, c_r1, child_level),
                              x_hat[x_hat_offset + c_r2],
                              true,
                              false);
      b_hat.push_back(b_r1_cl + Ub_splits[0]);

      Matrix b_r2_cl = matmul(A.S(c_r2, c_r1, child_level),
                              x_hat[x_hat_offset + c_r1]);
      b_hat.push_back(b_r2_cl + Ub_splits[1]);
    }
    b_hat_offset += nblocks;
  }


  Matrix b(x.rows, 1);
  auto b_splits = b.split(leaf_nblocks, 1);
  for (int i = 0; i < leaf_nblocks; ++i) {
    Matrix temp = matmul(A.U(i, A.max_level), b_hat[b_hat_offset + i]) +
      matmul(A.D(i, i, A.max_level), x_splits[i]);
    b_splits[i] = temp;
  }

  return b;
}
