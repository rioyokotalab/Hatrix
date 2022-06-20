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

    std::cout << "ranks: " ;
    for (int i = 0; i < pow(2, level); ++i) {
      std::cout << "i -> " << i << " l -> " << level <<  " " <<  A.ranks(i, level) << "\n";
    }
    std::cout << std::endl;

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
}

Matrix
solve(const SymmetricSharedBasisMatrix& A, const Matrix& x) {
  Matrix b(x);

  return b;
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
