#include "franklin/franklin.hpp"

#include "SymmetricSharedBasisMatrix.hpp"
#include "operations.hpp"

using namespace Hatrix;

void factorize(SymmetricSharedBasisMatrix& A) {

}

Matrix
solve(const SymmetricSharedBasisMatrix& A, const Matrix& x) {
  Matrix b(x);

  return b;
}

Matrix
matmul(const SymmetricSharedBasisMatrix& A, const Matrix& x) {
  int leaf_nblocks = A.level_blocks[A.height];
  std::vector<Matrix> x_hat;
  auto x_splits = x.split(leaf_nblocks, 1);

  // V leaf nodes
  for (int i = 0; i < leaf_nblocks; ++i) {
    x_hat.push_back(matmul(A.U(i, A.height), x_splits[i], true, false, 1.0));
  }

  int x_hat_offset = 0;
  for (int64_t level = A.height - 1; level > 0; --level) {
    int64_t nblocks = A.level_blocks[level];
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

    x_hat_offset += A.level_blocks[level+1];
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

  for (int64_t level = 1; level < A.height; ++level) {
    int64_t nblocks = A.level_blocks[level];
    int64_t child_level = level + 1;
    x_hat_offset -= A.level_blocks[child_level];

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
    b_hat_offset += A.level_blocks[level];
  }


  Matrix b(x.rows, 1);
  auto b_splits = b.split(leaf_nblocks, 1);
  for (int i = 0; i < leaf_nblocks; ++i) {
    Matrix temp = matmul(A.U(i, A.height), b_hat[b_hat_offset + i]) +
      matmul(A.D(i, i, A.height), x_splits[i]);
    b_splits[i] = temp;
  }

  return b;
}
