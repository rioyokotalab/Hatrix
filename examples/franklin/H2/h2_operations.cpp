#include <algorithm>
#include <exception>
#include <cmath>

#include "Hatrix/Hatrix.h"
#include "franklin/franklin.hpp"

#include "h2_operations.hpp"

using namespace Hatrix;

Matrix
matmul(const SymmetricSharedBasisMatrix& A, const Matrix& x) {
  int leaf_nblocks = pow(2, A.max_level);
  std::vector<Matrix> x_hat;
  auto x_splits = x.split(leaf_nblocks, 1);

  // V leaf nodes
  for (int i = 0; i < leaf_nblocks; ++i) {
    x_hat.push_back(matmul(A.U(i, A.max_level), x_splits[i], true, false, 1.0));
  }

  int64_t x_hat_offset = 0;     // index offset for the x_hat array.
  for (int64_t level = A.max_level - 1; level > A.min_level; --level) {
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
  int64_t nblocks = pow(2, A.min_level);
  std::vector<Matrix> b_hat;
  for (int64_t i = 0; i < nblocks; ++i) {
    b_hat.push_back(Matrix(A.ranks(i, A.min_level), 1));
  }

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < i; ++j) {
      if (A.is_admissible.exists(i, j, level) && A.is_admissible(i, j, level)) {
        matmul(A.S(i, j, A.min_level), x_hat[x_hat_offset+j], b_hat[i], false, false, 1.0, 1.0);
        matmul(A.S(i, j, A.min_level), x_hat[x_hat_offset+i], b_hat[j], true, false, 1.0, 1.0);
      }
    }
  }

  // Multiply the S blocks at the top-most level with the corresponding xhat.
  Matrix b1_2 = matmul(A.S(1, 0, level), x_hat[x_hat_offset]);
  Matrix b1_1 = matmul(A.S(1, 0, level), x_hat[x_hat_offset+1], true, false);
  b_hat.push_back(b1_1);
  b_hat.push_back(b1_2);
  int b_hat_offset = 0;

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
