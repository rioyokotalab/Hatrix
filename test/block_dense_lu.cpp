#include "Hatrix/Hatrix.h"

#include "gtest/gtest.h"

#include <cstdint>
using std::int64_t;
#include <vector>


TEST(BlockDense, lu) {
  int64_t block_size = 4;
  std::vector<std::vector<Hatrix::Matrix>> A(2);
  A[0] = std::vector<Hatrix::Matrix>{
    Hatrix::Matrix(block_size, block_size),
    Hatrix::Matrix(block_size, block_size)
  };
  A[1] = std::vector<Hatrix::Matrix>{
    Hatrix::Matrix(block_size, block_size),
    Hatrix::Matrix(block_size, block_size)
  };
  // Initialize to non-singular matrix
  for (int64_t i=0; i<block_size; ++i) for (int64_t j=0; j<block_size; ++j) {
    A[0][0](i, j) = i*2*block_size + j;
    A[0][1](i, j) = i*2*block_size + block_size + j;
    A[1][0](i, j) = (i+block_size)*2*block_size + j;
    A[1][1](i, j) = (i+block_size)*2*block_size + block_size + j;
  }
  for (int64_t i=0; i<block_size; ++i) {
    A[0][0](i, i) += 100;
    A[1][1](i, i) += 100;
  }

  // b = A*x
  Hatrix::Matrix x0(block_size, 1), x1(block_size, 1);
  for (int64_t i=0; i<block_size; ++i) {
    x0(i, 0) = i+1;
    x1(i, 0) = block_size+i+1;
  }
  Hatrix::Matrix b0(block_size, 1), b1(block_size, 1);
  Hatrix::matmul(A[0][0], x0, b0, false, false, 1, 0);
  Hatrix::matmul(A[0][1], x1, b0, false, false, 1, 1);
  Hatrix::matmul(A[1][0], x0, b1, false, false, 1, 0);
  Hatrix::matmul(A[1][1], x1, b1, false, false, 1, 1);

  // Block LU
  Hatrix::Matrix L0(block_size, block_size);
  Hatrix::Matrix U0(block_size, block_size);
  Hatrix::lu(A[0][0], L0, U0);
  Hatrix::solve_triangular(L0, A[0][1], Hatrix::Left, Hatrix::Lower, true);
  Hatrix::solve_triangular(U0, A[1][0], Hatrix::Right, Hatrix::Upper, false);
  Hatrix::matmul(A[1][0], A[0][1], A[1][1], false, false, -1, 1);
  Hatrix::Matrix L1(block_size, block_size);
  Hatrix::Matrix U1(block_size, block_size);
  Hatrix::lu(A[1][1], L1, U1);

  // Forward substitution
  Hatrix::solve_triangular(L0, b0, Hatrix::Left, Hatrix::Lower, true);
  Hatrix::matmul(A[1][0], b0, b1, false, false, -1, 1);
  Hatrix::solve_triangular(L1, b1, Hatrix::Left, Hatrix::Lower, true);
  // Backward substitution
  Hatrix::solve_triangular(U1, b1, Hatrix::Left, Hatrix::Upper, false);
  Hatrix::matmul(A[0][1], b1, b0, false, false, -1, 1);
  Hatrix::solve_triangular(U0, b0, Hatrix::Left, Hatrix::Upper, false);

  // Check result
  for (int64_t i=0; i<block_size; ++i) {
    EXPECT_DOUBLE_EQ(x0(i, 0), b0(i, 0));
  }
  for (int64_t i=0; i<block_size; ++i) {
    EXPECT_DOUBLE_EQ(x1(i, 0), b1(i, 0));
  }
}
