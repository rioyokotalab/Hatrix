#include "Hatrix/Hatrix.h"

#include "gtest/gtest.h"

#include <iostream>
#include <vector>


TEST(BlockDense, getrf) {
  int block_size = 4;
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
  for (int i=0; i<block_size; ++i) for (int j=0; j<block_size; ++j) {
    A[0][0](i, j) = i*2*block_size + j;
    A[0][1](i, j) = i*2*block_size + block_size + j;
    A[1][0](i, j) = (i+block_size)*2*block_size + j;
    A[1][1](i, j) = (i+block_size)*2*block_size + block_size + j;
  }
  for (int i=0; i<block_size; ++i) {
    A[0][0](i, i) += 100;
    A[1][1](i, i) += 100;
  }

  // b = A*x
  Hatrix::Matrix x0(block_size, 1), x1(block_size, 1);
  for (int i=0; i<block_size; ++i) {
    x0(i, 0) = i+1;
    x1(i, 0) = block_size+i+1;
  }
  Hatrix::Matrix b0(block_size, 1), b1(block_size, 1);
  Hatrix::gemm(A[0][0], x0, b0, 'N', 'N', 1, 0);
  Hatrix::gemm(A[0][1], x1, b0, 'N', 'N', 1, 1);
  Hatrix::gemm(A[1][0], x0, b1, 'N', 'N', 1, 0);
  Hatrix::gemm(A[1][1], x1, b1, 'N', 'N', 1, 1);

  // Block LU
  Hatrix::getrf(A[0][0]);
  Hatrix::trsm(A[0][0], A[0][1], 'L', 'L', 'N', 'U', 1);
  Hatrix::trsm(A[0][0], A[1][0], 'R', 'U', 'N', 'N', 1);
  Hatrix::gemm(A[1][0], A[0][1], A[1][1], 'N', 'N', -1, 1);
  Hatrix::getrf(A[1][1]);

  // Forward substitution
  Hatrix::trsm(A[0][0], b0, 'L', 'L', 'N', 'U', 1);
  Hatrix::gemm(A[1][0], b0, b1, 'N', 'N', -1, 1);
  Hatrix::trsm(A[1][1], b1, 'L', 'L', 'N', 'U', 1);
  // Backward substitution
  Hatrix::trsm(A[1][1], b1, 'L', 'U', 'N', 'N', 1);
  Hatrix::gemm(A[0][1], b1, b0, 'N', 'N', -1, 1);
  Hatrix::trsm(A[0][0], b0, 'L', 'U', 'N', 'N', 1);

  // Check result
  for (int i=0; i<block_size; ++i) {
    EXPECT_DOUBLE_EQ(x0(i, 0), b0(i, 0));
  }
  for (int i=0; i<block_size; ++i) {
    EXPECT_DOUBLE_EQ(x1(i, 0), b1(i, 0));
  }
}
