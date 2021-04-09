#include "Hatrix/Hatrix.h"

#include "gtest/gtest.h"

#include <iostream>


TEST(MatrixTests, Constructor) {
  int block_size = 16;
  Hatrix::Matrix A(block_size, block_size);

  for (int i=0; i<block_size; ++i) for (int j=0; j<block_size; ++j) {
    ASSERT_EQ(A(i, j), 0);
  }
}

TEST(MatrixTests, CopyConstructor) {
  int block_size = 16;
  Hatrix::Matrix A(block_size, block_size);
  for (int i=0; i<block_size; ++i) for (int j=0; j<block_size; ++j) {
    A(i, j) = i*block_size+j;
  }
  Hatrix::Matrix A_copy(A);

  for (int i=0; i<block_size; ++i) for (int j=0; j<block_size; ++j) {
    ASSERT_EQ(A(i, j), A_copy(i, j));
  }
}

TEST(MatrixTests, shrink) {
  int block_size = 16;
  Hatrix::Matrix A(block_size, block_size);
  for (int i=0; i<block_size; ++i) for (int j=0; j<block_size; ++j) {
    A(i, j) = i*block_size+j;
  }
  Hatrix::Matrix A_copy(A);

  int shrunk_size = 8;
  A.shrink(shrunk_size, shrunk_size);

  // Check result
  for (int i=0; i<shrunk_size; ++i) for (int j=0; j<shrunk_size; ++j) {
    ASSERT_EQ(A(i, j), A_copy(i, j));
  }
}
