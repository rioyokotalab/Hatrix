#include "Hatrix/Hatrix.h"

#include "gtest/gtest.h"

#include <iostream>


TEST(MatrixTests, Constructor) {
  int block_size = 16;
  Hatrix::Matrix A(block_size, block_size);

  for (int i=0; i<block_size; ++i) for (int j=0; j<block_size; ++j) {
    EXPECT_EQ(0, A(i, j));
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
    EXPECT_EQ(A(i, j), A_copy(i, j));
  }
}

TEST(MatrixTests, shrink) {
  int block_size = 16;
  Hatrix::Matrix A(block_size, block_size);
  for (int i=0; i<block_size; ++i) for (int j=0; j<block_size; ++j) {
    A(i, j) = i*block_size+j;
  }
  Hatrix::Matrix A_shrink(A);

  int shrunk_size = 8;
  A_shrink.shrink(shrunk_size, shrunk_size);

  // Check result
  for (int i=0; i<shrunk_size; ++i) for (int j=0; j<shrunk_size; ++j) {
    EXPECT_EQ(A(i, j), A_shrink(i, j));
  }
}

TEST(MatrixTests, MoveConstructor) {
  int block_size = 16;
  Hatrix::Matrix A(block_size, block_size);
  for (int i=0; i<block_size; ++i) for (int j=0; j<block_size; ++j) {
    A(i, j) = i*block_size+j;
  }
  Hatrix::Matrix A_copy = A;

  Hatrix::Matrix A_move(std::move(A));
  // Check result
  for (int i=0; i<block_size; ++i) for (int j=0; j<block_size; ++j) {
    EXPECT_EQ(A_copy(i, j), A_move(i, j));
  }
}

TEST(MatrixTests, MoveAssignment) {
  int block_size = 16;
  Hatrix::Matrix A(block_size, block_size);
  for (int i=0; i<block_size; ++i) for (int j=0; j<block_size; ++j) {
    A(i, j) = i*block_size+j;
  }
  Hatrix::Matrix A_copy(A), A_move(A);

  A_move = std::move(A);
  // Check result
  for (int i=0; i<block_size; ++i) for (int j=0; j<block_size; ++j) {
    EXPECT_EQ(A_copy(i, j), A_move(i, j));
  }
}
