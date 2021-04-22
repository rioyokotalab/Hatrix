#include "Hatrix/Hatrix.h"

#include "gtest/gtest.h"

#include <cstdint>
using std::uint64_t;
#include <utility>


TEST(MatrixTests, Constructor) {
  uint64_t block_size = 16;
  Hatrix::Matrix A(block_size, block_size);

  for (uint64_t i=0; i<block_size; ++i) for (uint64_t j=0; j<block_size; ++j) {
    EXPECT_EQ(0, A(i, j));
  }
}

TEST(MatrixTests, CopyConstructor) {
  uint64_t block_size = 16;
  Hatrix::Matrix A(block_size, block_size);
  for (uint64_t i=0; i<block_size; ++i) for (uint64_t j=0; j<block_size; ++j) {
    A(i, j) = i*block_size+j;
  }
  Hatrix::Matrix A_copy(A);

  for (uint64_t i=0; i<block_size; ++i) for (uint64_t j=0; j<block_size; ++j) {
    EXPECT_EQ(A(i, j), A_copy(i, j));
  }
}

TEST(MatrixTests, shrink) {
  uint64_t block_size = 16;
  Hatrix::Matrix A(block_size, block_size);
  for (uint64_t i=0; i<block_size; ++i) for (uint64_t j=0; j<block_size; ++j) {
    A(i, j) = i*block_size+j;
  }
  Hatrix::Matrix A_shrink(A);

  uint64_t shrunk_size = 8;
  A_shrink.shrink(shrunk_size, shrunk_size);

  // Check result
  for (uint64_t i=0; i<shrunk_size; ++i) for (uint64_t j=0; j<shrunk_size; ++j) {
    EXPECT_EQ(A(i, j), A_shrink(i, j));
  }
}

TEST(MatrixTests, MoveConstructor) {
  uint64_t block_size = 16;
  Hatrix::Matrix A(block_size, block_size);
  for (uint64_t i=0; i<block_size; ++i) for (uint64_t j=0; j<block_size; ++j) {
    A(i, j) = i*block_size+j;
  }
  Hatrix::Matrix A_copy = A;

  Hatrix::Matrix A_move(std::move(A));
  // Check result
  for (uint64_t i=0; i<block_size; ++i) for (uint64_t j=0; j<block_size; ++j) {
    EXPECT_EQ(A_copy(i, j), A_move(i, j));
  }
}

TEST(MatrixTests, MoveAssignment) {
  uint64_t block_size = 16;
  Hatrix::Matrix A(block_size, block_size);
  for (uint64_t i=0; i<block_size; ++i) for (uint64_t j=0; j<block_size; ++j) {
    A(i, j) = i*block_size+j;
  }
  Hatrix::Matrix A_copy(A), A_move(A);

  A_move = std::move(A);
  // Check result
  for (uint64_t i=0; i<block_size; ++i) for (uint64_t j=0; j<block_size; ++j) {
    EXPECT_EQ(A_copy(i, j), A_move(i, j));
  }
}
