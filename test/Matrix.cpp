#include <cstdint>
#include <utility>

#include "Hatrix/Hatrix.h"
#include "gtest/gtest.h"

TEST(MatrixTests, Constructor) {
  int64_t block_size = 16;
  Hatrix::Matrix A(block_size, block_size);

  for (int64_t i = 0; i < block_size; ++i)
    for (int64_t j = 0; j < block_size; ++j) {
      EXPECT_EQ(0, A(i, j));
    }
}

TEST(MatrixTests, CopyConstructor) {
  int64_t block_size = 16;
  Hatrix::Matrix A = Hatrix::generate_random_matrix(block_size, block_size);
  Hatrix::Matrix A_copy(A);
  for (int64_t i = 0; i < block_size; ++i) {
    for (int64_t j = 0; j < block_size; ++j) {
      EXPECT_EQ(A(i, j), A_copy(i, j));
    }
  }
}

TEST(MatrixTests, CopyAssignment) {
  int64_t block_size = 16;
  Hatrix::Matrix A = Hatrix::generate_random_matrix(block_size, block_size);
  Hatrix::Matrix A_copy(block_size, block_size);

  A_copy = A;
  for (int64_t i = 0; i < block_size; ++i) {
    for (int64_t j = 0; j < block_size; ++j) {
      EXPECT_EQ(A(i, j), A_copy(i, j));
    }
  }
}

TEST(MatrixTests, MoveConstructor) {
  int64_t block_size = 16;
  Hatrix::Matrix A = Hatrix::generate_random_matrix(block_size, block_size);
  Hatrix::Matrix A_copy = A;

  Hatrix::Matrix A_move(std::move(A));
  // Check result
  for (int64_t i = 0; i < block_size; ++i) {
    for (int64_t j = 0; j < block_size; ++j) {
      EXPECT_EQ(A_copy(i, j), A_move(i, j));
    }
  }
}

TEST(MatrixTests, MoveAssignment) {
  int64_t block_size = 16;
  Hatrix::Matrix A = Hatrix::generate_random_matrix(block_size, block_size);
  Hatrix::Matrix A_copy(A), A_move(A);

  A_move = std::move(A);
  // Check result
  for (int64_t i = 0; i < block_size; ++i) {
    for (int64_t j = 0; j < block_size; ++j) {
      EXPECT_EQ(A_copy(i, j), A_move(i, j));
    }
  }
}

TEST(MatrixTests, shrink) {
  int64_t block_size = 16;
  Hatrix::Matrix A = Hatrix::generate_random_matrix(block_size, block_size);
  Hatrix::Matrix A_shrunk(A);

  int64_t shrunk_size = 8;
  A_shrunk.shrink(shrunk_size, shrunk_size);

  // Check result
  for (int64_t i = 0; i < shrunk_size; ++i) {
    for (int64_t j = 0; j < shrunk_size; ++j) {
      EXPECT_EQ(A(i, j), A_shrunk(i, j));
    }
  }
}