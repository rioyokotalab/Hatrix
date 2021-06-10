#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <tuple>

#include "Hatrix/Hatrix.h"
#include "gtest/gtest.h"

class TriangularMatMulTests
    : public testing::TestWithParam<std::tuple<
          int64_t, int64_t, Hatrix::Side, Hatrix::Mode, bool, bool, double> > {
};

TEST_P(TriangularMatMulTests, triangular_matmul) {
  int64_t M, N;
  Hatrix::Side side;
  Hatrix::Mode uplo;
  bool transA, diag;
  double alpha;
  Hatrix::Context::init();
  std::tie(M, N, side, uplo, transA, diag, alpha) = GetParam();
  Hatrix::Matrix B = Hatrix::generate_random_matrix(M, N);
  Hatrix::Matrix A = Hatrix::generate_random_matrix(
      side == Hatrix::Left ? M : N, side == Hatrix::Left ? M : N);
  Hatrix::Matrix B_copy(B);
  Hatrix::Matrix A_tri(A);
  // Construct triangular A_tri
  for (int64_t j = 0; j < A_tri.cols; j++) {
    A_tri(j, j) = diag ? 1. : A(j, j);
    if (uplo == Hatrix::Lower)
      for (int i = 0; i < j; i++) A_tri(i, j) = 0.;
    else
      for (int i = j + 1; i < A_tri.rows; i++) A_tri(i, j) = 0.;
  }

  Hatrix::triangular_matmul(A, B, side, uplo, transA, diag, alpha);
  Hatrix::Context::join();

  // Manual matmul
  // B_check = A_tri*B_copy or B_copy*A_tri
  Hatrix::Matrix B_check(M, N);
  for (int64_t i = 0; i < M; i++) {
    for (int64_t j = 0; j < N; j++) {
      if (side == Hatrix::Left) {
        for (int64_t k = 0; k < M; k++) {
          if (transA)
            B_check(i, j) += alpha * A_tri(k, i) * B_copy(k, j);
          else
            B_check(i, j) += alpha * A_tri(i, k) * B_copy(k, j);
        }
      } else {
        for (int64_t k = 0; k < N; k++) {
          if (transA)
            B_check(i, j) += alpha * B_copy(i, k) * A_tri(j, k);
          else
            B_check(i, j) += alpha * B_copy(i, k) * A_tri(k, j);
        }
      }
    }
  }

  // Check result
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      EXPECT_NEAR(B_check(i, j), B(i, j), 10e-14);
    }
  }
  Hatrix::Context::finalize();
}

INSTANTIATE_TEST_SUITE_P(
    Params, TriangularMatMulTests,
    testing::Combine(testing::Values(16, 32), testing::Values(16, 32),
                     testing::Values(Hatrix::Left, Hatrix::Right),
                     testing::Values(Hatrix::Upper, Hatrix::Lower),
                     testing::Values(true, false), testing::Values(true, false),
                     testing::Values(-1., -0.5, 0., 0.5, 1.)));
