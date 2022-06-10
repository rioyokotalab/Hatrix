#include <cstdint>
#include <iostream>
#include <tuple>

#include "Hatrix/Hatrix.h"
#include "gtest/gtest.h"

class RQTests
    : public testing::TestWithParam<std::tuple<int64_t, int64_t, int64_t>> {};

TEST_P(RQTests, rq) {
  Hatrix::Context::init();
  int64_t m, n, k;
  std::tie(m, n, k) = GetParam();
  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix R(m, k), Q(k, n);
  Hatrix::Matrix A_copy(A);
  Hatrix::rq(A, R, Q);
  Hatrix::Matrix RQ = Hatrix::matmul(R, Q);
  // Check accuracy
  for (int64_t i = 0; i < RQ.rows; i++) {
    for (int64_t j = 0; j < RQ.cols; j++) {
      EXPECT_NEAR(A_copy(i, j), RQ(i, j), 1e-12);
    }
  }
  // Check orthogonality
  Hatrix::Matrix QQt = Hatrix::matmul(Q, Q, false, true);
  for (int64_t i = 0; i < QQt.rows; i++) {
    for (int64_t j = 0; j < QQt.cols; j++) {
      if (i == j)
        EXPECT_NEAR(QQt(i, j), 1.0, 1e-12);
      else
        EXPECT_NEAR(QQt(i, j), 0.0, 1e-12);
    }
  }

  Hatrix::Context::finalize();
}


INSTANTIATE_TEST_SUITE_P(LAPACK, RQTests,
                         testing::Values(std::make_tuple(16, 16, 16),
                                         std::make_tuple(16, 8, 8),
                                         std::make_tuple(8, 16, 16),
                                         std::make_tuple(8, 16, 8)));
