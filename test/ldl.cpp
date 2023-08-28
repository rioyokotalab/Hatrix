#include <cstdint>
#include <string>
#include <tuple>

#include "Hatrix/Hatrix.hpp"
#include "gtest/gtest.h"

class LDLTests
  : public testing::TestWithParam<std::tuple<int64_t>> {};
class SolveDiagonalTests :
  public testing::TestWithParam<std::tuple<int64_t, int64_t, Hatrix::Side, double>> {};

TEST_P(LDLTests, ldl) {
  Hatrix::Context::init();
  int64_t m;
  std::tie(m) = GetParam();

  // Generate SPD Matrix
  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, m);
  int64_t d = m * m;
  for (int64_t i = 0; i < m; ++i) {
    A(i, i) += d--;
    for(int64_t j = i+1; j < m; j++) {
      A(i, j) = A(j, i);
    }
  }

  Hatrix::Matrix A_copy(A);
  Hatrix::Matrix A_rebuilt(m, m);
  Hatrix::ldl(A);
  
  Hatrix::Matrix L = lower_tri(A, true);
  Hatrix::Matrix D(m, m);
  for(int64_t i = 0; i < m; i++) D(i, i) = A(i, i);  
  Hatrix::matmul(L*D, L, A_rebuilt, false, true, 1, 0);

  // Check result
  for (int64_t i = 0; i < A.rows; ++i) {
    for (int64_t j = 0; j < A.cols; ++j) {
      EXPECT_FLOAT_EQ(A_rebuilt(i, j), A_copy(i, j));
    }
  }

  Hatrix::Context::finalize();
}

TEST_P(SolveDiagonalTests, solve_diagonal) {
  Hatrix::Context::init();
  int64_t m, n;
  Hatrix::Side side;
  double alpha;
  std::tie(m, n, side, alpha) = GetParam();

  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, m);
  Hatrix::Matrix B = Hatrix::generate_random_matrix(
    side == Hatrix::Left ? m : n, side == Hatrix::Left ? n : m);
  
  // Set a large value on the diagonal
  int64_t d = m * m;
  for (int64_t i = 0; i < m; ++i) {
    A(i, i) += d--;
  }

  Hatrix::Matrix B_copy(B);
  Hatrix::solve_diagonal(A, B, side, alpha);
  
  // Check result
  for (int64_t i = 0; i < B.rows; ++i) {
    for (int64_t j = 0; j < B.cols; ++j) {
      double val = B(i, j) * (side == Hatrix::Left ? A(i, i) : A(j, j)) / alpha;
      EXPECT_NEAR(B_copy(i, j), val, 1e-14);
    }
  }

  Hatrix::Context::finalize();
}

INSTANTIATE_TEST_SUITE_P(
    LAPACK, LDLTests,
    testing::Combine(testing::Values(8, 16, 32)),
    [](const testing::TestParamInfo<LDLTests::ParamType>& info) {
      std::string name = ("m" + std::to_string(std::get<0>(info.param)));
      return name;
    });

INSTANTIATE_TEST_SUITE_P(
    BLAS, SolveDiagonalTests,
    testing::Combine(testing::Values(16, 32), testing::Values(16, 32),
                     testing::Values(Hatrix::Left, Hatrix::Right),
                     testing::Values(-1., 0.5, 1.)));
