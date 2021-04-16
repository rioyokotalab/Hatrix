#include "Hatrix/Hatrix.h"

#include "gtest/gtest.h"

#include <tuple>


class MatMulTests : public testing::TestWithParam<std::tuple<int, int, int>> {};

TEST_P(MatMulTests, matmul) {
  int m, n, k;
  std::tie(m, n, k) = GetParam();
  Hatrix::Matrix A(m, k), B(k, n), C(m, n);
  A = 2;
  B = 4;
  C = 1;
  Hatrix::Matrix A_check(A), B_check(B), C_check(C);
  Hatrix::matmul(A, B, C, false, false, 1., 1.);

  // Manual matmul
  for (int i=0; i<m; ++i) {
    for (int j=0; j<n; ++j) {
      for (int k_=0; k_<k; ++k_) {
        C_check(i, j) += A_check(i, k_) * B_check(k_, j);
      }
    }
  }

  // Check result
  for (int i=0; i<m; ++i) {
    for (int j=0; j<n; ++j) {
      EXPECT_DOUBLE_EQ(C_check(i, j), C(i, j));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
  BLAS, MatMulTests,
  testing::Combine(
    testing::Values(16, 32, 64),
    testing::Values(16, 32),
    testing::Values(16, 32, 64)
  ),
  [](const testing::TestParamInfo<MatMulTests::ParamType>& info) {
    std::string name = (
      "m" + std::to_string(std::get<0>(info.param))
      + "k" + std::to_string(std::get<1>(info.param))
      + "n" + std::to_string(std::get<2>(info.param))
    );
    return name;
  }
);
