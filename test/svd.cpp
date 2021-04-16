#include "Hatrix/Hatrix.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <iostream>

class SVDTests : public testing::TestWithParam<std::tuple<int, int>>{};

TEST_P(SVDTests, svd){
  int m, n;
  std::tie(m, n) = GetParam();

  Hatrix::Matrix A(m, n);
  A = 5.5;

  int s_dim = A.min_dim();
  Hatrix::Matrix A_copy(A);
  Hatrix::Matrix U(m, s_dim), S(s_dim, s_dim), V(s_dim, n), A_rebuilt(m, n);
  Hatrix::svd(A, U, S, V);
  Hatrix::Matrix temp(m, s_dim);
  Hatrix::matmul(U, S, temp, false, false, 1, 0);
  Hatrix::matmul(temp, V, A_rebuilt, false, false, 1, 0);

    // Check result
  for (int i=0; i<A.rows; ++i) {
    for (int j=0; j<A.cols; ++j) {
      EXPECT_DOUBLE_EQ(A_rebuilt(i, j), A_copy(i, j));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
  LAPACK, SVDTests,
  testing::Combine(
    testing::Values(8, 16),
    testing::Values(8, 16)
  ),
  [](const testing::TestParamInfo<SVDTests::ParamType>& info) {
    std::string name = (
      "m" + std::to_string(std::get<0>(info.param))
      + "n" + std::to_string(std::get<1>(info.param))
    );
    return name;
  }
);