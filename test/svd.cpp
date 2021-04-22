#include "Hatrix/Hatrix.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <iostream>

class SVDTests : public testing::TestWithParam<std::tuple<int, int>>{};

TEST(SVDTests, truncated_svd) {
  int m, n, k;
  m = n = 50;
  k = 5;

  Hatrix::Matrix A(m,k);
  Hatrix::Matrix B(k,n);
  Hatrix::Matrix C(m,n);

  std::random_device rd;
  std::mt19937 gen(rd());
  gen.seed(0);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (int i=0; i<m; ++i) {
    for (int j=0; j<k; ++j) {
      A(i,j) = dist(gen);
      B(j,i) = dist(gen);
    }
  }

  Hatrix::matmul(A, B, C, false, false, 1, 0);
  Hatrix::Matrix C_copy(C);
  Hatrix::Matrix U(m, m), S(m, m), V(m, n);
  Hatrix::svd(C, U, S, V);

  Hatrix::Matrix truncU(m, k), truncS(k, k), truncV(k, n);
  for (int i=0; i<m; ++i) {
    for (int j=0; j<k; ++j) {
      truncU(i,j) = U(i,j);
      truncV(j,i) = V(j,i);
    }
  }
  for (int i=0; i<k; ++i) {
    truncS(i,i) = S(i,i);
  }

  Hatrix::Matrix temp(m,k), C_rec(m,n);
  Hatrix::matmul(truncU, truncS, temp, false, false, 1, 0);
  Hatrix::matmul(temp, truncV, C_rec, false, false, 1, 0);

  for (int i=0; i<m; ++i) {
    for (int j=0; j<n; ++j) {
      EXPECT_NEAR(C_copy(i, j), C_rec(i, j), 10e-14);
    }
  }
}

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