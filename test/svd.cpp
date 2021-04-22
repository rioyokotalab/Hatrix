#include "Hatrix/Hatrix.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <cassert>
#include <iostream>

class SVDTests : public testing::TestWithParam<std::tuple<int64_t, int64_t>>{};
class truncSVDTests
: public testing::TestWithParam<std::tuple<int64_t, int64_t, int64_t>>{};

void check_frobenius_norm(
  Hatrix::Matrix& A, Hatrix::Matrix& B, double tolerance
) {
  assert(A.rows == B.rows);
  assert(A.cols == B.cols);
  double norm_diff = 0;
  for (int64_t i=0; i<A.rows; ++i) {
    for (int64_t j=0; j<A.cols; ++j) {
      norm_diff += (A(i, j) - B(i, j)) * (A(i, j) - B(i, j));
    }
  }
  EXPECT_NEAR(norm_diff, tolerance, 10e-14);
}

TEST_P(truncSVDTests, truncatedSVD) {
  int64_t m, n, rank;
  std::tie(m, n, rank) = GetParam();
  Hatrix::Matrix A(m, n);
  for (int64_t i=0; i<m; ++i) {
    for (int64_t j=0; j<n; ++j) {
      A(i, j) = 1./std::abs(i - j+n);
    }
  }

  int64_t dmin = A.min_dim();
  Hatrix::Matrix A_check(A);
  Hatrix::Matrix U(m, dmin);
  Hatrix::Matrix S(dmin, dmin);
  Hatrix::Matrix V(dmin, n);
  double tolerance = Hatrix::truncated_svd(A, U, S, V, rank);

  Hatrix::Matrix UxS(m, rank);
  Hatrix::matmul(U, S, UxS, false, false, 1, 0);
  Hatrix::matmul(UxS, V, A, false, false, 1, 0);
  check_frobenius_norm(A_check, A, tolerance);
}


TEST_P(SVDTests, SVD){
  int64_t m, n;
  std::tie(m, n) = GetParam();

  Hatrix::Matrix A(m, n);
  A = 5.5;

  int64_t s_dim = A.min_dim();
  Hatrix::Matrix A_copy(A);
  Hatrix::Matrix U(m, s_dim), S(s_dim, s_dim), V(s_dim, n), A_rebuilt(m, n);
  Hatrix::svd(A, U, S, V);
  Hatrix::Matrix temp(m, s_dim);
  Hatrix::matmul(U, S, temp, false, false, 1, 0);
  Hatrix::matmul(temp, V, A_rebuilt, false, false, 1, 0);

    // Check result
  for (int64_t i=0; i<A.rows; ++i) {
    for (int64_t j=0; j<A.cols; ++j) {
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

INSTANTIATE_TEST_SUITE_P(
  LAPACK, truncSVDTests,
  testing::Values(
    std::make_tuple(50, 50, 7),
    std::make_tuple(100, 80, 10),
    std::make_tuple(90, 120, 14),
    std::make_tuple(100, 100, 5)
   ),
  [](const testing::TestParamInfo<truncSVDTests::ParamType>& info) {
    std::string name = (
      "m" + std::to_string(std::get<0>(info.param))
      + "n" + std::to_string(std::get<1>(info.param))
      + "k" + std::to_string(std::get<2>(info.param))
    );
    return name;
  }
);