#include "Hatrix/Hatrix.h"

#include "gtest/gtest.h"

#include <cstdint>
using std::int64_t;
#include <string>
#include <tuple>
#include <cmath>


class NormTests : public testing::TestWithParam<std::tuple<int64_t, int64_t>>{};

TEST_P(NormTests, OneNorm){
  int64_t m, n;
  std::tie(m, n) = GetParam();

  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, n);

  double norm_check = A(0, 0)
  for (int64_t i=0; i<A.cols; ++i){
    int sum = 0;
    for (int64_t j=0; j<A.cols; ++j){
      sum += abs(A(i, j));
    }
    if (sum > norm_check)
      norm_check = sum;
  }
  
  double norm = Hatrix::norm(A, '1');
  EXPECT_FLOAT_EQ(norm, norm_check);
  norm = Hatrix::norm(A, 'O');
  EXPECT_FLOAT_EQ(norm, norm_check);
  norm = Hatrix::norm(A, 'o');
  EXPECT_FLOAT_EQ(norm, norm_check);
}

TEST_P(LUTests, lup){
  int64_t m, n;
  std::tie(m, n) = GetParam();

  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, n);

  //set a large value on the diagonal to avoid pivoting
  int64_t d = m * n;
  int64_t n_diag = A.min_dim();
  for (int64_t i=0; i<n_diag; ++i){
    A(i,i) += d--;
  }

  Hatrix::Matrix A_copy(A);
  Hatrix::Matrix L(m, n_diag), U(n_diag, n), P(m, m), A_rebuilt(m, n);
  Hatrix::lup(A, L, U, P);
  Hatrix::matmul(L, U, A, false, false, 1, 0);
  Hatrix::matmul(P, A, A_rebuilt, false, false, 1, 0);

    // Check result
  for (int64_t i=0; i<A.rows; ++i) {
    for (int64_t j=0; j<A.cols; ++j) {
      EXPECT_FLOAT_EQ(A_rebuilt(i, j), A_copy(i, j));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
  LAPACK, LUTests,
  testing::Combine(
    testing::Values(8, 16, 32),
    testing::Values(8, 16, 32)
  ),
  [](const testing::TestParamInfo<LUTests::ParamType>& info) {
    std::string name = (
      "m" + std::to_string(std::get<0>(info.param))
      + "n" + std::to_string(std::get<1>(info.param))
    );
    return name;
  }
);