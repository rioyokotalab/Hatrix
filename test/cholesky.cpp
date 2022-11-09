#include <cstdint>
#include <string>
#include <tuple>

#include "Hatrix/Hatrix.h"
#include "gtest/gtest.h"

class CholeskyTests : public testing::TestWithParam<std::tuple<int64_t>> {};

TEST_P(CholeskyTests, lu) {
  Hatrix::Context::init();
  int64_t m;
  std::tie(m) = GetParam();

  Hatrix::Matrix A = Hatrix::generate_random_spd_matrix(m, 10.0);
  // keep only the lower triangle of this matrix.
  for (int i = 0; i < m; ++i) {
    for (int j = i+1; j < m; ++j) {
      A(i, j) = 0;
    }
  }

  Hatrix::Matrix A_copy(A, true), A_actual(m, m);
  Hatrix::cholesky(A_copy, Hatrix::Lower);
  Hatrix::matmul(A_copy, A_copy, A_actual, false, true, 1, 0);

  for (int i = 0; i < m; ++i) {
    for (int j = i+1; j < m; ++j) {
      A_actual(i, j) = 0;
    }
  }

  // Check result
  for (int64_t i = 0; i < A.rows; ++i) {
    for (int64_t j = 0; j <= i; ++j) {
      EXPECT_FLOAT_EQ(A_actual(i, j), A(i, j));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    LAPACK, CholeskyTests,
    testing::Combine(testing::Values(8, 16, 32)),
    [](const testing::TestParamInfo<CholeskyTests::ParamType>& info) {
      std::string name = ("m" + std::to_string(std::get<0>(info.param)));
      return name;
    });
