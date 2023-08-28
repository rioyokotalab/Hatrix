#include <cstdint>
#include <string>
#include <tuple>

#include "Hatrix/Hatrix.hpp"
#include "gtest/gtest.h"

class LUTests : public testing::TestWithParam<std::tuple<int64_t, int64_t>> {};

TEST_P(LUTests, lu) {
  Hatrix::Context::init();
  int64_t m, n;
  std::tie(m, n) = GetParam();

  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, n);

  // Set a large value on the diagonal to avoid pivoting
  int64_t d = m * n;
  int64_t n_diag = A.min_dim();
  for (int64_t i = 0; i < n_diag; ++i) {
    A(i, i) += d--;
  }

  Hatrix::Matrix A_copy(A);
  Hatrix::Matrix L(m, n_diag), U(n_diag, n), A_rebuilt(m, n);
  Hatrix::lu(A, L, U);
  Hatrix::matmul(L, U, A_rebuilt, false, false, 1, 0);
  Hatrix::Context::join();

  // Check result
  for (int64_t i = 0; i < A.rows; ++i) {
    for (int64_t j = 0; j < A.cols; ++j) {
      EXPECT_FLOAT_EQ(A_rebuilt(i, j), A_copy(i, j));
    }
  }

  Hatrix::Context::finalize();
}

INSTANTIATE_TEST_SUITE_P(
    LAPACK, LUTests,
    testing::Combine(testing::Values(8, 16, 32), testing::Values(8, 16, 32)),
    [](const testing::TestParamInfo<LUTests::ParamType>& info) {
      std::string name = ("m" + std::to_string(std::get<0>(info.param)) + "n" +
                          std::to_string(std::get<1>(info.param)));
      return name;
    });
