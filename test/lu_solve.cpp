#include <cstdint>
#include <string>
#include <tuple>
#include <random>

#include "Hatrix/Hatrix.hpp"
#include "gtest/gtest.h"

class LU_SolveTests : public testing::TestWithParam<std::tuple<int64_t>> {};

TEST_P(LU_SolveTests, lu_solve) {
  int64_t m;
  std::tie(m) = GetParam();

  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, m);

  std::mt19937 gen(1);
  std::uniform_real_distribution<double> dist(100, 10000);
  Hatrix::Matrix b(m, 1);
  for (int i = 0; i < m; ++i) { b(i, 0) = dist(gen); }

  Hatrix::Matrix x = lu_solve(A, b);
  Hatrix::Matrix product = matmul(A, x);

  double rel_error = norm(product - b) / norm(b);

  EXPECT_NEAR(rel_error, 0, 1e-13);
}

INSTANTIATE_TEST_SUITE_P(
    LAPACK, LU_SolveTests,
    testing::Combine(testing::Values(64, 128, 256)),
    [](const testing::TestParamInfo<LU_SolveTests::ParamType>& info) {
      std::string name = ("m" + std::to_string(std::get<0>(info.param)));
      return name;
    });
