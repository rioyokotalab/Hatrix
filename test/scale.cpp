#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "Hatrix/Hatrix.h"
#include "gtest/gtest.h"

class ScaleTests
    : public testing::TestWithParam<std::tuple<int64_t, int64_t, double>> {};

TEST_P(ScaleTests, Scaling) {
  int64_t m, n;
  double alpha;
  std::tie(m, n, alpha) = GetParam();
  Hatrix::Context::init();
  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix A_copy(A);

  Hatrix::scale(A, alpha);
  for (int64_t j = 0; j < A.cols; ++j) {
    for (int64_t i = 0; i < A.rows; ++i) {
      EXPECT_EQ(A(i, j), A_copy(i, j) * alpha);
    }
  }
  Hatrix::Context::finalize();
}

TEST_P(ScaleTests, ScalingPart) {
  int64_t m, n;
  double alpha;
  std::tie(m, n, alpha) = GetParam();
  Hatrix::Context::init();
  Hatrix::Matrix A_big = Hatrix::generate_random_matrix(2*m, 2*n);
  std::vector<Hatrix::Matrix> A_split = A_big.split(2, 2);
  Hatrix::Matrix A_copy(A_split[0]);

  Hatrix::scale(A_split[0], alpha);
  for (int64_t j = 0; j < A_split[0].cols; ++j) {
    for (int64_t i = 0; i < A_split[0].rows; ++i) {
      EXPECT_EQ(A_big(i, j), A_copy(i, j) * alpha);
    }
  }
  Hatrix::Context::finalize();
}

INSTANTIATE_TEST_SUITE_P(
    BLAS, ScaleTests,
    testing::Values(std::make_tuple(10, 10, 4.32), std::make_tuple(1, 7, 2),
                    std::make_tuple(15, 3, 99.9), std::make_tuple(4, 1, 0.5),
                    std::make_tuple(8, 21, -3.4)),
    [](const testing::TestParamInfo<ScaleTests::ParamType>& info) {
      std::string name = ("m" + std::to_string(std::get<0>(info.param)) + "n" +
                          std::to_string(std::get<1>(info.param)));
      return name;
    });
