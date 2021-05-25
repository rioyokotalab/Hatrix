#include <cstdint>
#include <string>
#include <tuple>

#include "Hatrix/Hatrix.h"
#include "gtest/gtest.h"

class ScaleTests
    : public testing::TestWithParam<std::tuple<int64_t, int64_t, double>> {};

TEST_P(ScaleTests, Scaling) {
  int64_t m, n;
  double alpha;
  std::tie(m, n, alpha) = GetParam();
  Hatrix::init(1);
  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix A_copy(A);

  Hatrix::scale(A, alpha);
  for (int64_t j = 0; j < A.cols; ++j) {
    for (int64_t i = 0; i < A.rows; ++i) {
      EXPECT_EQ(A(i, j), A_copy(i, j) * alpha);
    }
  }
  Hatrix::term();
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
