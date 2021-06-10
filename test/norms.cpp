#include <cmath>
#include <cstdint>
#include <string>
#include <tuple>

#include "Hatrix/Hatrix.h"
#include "gtest/gtest.h"

class NormTests : public testing::TestWithParam<std::tuple<int64_t, int64_t>> {
};

TEST_P(NormTests, norm) {
  int64_t m, n;
  std::tie(m, n) = GetParam();
  Hatrix::Context::init();
  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, n);

  double norm = 0;
  for (int64_t j = 0; j < A.cols; ++j) {
    for (int64_t i = 0; i < A.rows; ++i) {
      norm += A(i, j) * A(i, j);
    }
  }
  norm = std::sqrt(norm);

  EXPECT_FLOAT_EQ(norm, Hatrix::norm(A));
  Hatrix::Context::finalize();
}

INSTANTIATE_TEST_SUITE_P(
    LAPACK, NormTests,
    testing::Values(std::make_tuple(100, 100), std::make_tuple(20, 70),
                    std::make_tuple(99, 55), std::make_tuple(1, 10),
                    std::make_tuple(13, 1)),
    [](const testing::TestParamInfo<NormTests::ParamType>& info) {
      std::string name = ("m" + std::to_string(std::get<0>(info.param)) + "n" +
                          std::to_string(std::get<1>(info.param)));
      return name;
    });
