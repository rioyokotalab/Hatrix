#include <cstdint>
#include <string>
#include <tuple>

#include "Hatrix/Hatrix.h"
#include "gtest/gtest.h"

class InterpolateTests : public testing::TestWithParam<std::tuple<int64_t, int64_t, int64_t>> {};

using namespace Hatrix;

TEST_P(InterpolateTests, interpolate_rank) {
  Hatrix::Context::init();
  int64_t m, n, rank;
  std::tie(m, n, rank) = GetParam();

  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix U, S, V; double error;
  std::tie(U, S, V, error) = Hatrix::truncated_svd(A, rank);

  Matrix Arank = matmul(matmul(U, S), V);

  Hatrix::Context::finalize();
}

INSTANTIATE_TEST_SUITE_P(
    INTERPOLATE, InterpolateTests,
    testing::Combine(testing::Values(50), testing::Values(50), testing::Values(10)),
    [](const testing::TestParamInfo<InterpolateTests::ParamType>& info) {
      std::string name = ("m" + std::to_string(std::get<0>(info.param)) +
                          "n" + std::to_string(std::get<1>(info.param)) +
                          "rank" + std::to_string(std::get<2>(info.param)));
      return name;
    });
