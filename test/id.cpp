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
  Matrix Ainterp, Apivots, Arank_copy(Arank);
  std::tie(Ainterp, Apivots) = truncated_interpolate(Arank, false, rank);

  // Generate A_CS
  Matrix A_CS(m, rank);
  for (int j = 0; j < rank; ++j) {
    int pivot_col = Apivots(j, 0) - 1;
    for (int i = 0; i < m; ++i) {
      A_CS(i, j) = Arank_copy(i, pivot_col);
    }
  }

  Matrix Arank_pivoted = matmul(A_CS, Ainterp, false, true);
  Matrix result(m, n);

  // Bring the original matrix into the pivoted form
  for (int j = 0; j < result.cols; ++j) {
    int pcol = Apivots(j, 0) - 1;
    for (int i = 0; i < result.rows; ++i) {
      result(i, j) = Arank_copy(i, pcol);
    }
  }

  EXPECT_NEAR(Hatrix::norm(result - Arank_pivoted) / Hatrix::norm(result), 0, 1e-12);

  Hatrix::Context::finalize();
}

TEST_P(InterpolateTests, interpolate_rank_transpose) {
  Hatrix::Context::init();
  int64_t m, n, rank;
  std::tie(m, n, rank) = GetParam();

  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix U, S, V; double error;
  std::tie(U, S, V, error) = Hatrix::truncated_svd(A, rank);

  Matrix Arank = matmul(matmul(U, S), V);
  Matrix Ainterp, Apivots, Arank_copy(Arank);
  std::tie(Ainterp, Apivots) = truncated_interpolate(Arank, true, rank);

  // Generate A_CS
  Matrix A_CS(m, rank);
  for (int j = 0; j < rank; ++j) {
    int pivot_col = Apivots(j, 0) - 1;
    for (int i = 0; i < m; ++i) {
      A_CS(i, j) = Arank_copy(i, pivot_col);
    }
  }

  Matrix Arank_pivoted = matmul(A_CS, Ainterp, false, true);
  Matrix result(m, n);

  // Bring the original matrix into the pivoted form
  for (int j = 0; j < result.cols; ++j) {
    int pcol = Apivots(j, 0) - 1;
    for (int i = 0; i < result.rows; ++i) {
      result(i, j) = Arank_copy(i, pcol);
    }
  }

  EXPECT_NEAR(Hatrix::norm(result - transpose(Arank_pivoted)) /
              Hatrix::norm(result), 0, 1e-12);

  Hatrix::Context::finalize();
}

INSTANTIATE_TEST_SUITE_P(
    INTERPOLATE, InterpolateTests,
    testing::Combine(testing::Values(30), testing::Values(30), testing::Values(10)),
    [](const testing::TestParamInfo<InterpolateTests::ParamType>& info) {
      std::string name = ("m" + std::to_string(std::get<0>(info.param)) +
                          "n" + std::to_string(std::get<1>(info.param)) +
                          "rank" + std::to_string(std::get<2>(info.param)));
      return name;
    });
