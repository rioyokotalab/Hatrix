#include <cstdint>
#include <string>
#include <tuple>

#include "Hatrix/Hatrix.hpp"
#include "gtest/gtest.h"

class InterpolateTests : public testing::TestWithParam<std::tuple<int64_t, int64_t>> {};
class TruncatedIDTests : public testing::TestWithParam<std::tuple<int64_t, int64_t, int64_t>> {};
class ErrorIDTests : public testing::TestWithParam<std::tuple<int64_t, int64_t, double, bool>> {};

using namespace Hatrix;

TEST_P(InterpolateTests, interpolate_rank) {
  int64_t m, n, rank = 10;
  std::tie(m, n) = GetParam();

  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix U, S, V; double error;
  std::tie(U, S, V, error) = Hatrix::truncated_svd(A, rank);

  Matrix Arank = matmul(matmul(U, S), V);
  Matrix Ainterp, Apivots, Arank_copy(Arank);
  std::tie(Ainterp, Apivots) = truncated_interpolate(Arank, rank);

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
}

TEST_P(InterpolateTests, interpolate_error) {
  int64_t m, n;
  std::tie(m, n) = GetParam();

  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix U, S, V; double error;
  std::tie(U, S, V, error) = Hatrix::truncated_svd(A, m / 2);

  Matrix A_error = matmul(matmul(U, S), V);
  double tol = 1e-9;

  Matrix A_interp, A_error_copy(A_error);
  std::vector<int64_t> A_pivots;
  int64_t rank;
  std::tie(A_interp, A_pivots, rank) = error_interpolate(A_error, tol);

    // Generate A_CS
  Matrix A_CS(m, rank);
  for (int j = 0; j < rank; ++j) {
    int pivot_col = A_pivots[j];
    for (int i = 0; i < m; ++i) {
      A_CS(i, j) = A_error_copy(i, pivot_col);
    }
  }

  Matrix A_error_pivoted = matmul(A_CS, A_interp, false, true);
  Matrix result(m, n);

  // Bring the original matrix into the pivoted form
  for (int j = 0; j < result.cols; ++j) {
    int pcol = A_pivots[j];
    for (int i = 0; i < result.rows; ++i) {
      result(i, j) = A_error_copy(i, pcol);
    }
  }

  EXPECT_NEAR(Hatrix::norm(result - A_error_pivoted) / Hatrix::norm(result), 0, 1e-12);
}

TEST_P(TruncatedIDTests, truncated_id_row) {
  int64_t m, n, rank;
  std::tie(m, n, rank) = GetParam();

  // Generate rank-k matrix
  const Hatrix::Matrix L = Hatrix::generate_random_matrix(m, rank);
  const Hatrix::Matrix R = Hatrix::generate_random_matrix(rank, n);
  const Hatrix::Matrix D = Hatrix::matmul(L, R);
  Hatrix::Matrix A(D);
  Hatrix::Matrix U;
  std::vector<int64_t> skel_rows;
  std::tie(U, skel_rows) = Hatrix::truncated_id_row(A, rank);

  Matrix A_skel_rows(rank, A.cols);
  for (int64_t i = 0; i < rank; i++) {
    const auto row = skel_rows[i];
    for (int64_t j = 0; j < A.cols; j++) {
      A_skel_rows(i, j) = D(row, j);
    }
  }

  // Check dimensions
  EXPECT_EQ(U.rows, A.rows);
  EXPECT_EQ(U.cols, A_skel_rows.rows);
  EXPECT_EQ(A_skel_rows.cols, D.cols);

  // Check compression error
  const double eps = 1e-13;
  const double error = Hatrix::norm(D - Hatrix::matmul(U, A_skel_rows));
  EXPECT_NEAR(error, 0, eps);
}

TEST_P(ErrorIDTests, error_id_row) {
  int64_t m, n;
  double eps;
  bool relative;
  std::tie(m, n, eps, relative) = GetParam();

  const Hatrix::Matrix D = Hatrix::generate_low_rank_matrix(m, n);
  Hatrix::Matrix A(D);
  Hatrix::Matrix U;
  std::vector<int64_t> skel_rows;
  std::tie(U, skel_rows) = Hatrix::error_id_row(A, eps * 1e-1, relative);

  const int64_t rank = U.cols;
  Matrix A_skel_rows(rank, A.cols);
  for (int64_t i = 0; i < rank; i++) {
    const auto row = skel_rows[i];
    for (int64_t j = 0; j < A.cols; j++) {
      A_skel_rows(i, j) = D(row, j);
    }
  }

  // Check dimensions
  EXPECT_EQ(U.rows, A.rows);
  EXPECT_EQ(U.cols, A_skel_rows.rows);
  EXPECT_EQ(A_skel_rows.cols, D.cols);

  // Check compression error
  const double dnorm = Hatrix::norm(D);
  const double diff = Hatrix::norm(D - Hatrix::matmul(U, A_skel_rows));
  const double error = relative ? diff / dnorm : diff;
  EXPECT_NEAR(error, 0, eps);
}

INSTANTIATE_TEST_SUITE_P(
    INTERPOLATE, InterpolateTests,
    testing::Combine(testing::Values(30), testing::Values(30)),
    [](const testing::TestParamInfo<InterpolateTests::ParamType>& info) {
      std::string name = ("m" + std::to_string(std::get<0>(info.param)) +
                          "n" + std::to_string(std::get<1>(info.param)));
      return name;
    });

INSTANTIATE_TEST_SUITE_P(INTERPOLATE, TruncatedIDTests,
                         testing::Values(std::make_tuple(32, 32, 4),
                                         std::make_tuple(32, 24, 4),
                                         std::make_tuple(24, 32, 4),
                                         std::make_tuple(32, 32, 16),
                                         std::make_tuple(32, 24, 16),
                                         std::make_tuple(24, 32, 16),
                                         std::make_tuple(32, 32, 32),
                                         std::make_tuple(32, 24, 24),
                                         std::make_tuple(24, 32, 24)));

INSTANTIATE_TEST_SUITE_P(INTERPOLATE, ErrorIDTests,
                         testing::Values(std::make_tuple(32, 32, 1e-6, true),
                                         std::make_tuple(32, 24, 1e-6, true),
                                         std::make_tuple(24, 32, 1e-6, true),
                                         std::make_tuple(32, 32, 1e-8, true),
                                         std::make_tuple(32, 24, 1e-8, true),
                                         std::make_tuple(24, 32, 1e-8, true),
                                         std::make_tuple(32, 32, 1e-10, true),
                                         std::make_tuple(32, 24, 1e-10, true),
                                         std::make_tuple(24, 32, 1e-10, true),
                                         std::make_tuple(32, 32, 1e-6, false),
                                         std::make_tuple(32, 24, 1e-6, false),
                                         std::make_tuple(24, 32, 1e-6, false),
                                         std::make_tuple(32, 32, 1e-8, false),
                                         std::make_tuple(32, 24, 1e-8, false),
                                         std::make_tuple(24, 32, 1e-8, false),
                                         std::make_tuple(32, 32, 1e-10, false),
                                         std::make_tuple(32, 24, 1e-10, false),
                                         std::make_tuple(24, 32, 1e-10, false)));

