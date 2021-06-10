#include <cstdint>
#include <iomanip>
#include <sstream>
#include <tuple>

#include "Hatrix/Hatrix.h"
#include "gtest/gtest.h"

class MatMulTests
    : public testing::TestWithParam<
          std::tuple<int64_t, int64_t, int64_t, bool, bool, double, double> > {
};

TEST_P(MatMulTests, matmul) {
  Hatrix::Context::init();
  int64_t M, N, K;
  bool transA, transB;
  double alpha, beta;
  std::tie(M, K, N, transA, transB, alpha, beta) = GetParam();
  Hatrix::Matrix A =
      Hatrix::generate_random_matrix(transA ? K : M, transA ? M : K);
  Hatrix::Matrix B =
      Hatrix::generate_random_matrix(transB ? N : K, transB ? K : N);
  Hatrix::Matrix C = Hatrix::generate_random_matrix(M, N);
  Hatrix::Matrix C_check(C);
  Hatrix::matmul(A, B, C, transA, transB, alpha, beta);
  Hatrix::Context::join();

  // Manual matmul
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      C_check(i, j) =
          (beta * C_check(i, j) +
           alpha * (transA ? A(0, i) : A(i, 0)) * (transB ? B(j, 0) : B(0, j)));
      for (int64_t k = 1; k < K; ++k) {
        C_check(i, j) += (alpha * (transA ? A(k, i) : A(i, k)) *
                          (transB ? B(j, k) : B(k, j)));
      }
    }
  }

  // Check result
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      EXPECT_NEAR(C_check(i, j), C(i, j), 10e-14);
    }
  }
  Hatrix::Context::finalize();
}

TEST_P(MatMulTests, matmulReturn) {
  Hatrix::Context::init();
  int64_t M, N, K;
  bool transA, transB;
  double alpha, _;
  std::tie(M, K, N, transA, transB, alpha, _) = GetParam();
  Hatrix::Matrix A =
      Hatrix::generate_random_matrix(transA ? K : M, transA ? M : K);
  Hatrix::Matrix B =
      Hatrix::generate_random_matrix(transB ? N : K, transB ? K : N);
  Hatrix::Matrix C = Hatrix::matmul(A, B, transA, transB, alpha);
  Hatrix::Context::join();

  // Manual matmul
  Hatrix::Matrix C_check(C.rows, C.cols);
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      for (int64_t k = 0; k < K; ++k) {
        C_check(i, j) += (alpha * (transA ? A(k, i) : A(i, k)) *
                          (transB ? B(j, k) : B(k, j)));
      }
    }
  }

  // Check result
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      EXPECT_NEAR(C_check(i, j), C(i, j), 10e-14);
    }
  }
  Hatrix::Context::finalize();
}

INSTANTIATE_TEST_SUITE_P(
    Params, MatMulTests,
    testing::Combine(testing::Values(13), testing::Values(47),
                     testing::Values(55), testing::Bool(), testing::Bool(),
                     testing::Range(-1.0, 1.5, 0.5),
                     testing::Range(-1.0, 1.5, 0.5)),
    [](const testing::TestParamInfo<MatMulTests::ParamType>& info) {
      std::stringstream alpha_;
      std::stringstream beta_;
      alpha_ << std::fixed << std::setprecision(1) << std::get<5>(info.param);
      beta_ << std::fixed << std::setprecision(1) << std::get<6>(info.param);
      std::string alpha = alpha_.str();
      alpha.replace(alpha.find_last_of("."), 1, "l");
      if (alpha.find_last_of("-") < alpha.length())
        alpha.replace(alpha.find_last_of("-"), 1, "m");
      std::string beta = beta_.str();
      beta.replace(beta.find_last_of("."), 1, "l");
      if (beta.find_last_of("-") < beta.length())
        beta.replace(beta.find_last_of("-"), 1, "m");
      std::string name =
          ("TA" + std::to_string(std::get<3>(info.param)) + "TB" +
           std::to_string(std::get<4>(info.param)) + "A" + alpha + "B" + beta);
      return name;
    });

INSTANTIATE_TEST_SUITE_P(
    Sizes, MatMulTests,
    testing::Combine(testing::Values(16, 32, 64), testing::Values(16, 32, 64),
                     testing::Values(16, 32, 64), testing::Values(false),
                     testing::Values(false), testing::Values(1),
                     testing::Values(1)),
    [](const testing::TestParamInfo<MatMulTests::ParamType>& info) {
      std::string name = ("M" + std::to_string(std::get<0>(info.param)) + "K" +
                          std::to_string(std::get<1>(info.param)) + "N" +
                          std::to_string(std::get<2>(info.param)));
      return name;
    });
