#include "Hatrix/Hatrix.h"

#include "gtest/gtest.h"

#include <cassert>
#include <cstdint>
using std::int64_t;
#include <string>
#include <tuple>


class ArithmeticTests : public testing::TestWithParam<std::tuple<int64_t, int64_t>>{};
class MatMulOperatorTests : public testing::TestWithParam<
  std::tuple<int64_t, int64_t, int64_t>
> {};
class ScalarMulOperatorTests : public testing::TestWithParam<
  std::tuple<int64_t, int64_t, double>
> {};

TEST_P(ArithmeticTests, PlusOperator) {
  int64_t m, n;
  std::tie(m, n) = GetParam();

  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix B = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix C = A + B;

  for (int64_t i=0; i<A.rows; ++i) for (int64_t j=0; j<A.cols; ++j) {
    EXPECT_EQ(C(i, j), A(i, j) + B(i, j));
  }
}

TEST_P(ArithmeticTests, PlusEqualsOperator) {
  int64_t m, n;
  std::tie(m, n) = GetParam();

  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix A_check(A);
  Hatrix::Matrix B = Hatrix::generate_random_matrix(m, n);
  A += B;

  for (int64_t i=0; i<A.rows; ++i) for (int64_t j=0; j<A.cols; ++j) {
    EXPECT_EQ(A_check(i, j) + B(i, j), A(i, j));
  }
}

TEST_P(ArithmeticTests, MinusOperator) {
  int64_t m, n;
  std::tie(m, n) = GetParam();

  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix B = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix C = A - B;

  for (int64_t i=0; i<A.rows; ++i) for (int64_t j=0; j<A.cols; ++j) {
    EXPECT_EQ(C(i, j), A(i, j) - B(i, j));
  }
}

TEST_P(ArithmeticTests, MinusEqualsOperator) {
  int64_t m, n;
  std::tie(m, n) = GetParam();

  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix A_check(A);
  Hatrix::Matrix B = Hatrix::generate_random_matrix(m, n);
  A -= B;

  for (int64_t i=0; i<A.rows; ++i) for (int64_t j=0; j<A.cols; ++j) {
    EXPECT_EQ(A_check(i, j) - B(i, j), A(i, j));
  }
}

TEST_P(ArithmeticTests, abs) {
  int64_t m, n;
  std::tie(m, n) = GetParam();

  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix A_check = abs(A);

  for (int64_t i=0; i<A.rows; ++i) for (int64_t j=0; j<A.cols; ++j) {
    EXPECT_EQ(A_check(i, j), A(i, j) < 0 ? -A(i, j) : A(i, j));
  }
}

INSTANTIATE_TEST_SUITE_P(
  LAPACK, ArithmeticTests,
  testing::Values(
    std::make_tuple(50, 50),
    std::make_tuple(23, 75),
    std::make_tuple(100, 66)
   ),
  [](const testing::TestParamInfo<ArithmeticTests::ParamType>& info) {
    std::string name = (
      "m" + std::to_string(std::get<0>(info.param))
      + "n" + std::to_string(std::get<1>(info.param))
    );
    return name;
  }
);

TEST_P(MatMulOperatorTests, MultiplicationOperator) {
  int64_t M, N, K;
  std::tie(M, K, N) = GetParam();
  Hatrix::Matrix A = Hatrix::generate_random_matrix(M, K);
  Hatrix::Matrix B = Hatrix::generate_random_matrix(K, N);
  Hatrix::Matrix C(M, N);
  Hatrix::Matrix C_check = A * B;
  Hatrix::matmul(A, B, C, false, false, 1, 0);

  // Check result
  for (int64_t i=0; i<M; ++i) {
    for (int64_t j=0; j<N; ++j) {
      EXPECT_FLOAT_EQ(C_check(i, j), C(i, j));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
  Operator, MatMulOperatorTests,
  testing::Combine(
    testing::Values(16, 32, 64),
    testing::Values(16, 32, 64),
    testing::Values(16, 32, 64)
  ),
  [](const testing::TestParamInfo<MatMulOperatorTests::ParamType>& info) {
    std::string name = (
      "M" + std::to_string(std::get<0>(info.param))
      + "K" + std::to_string(std::get<1>(info.param))
      + "N" + std::to_string(std::get<2>(info.param))
    );
    return name;
  }
);

TEST_P(ScalarMulOperatorTests, ScalarMultiplicationOperator) {
  int64_t M, N;
  double alpha;
  std::tie(M, N, alpha) = GetParam();
  Hatrix::Matrix A = Hatrix::generate_random_matrix(M, N);
  Hatrix::Matrix B = A * alpha;
  Hatrix::Matrix C = alpha * A;
  Hatrix::scale(A, alpha);

  // Check result
  for (int64_t i=0; i<M; ++i) {
    for (int64_t j=0; j<N; ++j) {
      EXPECT_EQ(A(i, j), C(i, j));
      EXPECT_EQ(A(i, j), B(i, j));
    }
  }
}

TEST_P(ScalarMulOperatorTests, ScalarMultiplicationEqualsOperator) {
  int64_t M, N;
  double alpha;
  std::tie(M, N, alpha) = GetParam();
  Hatrix::Matrix A = Hatrix::generate_random_matrix(M, N);
  Hatrix::Matrix A_copy(A);
  A *= alpha;

  // Check result
  for (int64_t i=0; i<M; ++i) {
    for (int64_t j=0; j<N; ++j) {
      EXPECT_EQ(A(i, j), A_copy(i, j) * alpha);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
  Operator, ScalarMulOperatorTests,
  testing::Values(
    std::make_tuple(5, 5, 7.9834),
    std::make_tuple(11, 21, -4),
    std::make_tuple(18, 5, 1/8)
   ),
  [](const testing::TestParamInfo<ScalarMulOperatorTests::ParamType>& info) {
    std::string name = (
      "M" + std::to_string(std::get<0>(info.param))
      + "N" + std::to_string(std::get<1>(info.param))
    );
    return name;
  }
);
