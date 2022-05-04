#include <cassert>
#include <cstdint>
#include <string>
#include <tuple>

#include "Hatrix/Hatrix.h"
#include "gtest/gtest.h"

class ArithmeticTests
    : public testing::TestWithParam<std::tuple<int64_t, int64_t>> {};
class MatMulOperatorTests
    : public testing::TestWithParam<std::tuple<int64_t, int64_t, int64_t>> {};
class ScalarMulOperatorTests
    : public testing::TestWithParam<std::tuple<int64_t, int64_t, double>> {};

TEST_P(ArithmeticTests, PlusOperator) {
  int64_t m, n;
  std::tie(m, n) = GetParam();

  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix B = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix C = A + B;

  for (int64_t i = 0; i < A.rows; ++i)
    for (int64_t j = 0; j < A.cols; ++j) {
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

  for (int64_t i = 0; i < A.rows; ++i)
    for (int64_t j = 0; j < A.cols; ++j) {
      EXPECT_EQ(A_check(i, j) + B(i, j), A(i, j));
    }
}

TEST_P(ArithmeticTests, MinusOperator) {
  int64_t m, n;
  std::tie(m, n) = GetParam();

  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix B = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix C = A - B;

  for (int64_t i = 0; i < A.rows; ++i)
    for (int64_t j = 0; j < A.cols; ++j) {
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

  for (int64_t i = 0; i < A.rows; ++i)
    for (int64_t j = 0; j < A.cols; ++j) {
      EXPECT_EQ(A_check(i, j) - B(i, j), A(i, j));
    }
}

TEST_P(ArithmeticTests, abs) {
  int64_t m, n;
  std::tie(m, n) = GetParam();

  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix A_check = abs(A);

  for (int64_t i = 0; i < A.rows; ++i)
    for (int64_t j = 0; j < A.cols; ++j) {
      EXPECT_EQ(A_check(i, j), A(i, j) < 0 ? -A(i, j) : A(i, j));
    }
}

INSTANTIATE_TEST_SUITE_P(
    LAPACK, ArithmeticTests,
    testing::Values(std::make_tuple(50, 50), std::make_tuple(23, 75),
                    std::make_tuple(100, 66)),
    [](const testing::TestParamInfo<ArithmeticTests::ParamType>& info) {
      std::string name = ("m" + std::to_string(std::get<0>(info.param)) + "n" +
                          std::to_string(std::get<1>(info.param)));
      return name;
    });

TEST_P(MatMulOperatorTests, MultiplicationOperator) {
  int64_t M, N, K;
  Hatrix::Context::init();
  std::tie(M, K, N) = GetParam();
  Hatrix::Matrix A = Hatrix::generate_random_matrix(M, K);
  Hatrix::Matrix B = Hatrix::generate_random_matrix(K, N);
  Hatrix::Matrix C(M, N);
  Hatrix::Matrix C_check = A * B;
  Hatrix::matmul(A, B, C, false, false, 1, 0);
  Hatrix::Context::join();

  // Check result
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      EXPECT_FLOAT_EQ(C_check(i, j), C(i, j));
    }
  }
  Hatrix::Context::finalize();
}

INSTANTIATE_TEST_SUITE_P(
    Operator, MatMulOperatorTests,
    testing::Combine(testing::Values(16, 32, 64), testing::Values(16, 32, 64),
                     testing::Values(16, 32, 64)),
    [](const testing::TestParamInfo<MatMulOperatorTests::ParamType>& info) {
      std::string name = ("M" + std::to_string(std::get<0>(info.param)) + "K" +
                          std::to_string(std::get<1>(info.param)) + "N" +
                          std::to_string(std::get<2>(info.param)));
      return name;
    });

TEST_P(ScalarMulOperatorTests, ScalarMultiplicationOperator) {
  int64_t M, N;
  double alpha;
  Hatrix::Context::init();
  std::tie(M, N, alpha) = GetParam();
  Hatrix::Matrix A = Hatrix::generate_random_matrix(M, N);
  Hatrix::Matrix B = A * alpha;
  Hatrix::Matrix C = alpha * A;
  Hatrix::scale(A, alpha);

  // Check result
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      EXPECT_EQ(A(i, j), C(i, j));
      EXPECT_EQ(A(i, j), B(i, j));
    }
  }
  Hatrix::Context::finalize();
}

TEST_P(ScalarMulOperatorTests, ScalarMultiplicationEqualsOperator) {
  int64_t M, N;
  double alpha;
  Hatrix::Context::init();
  std::tie(M, N, alpha) = GetParam();
  Hatrix::Matrix A = Hatrix::generate_random_matrix(M, N);
  Hatrix::Matrix A_copy(A);
  A *= alpha;

  // Check result
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      EXPECT_EQ(A(i, j), A_copy(i, j) * alpha);
    }
  }
  Hatrix::Context::finalize();
}

INSTANTIATE_TEST_SUITE_P(
    Operator, ScalarMulOperatorTests,
    testing::Values(std::make_tuple(5, 5, 7.9834), std::make_tuple(11, 21, -4),
                    std::make_tuple(18, 5, 1 / 8)),
    [](const testing::TestParamInfo<ScalarMulOperatorTests::ParamType>& info) {
      std::string name = ("M" + std::to_string(std::get<0>(info.param)) + "N" +
                          std::to_string(std::get<1>(info.param)));
      return name;
    });

TEST_P(ArithmeticTests, Transpose) {
  int64_t m, n;
  Hatrix::Context::init();
  std::tie(m, n) = GetParam();

  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix A_trans = transpose(A);

  EXPECT_EQ(A_trans.rows, n);
  EXPECT_EQ(A_trans.cols, m);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      EXPECT_EQ(A(i, j), A_trans(j, i));
    }
  }
  Hatrix::Context::finalize();
}

TEST_P(ArithmeticTests, LowerTriangularPart) {
  int64_t m = 10, n = 10;
  Hatrix::Context::init();
  // std::tie(m, n) = GetParam();

  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix A_nounit_lower = lower_tri(A);
  Hatrix::Matrix A_unit_lower = lower_tri(A, true);

  EXPECT_EQ(A_nounit_lower.rows, m);
  EXPECT_EQ(A_nounit_lower.cols, n);
  EXPECT_EQ(A_unit_lower.rows, m);
  EXPECT_EQ(A_unit_lower.cols, n);

  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      if(i == j) {
        EXPECT_EQ(A(i, j), A_nounit_lower(i, j));
        EXPECT_EQ(1., A_unit_lower(i, j));
      }
      else if(i > j) {
        EXPECT_EQ(A(i, j), A_nounit_lower(i, j));
        EXPECT_EQ(A(i, j), A_unit_lower(i, j));
      }
      else {
        EXPECT_EQ(0., A_nounit_lower(i, j));
        EXPECT_EQ(0., A_unit_lower(i, j));
      }
    }
  }
  Hatrix::Context::finalize();
}

TEST_P(ArithmeticTests, UpperTriangularPart) {
  int64_t m, n;
  Hatrix::Context::init();
  std::tie(m, n) = GetParam();

  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix A_nounit_upper = upper_tri(A);
  Hatrix::Matrix A_unit_upper = upper_tri(A, true);

  EXPECT_EQ(A_nounit_upper.rows, m);
  EXPECT_EQ(A_nounit_upper.cols, n);
  EXPECT_EQ(A_unit_upper.rows, m);
  EXPECT_EQ(A_unit_upper.cols, n);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      if(i == j) {
	EXPECT_EQ(A(i, j), A_nounit_upper(i, j));
	EXPECT_EQ(1., A_unit_upper(i, j));
      }
      else if(i > j) {
	EXPECT_EQ(0., A_nounit_upper(i, j));
	EXPECT_EQ(0., A_unit_upper(i, j));
      }
      else {
	EXPECT_EQ(A(i, j), A_nounit_upper(i, j));
	EXPECT_EQ(A(i, j), A_unit_upper(i, j));
      }
    }
  }
  Hatrix::Context::finalize();
}
