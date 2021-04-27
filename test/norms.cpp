#include "Hatrix/Hatrix.h"

#include "gtest/gtest.h"

#include <cstdint>
using std::int64_t;
#include <string>
#include <tuple>
#include <cmath>
#include <iostream>


class NormTests : public testing::TestWithParam<std::tuple<int64_t, int64_t>>{};

TEST_P(NormTests, OneNorm){
  int64_t m, n;
  std::tie(m, n) = GetParam();

  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, n);

  double norm = A(0, 0);
  for (int64_t j=0; j<A.cols; ++j){
    double sum = 0;
    for (int64_t i=0; i<A.rows; ++i){
      sum += std::abs(A(i, j));
    }
    if (sum > norm)
      norm = sum;
  }

  EXPECT_FLOAT_EQ(norm, Hatrix::norm(A, Hatrix::OneNorm));
  EXPECT_FLOAT_EQ(norm, Hatrix::one_norm(A));
}

TEST_P(NormTests, MaxNorm){
  int64_t m, n;
  std::tie(m, n) = GetParam();

  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, n);

  double norm = A(0, 0);
  double absVal;
  for (int64_t j=0; j<A.cols; ++j){
    for (int64_t i=0; i<A.rows; ++i){
      absVal = std::abs(A(i, j));
      if (absVal > norm)
        norm = absVal;
    }
  }
  
  EXPECT_FLOAT_EQ(norm, Hatrix::norm(A, Hatrix::MaxNorm));
  EXPECT_FLOAT_EQ(norm, Hatrix::max_norm(A));
}

TEST_P(NormTests, InfNorm){
  int64_t m, n;
  std::tie(m, n) = GetParam();

  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, n);

  double norm = A(0, 0);
  for (int64_t i=0; i<A.rows; ++i){
    double sum = 0;
    for (int64_t j=0; j<A.cols; ++j){
      sum += std::abs(A(i, j));
    }
    if (sum > norm)
      norm = sum;
  }
  
  EXPECT_FLOAT_EQ(norm, Hatrix::norm(A, Hatrix::InfinityNorm));
  EXPECT_FLOAT_EQ(norm, Hatrix::infinity_norm(A));
}

TEST_P(NormTests, FrobNorm){
  int64_t m, n;
  std::tie(m, n) = GetParam();

  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, n);

  double norm = 0;
  for (int64_t j=0; j<A.cols; ++j){
    for (int64_t i=0; i<A.rows; ++i){
      norm += A(i, j) * A(i, j);
    }
  }
  norm = std::sqrt(norm);

  EXPECT_FLOAT_EQ(norm, Hatrix::norm(A, Hatrix::FrobeniusNorm));
  EXPECT_FLOAT_EQ(norm, Hatrix::frobenius_norm(A));
}

INSTANTIATE_TEST_SUITE_P(
  LAPACK, NormTests,
 testing::Values(
    std::make_tuple(100, 100),
    std::make_tuple(20, 70),
    std::make_tuple(99, 55),
    std::make_tuple(1, 10),
    std::make_tuple(13, 1)
   ),
  [](const testing::TestParamInfo<NormTests::ParamType>& info) {
    std::string name = (
      "m" + std::to_string(std::get<0>(info.param))
      + "n" + std::to_string(std::get<1>(info.param))
    );
    return name;
  }
);