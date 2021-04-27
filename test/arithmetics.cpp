#include "Hatrix/Hatrix.h"

#include "gtest/gtest.h"

#include <cassert>
#include <tuple>

class ArithmeticTests : public testing::TestWithParam<std::tuple<int64_t, int64_t>>{};

TEST_P(ArithmeticTests, Addition) {
  int64_t m, n;
  std::tie(m, n) = GetParam();

  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix B = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix C = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix C_copy(C);
  Hatrix::Matrix D(m, n);
  Hatrix::Matrix E = A + B;
  
  Hatrix::matadd(A, B, C);
  Hatrix::matadd(A, B, D);

  for (int64_t i=0; i<A.rows; ++i){
    for (int64_t j=0; j<A.cols; ++j){
      EXPECT_FLOAT_EQ(D(i,j), A(i,j)+B(i,j));
      EXPECT_FLOAT_EQ(D(i,j), E(i,j));
      EXPECT_FLOAT_EQ(C(i,j), A(i,j)+B(i,j)+C_copy(i,j));
    }
  }
}


TEST_P(ArithmeticTests, Subtraction) {
  int64_t m, n;
  std::tie(m, n) = GetParam();

  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix B = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix C = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix C_copy(C);
  Hatrix::Matrix D(m, n);
  Hatrix::Matrix E = A - B;
  
  Hatrix::matsub(A, B, C);
  Hatrix::matsub(A, B, D);


  for (int64_t i=0; i<A.rows; ++i){
    for (int64_t j=0; j<A.cols; ++j){
      EXPECT_FLOAT_EQ(D(i,j), A(i,j)-B(i,j));
      EXPECT_FLOAT_EQ(D(i,j), E(i,j));
      EXPECT_FLOAT_EQ(C(i,j), A(i,j)-B(i,j)+C_copy(i,j));
    }
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
