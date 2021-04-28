#include "Hatrix/Hatrix.h"

#include "gtest/gtest.h"

#include <cstdint>
using std::int64_t;
#include <string>
#include <tuple>
#include <vector>
#include <unistd.h>
#include <stdio.h>
#include <limits.h>


class MatrixGeneratorTests : public testing::TestWithParam<std::tuple<int64_t, int64_t>>{};

TEST(MatrixGeneratorTests, from_csv1_no_params){
  const int64_t N = 5;

  //TODO change to relative path somehow
  Hatrix::Matrix A = Hatrix::generate_from_csv("/home/thomas/Hatrix/test/testdata/readCSV1.csv");
  EXPECT_EQ(A.rows, N);
  EXPECT_EQ(A.cols, N);
  
  std::vector<double> check = {
      -0.1309,-0.0901,-0.1254,0.0252,-0.0038,
      0.0178,-0.1253,0.1186,-0.0980,0.0247,
      -0.1537,0.0066,0.1214,0.0501,0.0896,
      -0.0085,0.2047,0.2218,0.1987,0.0964,
      0.1727,0.1470,-0.0793,-0.0622,-0.0323
  };
  
  for (int64_t j=0; j<A.cols; ++j){
    for (int64_t i=0; i<A.rows; ++i){
      EXPECT_EQ(A(i, j), check.at(i*N+j));
    }
  }
}

TEST(MatrixGeneratorTests, from_csv2_no_params){
  const int64_t N = 5;

  //TODO change to relative path somehow
  Hatrix::Matrix A = Hatrix::generate_from_csv("/home/thomas/Hatrix/test/testdata/readCSV2.csv");
  EXPECT_EQ(A.rows, N);
  EXPECT_EQ(A.cols, N);
  
  std::vector<double> check = {
      -0.1309,1,6,0,0,
      0.0178,-0.1253,0.1186,-0.0980,0.0247,
      -0.1537,0,0,0,0,
      -0.0085,0.2047,0.2218,0.1987,0,
      3,5,100,0,0
  };
  
  for (int64_t j=0; j<A.cols; ++j){
    for (int64_t i=0; i<A.rows; ++i){
      EXPECT_EQ(A(i, j), check.at(i*N+j));
    }
  }
}

TEST_P(MatrixGeneratorTests, from_csv1){
  int64_t m, n;
  const int64_t N = 5;
  std::tie(m, n) = GetParam();

  //TODO change to relative path somehow
  Hatrix::Matrix A = Hatrix::generate_from_csv("/home/thomas/Hatrix/test/testdata/readCSV1.csv", ',', m, n);
  EXPECT_EQ(A.rows, m);
  EXPECT_EQ(A.cols, n);
  
  std::vector<double> check = {
      -0.1309,-0.0901,-0.1254,0.0252,-0.0038,
      0.0178,-0.1253,0.1186,-0.0980,0.0247,
      -0.1537,0.0066,0.1214,0.0501,0.0896,
      -0.0085,0.2047,0.2218,0.1987,0.0964,
      0.1727,0.1470,-0.0793,-0.0622,-0.0323
  };
  
  for (int64_t j=0; j<A.cols; ++j){
    for (int64_t i=0; i<A.rows; ++i){
      if (i<N && j<N)
        EXPECT_EQ(A(i, j), check.at(i*N+j));
      else
        EXPECT_EQ(A(i, j), 0);
    }
  }
}

TEST_P(MatrixGeneratorTests, from_csv2){
  int64_t m, n;
  const int64_t N = 5;
  std::tie(m, n) = GetParam();

  //TODO change to relative path somehow
  Hatrix::Matrix A = Hatrix::generate_from_csv("/home/thomas/Hatrix/test/testdata/readCSV2.csv", ',', m, n);
  EXPECT_EQ(A.rows, m);
  EXPECT_EQ(A.cols, n);
  
  std::vector<double> check = {
      -0.1309,1,6,0,0,
      0.0178,-0.1253,0.1186,-0.0980,0.0247,
      -0.1537,0,0,0,0,
      -0.0085,0.2047,0.2218,0.1987,0,
      3,5,100,0,0
  };  

  for (int64_t j=0; j<A.cols; ++j){
    for (int64_t i=0; i<A.rows; ++i){
      if (i<N && j<N)
        EXPECT_EQ(A(i, j), check.at(i*N+j));
      else
        EXPECT_EQ(A(i, j), 0);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
  MatrixGenerators, MatrixGeneratorTests,
 testing::Values(
    std::make_tuple(10, 10),
    std::make_tuple(2, 3),
    std::make_tuple(5, 1),
    std::make_tuple(1, 10),
    std::make_tuple(13, 4)
   ),
  [](const testing::TestParamInfo<MatrixGeneratorTests::ParamType>& info) {
    std::string name = (
      "m" + std::to_string(std::get<0>(info.param))
      + "n" + std::to_string(std::get<1>(info.param))
    );
    return name;
  }
);