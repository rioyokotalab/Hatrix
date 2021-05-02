#include "Hatrix/Hatrix.h"

#include "gtest/gtest.h"

#include <cstdint>
using std::int64_t;


class ErrorTests : public testing::Test {
  protected:
    void SetUp() override {
      A = Hatrix::generate_from_csv("../test/testdata/A_error.csv");
      b = Hatrix::generate_from_csv("../test/testdata/b_error.csv");
      x = Hatrix::generate_from_csv("../test/testdata/x_error.csv");
      res = Hatrix::generate_from_csv("../test/testdata/res_error.csv");
    }
  
    Hatrix::Matrix A, b, x, res;
};


TEST_F(ErrorTests, NormwiseBackwardError){
  double nbe = 1.368558645412928e-16;
  EXPECT_FLOAT_EQ(Hatrix::norm_bw_error(res, A, x, b), nbe);
}

TEST_F(ErrorTests, ComponentwiseBackwardError){
  double cbe = 5.624678270495485e-16;
  EXPECT_FLOAT_EQ(Hatrix::comp_bw_error(res, A, x, b), cbe);
}
