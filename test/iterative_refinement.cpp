#include "Hatrix/Hatrix.h"

#include "gtest/gtest.h"

#include <cstdint>
using std::int64_t;


class IRTests : public testing::Test {
  protected:
    void SetUp() override {
      A = Hatrix::generate_from_csv("../test/testdata/A_error.csv");
      b = Hatrix::generate_from_csv("../test/testdata/b_error.csv");
      x = Hatrix::generate_from_csv("../test/testdata/x_error.csv");
      res = Hatrix::generate_from_csv("../test/testdata/res_error.csv");
    }
  
    Hatrix::Matrix A, b, x, res;
};


TEST_F(IRTests, IterativeRefinement){
  Hatrix::gesv_IR(A, b, 10);
}
