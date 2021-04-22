#include "Hatrix/Hatrix.h"

#include "gtest/gtest.h"

#include <cstdint>
using std::uint64_t;
#include <string>
#include <tuple>


class LUTests : public testing::TestWithParam<std::tuple<uint64_t, uint64_t>>{};

TEST_P(LUTests, lu){
  uint64_t m, n;
  std::tie(m, n) = GetParam();

  Hatrix::Matrix A(m, n);
  A = 1.5;

  //set a large value on the diagonal to avoid pivoting
  uint64_t d = m * n;
  uint64_t n_diag = A.min_dim();
  for (uint64_t i=0; i<n_diag; ++i){
    A(i,i) += d--;
  }

  Hatrix::Matrix A_copy(A);
  Hatrix::Matrix L(m, n_diag), U(n_diag, n), A_rebuilt(m, n);
  Hatrix::lu(A, L, U);
  Hatrix::matmul(L, U, A_rebuilt, false, false, 1, 0);

    // Check result
  for (uint64_t i=0; i<A.rows; ++i) {
    for (uint64_t j=0; j<A.cols; ++j) {
      EXPECT_DOUBLE_EQ(A_rebuilt(i, j), A_copy(i, j));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
  LAPACK, LUTests,
  testing::Combine(
    testing::Values(8, 16, 32),
    testing::Values(8, 16, 32)
  ),
  [](const testing::TestParamInfo<LUTests::ParamType>& info) {
    std::string name = (
      "m" + std::to_string(std::get<0>(info.param))
      + "n" + std::to_string(std::get<1>(info.param))
    );
    return name;
  }
);
