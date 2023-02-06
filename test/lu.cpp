#include <cstdint>
#include <string>
#include <tuple>

#include "Hatrix/Hatrix.h"
#include "gtest/gtest.h"

template <typename DT>
class LuTests : public testing::Test {
  protected:
  // Matrix dimensions used in the tests
  std::vector<int64_t> sizes = {8, 16, 32};
  std::vector<std::tuple<int64_t, int64_t>> dims;

  void SetUp() override {
    for (size_t i = 0; i < sizes.size(); ++i) {
      for (size_t j = 0; j < sizes.size(); ++j) {
        dims.push_back(std::make_tuple(sizes[i], sizes[j]));
      }
    }
  }
};

// templated function to compare floats and doubles respectively
template <typename DT>
void inline expect_fp_eq(const DT a, const DT b, const std::basic_string<char>& err_msg) {
  if (std::is_same<DT, double>::value){
    EXPECT_DOUBLE_EQ(a, b) << err_msg;
  }     
  else {
    EXPECT_FLOAT_EQ(a, b) << err_msg;
  }                                        
}

// template types used in the tests
using Types = ::testing::Types<float, double>;
TYPED_TEST_SUITE(LuTests, Types);

TYPED_TEST(LuTests, LU) {
  for (auto const& [m, n] : this->dims) {

    Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(m, n);

    // Set a large value on the diagonal to avoid pivoting
    int64_t d = m * n;
    int64_t n_diag = A.min_dim();
    for (int64_t i = 0; i < n_diag; ++i) {
      A(i, i) += d--;
    }

    Hatrix::Matrix<TypeParam> A_copy(A);
    Hatrix::Matrix<TypeParam> L(m, n_diag), U(n_diag, n), A_rebuilt(m, n);
    Hatrix::lu(A, L, U);
    Hatrix::matmul(L, U, A_rebuilt, false, false, 1, 0);

    // Check result
    for (int64_t i = 0; i < A.rows; ++i) {
      for (int64_t j = 0; j < A.cols; ++j) {
        expect_fp_eq(A_rebuilt(i, j), A_copy(i, j),
        "Wrong value at index ["+std::to_string(i)+", "+std::to_string(j)
      +"] (" + std::to_string(m) + "x" + std::to_string(n) + " matrix)");
      }
    }
  }
}
