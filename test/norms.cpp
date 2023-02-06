#include <cmath>
#include <cstdint>
#include <string>
#include <tuple>

#include "Hatrix/Hatrix.h"
#include "gtest/gtest.h"

template <typename DT>
class NORMTests : public testing::Test {
  protected:
  // Matrix dimensions used in the tests
  std::vector<int64_t> sizes = {1, 10, 20, 70, 99, 100};
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
    EXPECT_NEAR(a, b, 10e-13) << err_msg;
  }     
  else {
    EXPECT_NEAR(a, b, 10e-5) << err_msg;
  }                                        
}

// template types used in the tests
using Types = ::testing::Types<float, double>;
TYPED_TEST_SUITE(NORMTests, Types);

TYPED_TEST(NORMTests, Norm) {
  for (auto const& [m, n] : this->dims) {
    Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(m, n);

    TypeParam norm = 0;
    for (int64_t j = 0; j < A.cols; ++j) {
      for (int64_t i = 0; i < A.rows; ++i) {
        norm += A(i, j) * A(i, j);
      }
    }
    norm = std::sqrt(norm);

    expect_fp_eq(norm, Hatrix::norm(A),
      "Values are different (" + std::to_string(m) + "x"
      + std::to_string(n) + " matrix)");
  }
}
