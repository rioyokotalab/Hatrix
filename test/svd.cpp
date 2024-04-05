#include <algorithm>
#include <cassert>
#include <tuple>

#include "Hatrix/Hatrix.h"
#include "gtest/gtest.h"

template <typename DT>
class SvdTests : public testing::Test {
  protected:
  // Matrix dimensions used in the tests
  std::vector<int64_t> sizes = {8, 16};
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
    // TÓDO this fails sometimes with 10e-15 by a tiny amount
    EXPECT_NEAR(a, b, 10e-14) << err_msg;
  }     
  else {
    EXPECT_NEAR(a, b, 10e-7) << err_msg;
  }                                        
}

// template types used in the tests
using Types = ::testing::Types<float, double>;
TYPED_TEST_SUITE(SvdTests, Types);

TYPED_TEST(SvdTests, Svd) {
  for (auto const& [m, n] : this->dims) {

    Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(m, n);

    int64_t s_dim = A.min_dim();
    Hatrix::Matrix<TypeParam> A_copy(A);
    Hatrix::Matrix<TypeParam> U(m, s_dim), S(s_dim, s_dim), V(s_dim, n), A_rebuilt(m, n);
    Hatrix::svd(A, U, S, V);
    Hatrix::Matrix<TypeParam> UxS(m, s_dim);
    Hatrix::matmul(U, S, UxS, false, false, 1, 0);
    Hatrix::matmul(UxS, V, A_rebuilt, false, false, 1, 0);

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

template <typename DT>
class truncatedSvdTests : public testing::Test {
  protected:
  // Matrix dimensions used in the tests
  std::vector<std::tuple<int64_t, int64_t, int64_t>> params = {
    std::make_tuple(50, 50, 7),
    std::make_tuple(100, 80, 10),
    std::make_tuple(90, 120, 14),
    std::make_tuple(100, 100, 5)
  };
};

// template types used in the tests
using Types = ::testing::Types<float, double>;
TYPED_TEST_SUITE(truncatedSvdTests, Types);

TYPED_TEST(truncatedSvdTests, truncatedSvd) {
  for (auto const& [m, n, rank] : this->params) {
    Hatrix::Matrix<TypeParam> A = Hatrix::generate_low_rank_matrix<TypeParam>(m, n);

    int64_t dmin = A.min_dim();
    Hatrix::Matrix<TypeParam> A_check(A);
    Hatrix::Matrix<TypeParam> U(m, dmin);
    Hatrix::Matrix<TypeParam> S(dmin, dmin);
    Hatrix::Matrix<TypeParam> V(dmin, n);
    double tolerance = Hatrix::truncated_svd(A, U, S, V, rank);

    Hatrix::Matrix<TypeParam> UxS(m, rank);
    Hatrix::matmul(U, S, UxS, false, false, 1, 0);
    Hatrix::matmul(UxS, V, A, false, false, 1, 0);
    double norm_diff = Hatrix::norm(A_check - A);
    expect_fp_eq((TypeParam) norm_diff, (TypeParam) tolerance,
        "Norms are different (" + std::to_string(m) + "x"
        + std::to_string(n) + " matrix)");
  }
}

TYPED_TEST(truncatedSvdTests, truncatedSvdReturn) {
  for (auto const& [m, n, rank] : this->params) {
    Hatrix::Matrix<TypeParam> A = Hatrix::generate_low_rank_matrix<TypeParam>(m, n);

    Hatrix::Matrix<TypeParam> A_check(A), U, S, V;
    double tolerance;
    std::tie(U, S, V, tolerance) = Hatrix::truncated_svd(A, rank);

    Hatrix::Matrix<TypeParam> UxS(m, rank);
    Hatrix::matmul(U, S, UxS, false, false, 1, 0);
    Hatrix::matmul(UxS, V, A, false, false, 1, 0);
    double norm_diff = Hatrix::norm(A_check - A);
    expect_fp_eq((TypeParam) norm_diff, (TypeParam) tolerance,
        "Norms are different (" + std::to_string(m) + "x"
        + std::to_string(n) + " matrix)");
  }
}
