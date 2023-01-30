#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "Hatrix/Hatrix.h"
#include "gtest/gtest.h"

template <typename DT>
class ScaleTests : public testing::Test {
  protected:
  // Matrix dimensions and scalar parameter used in the tests
  std::vector<std::tuple<int64_t, int64_t, DT>> params = {
    std::make_tuple(10, 10, 4.32),
    std::make_tuple(1, 7, 2),
    std::make_tuple(15, 3, 99.9),
    std::make_tuple(4, 1, 0.5),
    std::make_tuple(8, 21, -3.4)
  };
};

// template types used in the tests
using Types = ::testing::Types<float, double>;
TYPED_TEST_SUITE(ScaleTests, Types);

TYPED_TEST(ScaleTests, Scaling) {
  for (auto const& [m, n, alpha] : this->params) {
    Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(m, n);
    Hatrix::Matrix<TypeParam> A_copy(A);

    Hatrix::scale(A, alpha);
    for (int64_t j = 0; j < A.cols; ++j) {
      for (int64_t i = 0; i < A.rows; ++i) {
        EXPECT_EQ(A(i, j), A_copy(i, j) * alpha) << "Wrong value at index [" << i << ", " << j << "] ("
          << m << "x" << n << " matrix, alpha = "
          << alpha << ")";
      }
    }
  }
}

//TODO What is this even testing?
TYPED_TEST(ScaleTests, ScalingPart) {
  for (auto const& [m, n, alpha] : this->params) {
    Hatrix::Matrix<TypeParam> A_big = Hatrix::generate_random_matrix<TypeParam>(2*m, 2*n);
    //TODO is there a way around this?
    auto A_split = A_big.split(2, 2);
    Hatrix::Matrix<TypeParam> A_copy(A_split[0], true);

    Hatrix::scale(A_split[0], alpha);
    for (int64_t j = 0; j < A_split[0].cols; ++j) {
      for (int64_t i = 0; i < A_split[0].rows; ++i) {
        EXPECT_EQ(A_big(i, j), A_copy(i, j) * alpha) << "Wrong value at index [" << i << ", " << j << "] ("
          << m << "x" << n << " matrix, alpha = "
          << alpha << ")";
      }
    }
  }
}

TYPED_TEST(ScaleTests, RowScaling) {
  for (auto const& [m, n, alpha] : this->params) {
    Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(m, n);
    Hatrix::Matrix<TypeParam> A_copy(A);
    Hatrix::Matrix<TypeParam> D = Hatrix::generate_random_matrix<TypeParam>(m, m);

    Hatrix::row_scale(A, D);
    for (int64_t j = 0; j < A.cols; ++j) {
      for (int64_t i = 0; i < A.rows; ++i) {
        EXPECT_EQ(A(i, j), A_copy(i, j) * D(i, i)) << "Wrong value at index [" << i << ", " << j << "] ("
          << m << "x" << n << " matrix, alpha = "
          << alpha << ")";
      }
    }
  }
}

TYPED_TEST(ScaleTests, ColumnScaling) {
  for (auto const& [m, n, alpha] : this->params) {
    Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(m, n);
    Hatrix::Matrix<TypeParam> A_copy(A);
    Hatrix::Matrix<TypeParam> D = Hatrix::generate_random_matrix<TypeParam>(n, n);

    Hatrix::column_scale(A, D);
    for (int64_t j = 0; j < A.cols; ++j) {
      for (int64_t i = 0; i < A.rows; ++i) {
        EXPECT_EQ(A(i, j), A_copy(i, j) * D(j, j)) << "Wrong value at index [" << i << ", " << j << "] ("
          << m << "x" << n << " matrix, alpha = "
          << alpha << ")";
      }
    }
  }
}
