#include <cstdint>
#include <iostream>
#include <tuple>

#include "Hatrix/Hatrix.h"
#include "gtest/gtest.h"

template <typename DT>
class RqTests : public testing::Test {
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
    EXPECT_NEAR(a, b, 10e-15) << err_msg;
  }     
  else {
    EXPECT_NEAR(a, b, 10e-7) << err_msg;
  }                                        
}

// template types used in the tests
using Types = ::testing::Types<float, double>;
TYPED_TEST_SUITE(RqTests, Types);

TYPED_TEST(RqTests, Rq) {
  for (auto const& [m, n] : this->dims) {
    auto k = m > n ? n : m;
    Hatrix::Matrix<TypeParam> A = Hatrix::generate_random_matrix<TypeParam>(m, n);
    Hatrix::Matrix<TypeParam> R(m, k), Q(k, n);
    Hatrix::Matrix<TypeParam> A_copy(A);
    Hatrix::rq(A, R, Q);
    Hatrix::Matrix<TypeParam> RQ = Hatrix::matmul(R, Q);
    // Check accuracy
    for (int64_t i = 0; i < RQ.rows; i++) {
      for (int64_t j = 0; j < RQ.cols; j++) {
        expect_fp_eq(A_copy(i, j), RQ(i, j),
          "Wrong value at index ["+std::to_string(i)+", "+std::to_string(j)
          +"] (" + std::to_string(m) + "x" + std::to_string(m) + " matrix)");
      }
    }
    // Check orthogonality
    Hatrix::Matrix<TypeParam> QQt = Hatrix::matmul(Q, Q, false, true);
    for (int64_t i = 0; i < QQt.rows; i++) {
      for (int64_t j = 0; j < QQt.cols; j++) {
        if (i == j)
          expect_fp_eq(QQt(i, j), (TypeParam) 1.0,
            "Not orthogonal at ["+std::to_string(i)+", "+std::to_string(j)
            +"] (" + std::to_string(m) + "x" + std::to_string(m) + " matrix)");
        else
          expect_fp_eq(QQt(i, j), (TypeParam) 0.0,
            "Not orthogonal at ["+std::to_string(i)+", "+std::to_string(j)
            +"] (" + std::to_string(m) + "x" + std::to_string(m) + " matrix)");
      }
    }
  }
}


class RQTests
    : public testing::TestWithParam<std::tuple<int64_t, int64_t, int64_t>> {};

TEST_P(RQTests, rq) {
  Hatrix::Context::init();
  int64_t m, n, k;
  std::tie(m, n, k) = GetParam();
  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix R(m, k), Q(k, n);
  Hatrix::Matrix A_copy(A);
  Hatrix::rq(A, R, Q);
  Hatrix::Matrix RQ = Hatrix::matmul(R, Q);
  // Check accuracy
  for (int64_t i = 0; i < RQ.rows; i++) {
    for (int64_t j = 0; j < RQ.cols; j++) {
      EXPECT_NEAR(A_copy(i, j), RQ(i, j), 1e-12);
    }
  }
  // Check orthogonality
  Hatrix::Matrix QQt = Hatrix::matmul(Q, Q, false, true);
  for (int64_t i = 0; i < QQt.rows; i++) {
    for (int64_t j = 0; j < QQt.cols; j++) {
      if (i == j)
        EXPECT_NEAR(QQt(i, j), 1.0, 1e-12);
      else
        EXPECT_NEAR(QQt(i, j), 0.0, 1e-12);
    }
  }

  Hatrix::Context::finalize();
}


INSTANTIATE_TEST_SUITE_P(LAPACK, RQTests,
                         testing::Values(std::make_tuple(16, 16, 16),
                                         std::make_tuple(16, 8, 8),
                                         std::make_tuple(8, 16, 16),
                                         std::make_tuple(8, 16, 8)));
