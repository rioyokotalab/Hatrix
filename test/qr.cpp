#include "Hatrix/Hatrix.h"

#include "gtest/gtest.h"

#include <cstdint>
using std::int64_t;
#include <iostream>
#include <tuple>


class QRTests
: public testing::TestWithParam<std::tuple<int64_t, int64_t, int64_t>> {};

TEST_P(QRTests, qr) {
  int64_t m, n, k;
  std::tie(m, n, k) = GetParam();
  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix Q(m, k), R(k, n);
  Hatrix::Matrix A_copy(A);
  Hatrix::qr(A, Q, R);
  Hatrix::Matrix QR(m, n);
  Hatrix::matmul(Q, R, QR, false, false, 1., 0.);
  // Check accuracy
  for (int64_t i=0; i<QR.rows; i++) {
    for (int64_t j=0; j<QR.cols; j++) {
      EXPECT_FLOAT_EQ(A_copy(i, j), QR(i, j));
    }
  }
  // Check orthogonality
  Hatrix::Matrix QTQ(Q.cols, Q.cols);
  Hatrix::matmul(Q, Q, QTQ, true, false, 1., 0.);
  for (int64_t i=0; i<QTQ.rows; i++) {
    for (int64_t j=0; j<QTQ.cols; j++) {
      if(i == j) EXPECT_NEAR(QTQ(i, j), 1.0, 10e-14);
      else EXPECT_NEAR(QTQ(i, j), 0.0, 10e-14);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
  LAPACK, QRTests,
  testing::Values(
    std::make_tuple(16, 16, 16),
    std::make_tuple(16, 8, 16),
    std::make_tuple(16, 8, 8),
    std::make_tuple(8, 16, 8)
   ),
  [](const testing::TestParamInfo<QRTests::ParamType>& info) {
    std::string name = (
      "m" + std::to_string(std::get<0>(info.param))
      + "n" + std::to_string(std::get<1>(info.param))
      + "k" + std::to_string(std::get<2>(info.param))
    );
    return name;
  }
);
