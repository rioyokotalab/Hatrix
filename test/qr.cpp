#include "Hatrix/Hatrix.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <iostream>


class QRTests : public testing::TestWithParam<std::tuple<int, int, int>> {};

TEST_P(QRTests, qr) {
  int m, n, k;
  std::tie(m, n, k) = GetParam();
  Hatrix::Matrix A(m, n), Q(m, k), R(k, n);
  Hatrix::Matrix QR(m, n);
  A = 4.0;
  Hatrix::qr(A, Q, R);
  Hatrix::matmul(Q, R, QR, false, false, 1., 0.);
  // Check accuracy
  for (int i=0; i<QR.rows; i++) {
    for (int j=0; j<QR.cols; j++) {
      EXPECT_NEAR(A(i, j), QR(i, j), 10e-14);
    }
  }
  // Check orthogonality
  Hatrix::Matrix QTQ(4, 4);
  Hatrix::Matrix Q_copy(Q);
  Hatrix::matmul(Q, Q_copy, QTQ, true, false, 1., 0.);
  for (int i=0; i<QTQ.rows; i++) {
    for (int j=0; j<QTQ.cols; j++) {
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
