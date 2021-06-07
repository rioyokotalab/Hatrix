#include <cstdint>
#include <iostream>
#include <tuple>

#include "Hatrix/Hatrix.h"
#include "gtest/gtest.h"

class QRTests
    : public testing::TestWithParam<std::tuple<int64_t, int64_t, int64_t>> {};
class HouseholderQRCompactWYTests
    : public testing::TestWithParam<std::tuple<int64_t, int64_t>> {};
class ApplyBlockReflectorTests
    : public testing::TestWithParam<std::tuple<int64_t, int64_t, int, bool>> {};

TEST_P(QRTests, qr) {
  Hatrix::Context::init();
  int64_t m, n, k;
  std::tie(m, n, k) = GetParam();
  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix Q(m, k), R(k, n);
  Hatrix::Matrix A_copy(A);
  Hatrix::qr(A, Q, R);
  Hatrix::Matrix QR(m, n);
  Hatrix::matmul(Q, R, QR, false, false, 1., 0.);
  Hatrix::Context::join();
  // Check accuracy
  for (int64_t i = 0; i < QR.rows; i++) {
    for (int64_t j = 0; j < QR.cols; j++) {
      EXPECT_NEAR(A_copy(i, j), QR(i, j), 10e-14);
    }
  }
  // Check orthogonality
  Hatrix::Matrix QTQ(Q.cols, Q.cols);
  Hatrix::matmul(Q, Q, QTQ, true, false, 1., 0.);
  for (int64_t i = 0; i < QTQ.rows; i++) {
    for (int64_t j = 0; j < QTQ.cols; j++) {
      if (i == j)
        EXPECT_NEAR(QTQ(i, j), 1.0, 10e-14);
      else
        EXPECT_NEAR(QTQ(i, j), 0.0, 10e-14);
    }
  }

  Hatrix::Context::finalize();
}

TEST_P(HouseholderQRCompactWYTests, HouseholderQRCompactWY) {
  int64_t m, n;
  std::tie(m, n) = GetParam();
  Hatrix::Matrix A = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix T(n, n);
  Hatrix::Matrix A_copy(A);

  Hatrix::householder_qr_compact_wy(A, T);

  // Separate A into Y\R
  Hatrix::Matrix R(m, n);
  Hatrix::Matrix Y(m, A.min_dim());
  for (int64_t j = 0; j < n; j++) {
    for (int64_t i = 0; i <= std::min(j, m - 1); i++) {
      R(i, j) = A(i, j);
    }
  }
  for (int64_t j = 0; j < A.min_dim(); j++) {
    Y(j, j) = 1.0;
    for (int64_t i = j + 1; i < m; i++) {
      Y(i, j) = A(i, j);
    }
  }

  // Construct Q = I-Y*T*(Y^T)
  Hatrix::Matrix YT(Y);
  Hatrix::triangular_matmul(T, YT, Hatrix::Right, Hatrix::Upper, false, false,
                            1.);
  Hatrix::Matrix Q = Hatrix::generate_identity_matrix(m, m);
  Hatrix::matmul(YT, Y, Q, false, true, -1., 1.);

  Hatrix::Matrix QR = Q * R;
  // Check accuracy
  for (int64_t i = 0; i < QR.rows; i++) {
    for (int64_t j = 0; j < QR.cols; j++) {
      EXPECT_NEAR(A_copy(i, j), QR(i, j), 10e-14);
    }
  }
  // Check orthogonality
  Hatrix::Matrix QTQ(Q.cols, Q.cols);
  Hatrix::matmul(Q, Q, QTQ, true, false, 1., 0.);
  for (int64_t i = 0; i < QTQ.rows; i++) {
    for (int64_t j = 0; j < QTQ.cols; j++) {
      if (i == j)
        EXPECT_NEAR(QTQ(i, j), 1.0, 10e-14);
      else
        EXPECT_NEAR(QTQ(i, j), 0.0, 10e-14);
    }
  }
}

TEST_P(ApplyBlockReflectorTests, ApplyBlockReflector) {
  int64_t m, n;
  int side;
  bool trans;
  std::tie(m, n, side, trans) = GetParam();
  Hatrix::Matrix C = Hatrix::generate_random_matrix(m, n);
  Hatrix::Matrix A = Hatrix::generate_random_matrix(
      side == Hatrix::Left ? m : n, side == Hatrix::Left ? m : n);
  Hatrix::Matrix T(A.cols, A.cols);
  Hatrix::householder_qr_compact_wy(A, T);
  Hatrix::Matrix C_copy(C);
  Hatrix::apply_block_reflector(A, T, C, side, trans);

  // Take Y from elements below diagonal of A
  Hatrix::Matrix Y(A.rows, A.min_dim());
  for (int64_t j = 0; j < A.min_dim(); j++) {
    Y(j, j) = 1.0;
    for (int64_t i = j + 1; i < A.rows; i++) {
      Y(i, j) = A(i, j);
    }
  }
  // Manually Construct H = I-Y*T*(Y^T) or I-Y*(T^T)*(Y^T)
  Hatrix::Matrix YT(Y);
  Hatrix::triangular_matmul(T, YT, Hatrix::Right, Hatrix::Upper, trans, false,
                            1.);
  Hatrix::Matrix H = Hatrix::generate_identity_matrix(A.rows, A.rows);
  Hatrix::matmul(YT, Y, H, false, true, -1., 1.);
  // Multiply H or H^T to C
  Hatrix::Matrix C_check = side == Hatrix::Left ? H * C_copy : C_copy * H;

  // Check result
  for (int64_t i = 0; i < m; i++) {
    for (int64_t j = 0; j < n; j++) {
      EXPECT_NEAR(C(i, j), C_check(i, j), 10e-14);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(LAPACK, QRTests,
                         testing::Values(std::make_tuple(16, 16, 16),
                                         std::make_tuple(16, 8, 16),
                                         std::make_tuple(16, 8, 8),
                                         std::make_tuple(8, 16, 8)));

INSTANTIATE_TEST_SUITE_P(LAPACK, HouseholderQRCompactWYTests,
                         testing::Values(std::make_tuple(16, 16),
                                         std::make_tuple(16, 8)));

INSTANTIATE_TEST_SUITE_P(
    LAPACK, ApplyBlockReflectorTests,
    testing::Combine(testing::Values(16, 32), testing::Values(16, 32),
                     testing::Values(Hatrix::Left, Hatrix::Right),
                     testing::Values(true, false)));
