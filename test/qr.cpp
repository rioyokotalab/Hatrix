#include "Hatrix/Hatrix.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <iostream>

TEST(LAPACKTests, qr) {
  //Full QR
  Hatrix::Matrix A(8, 4), Q(8, 4), R(4, 4);
  Hatrix::Matrix QR(8, 4);
  A = 4.0;
  Hatrix::Matrix A_copy(A);
  Hatrix::qr(A_copy, Q, R);
  Hatrix::gemm(Q, R, QR, 'N', 'N', 1., 0.);

  // Check result
  for (int i=0; i<QR.rows; i++) {
    for (int j=0; j<QR.cols; j++) {
      EXPECT_NEAR(A(i, j), QR(i, j), 10e-14);
    }
  }
}
