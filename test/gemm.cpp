#include "Hatrix/Hatrix.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <iostream>


TEST(BLASTests, gemm) {
  Hatrix::Matrix A(8, 4), B(4, 8), C(8, 8);
  A = 2;
  B = 4;
  C = 1;
  Hatrix::Matrix A_check(A), B_check(B), C_check(C);
  Hatrix::gemm(A, B, C, 'N', 'N', 1., 1.);

  // Manual gemm
  for (int i=0; i<A.rows; ++i) {
    for (int j=0; j<B.cols; ++j) {
      for (int k=0; k<A.cols; ++k) {
        C_check(i, j) += A_check(i, k) * B_check(k, j);
      }
    }
  }

  // Check result
  for (int i=0; i<C.rows; ++i) {
    for (int j=0; j<C.cols; ++j) {
      ASSERT_FLOAT_EQ(C(i, j), C_check(i, j));
    }
  }
}
