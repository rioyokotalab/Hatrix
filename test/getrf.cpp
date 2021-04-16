#include "Hatrix/Hatrix.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <iostream>

void test_getrf(int m, int n, double value) {
  Hatrix::Matrix A(m, n);
  A = value;

  //set a large value on the diagonal to avoid pivoting
  int d = m * n;
  int n_diag = A.min_dim();
  for (int i=0; i<n_diag; ++i){
    A(i,i) += d--;
  }
  Hatrix::Matrix L(m, n_diag), U(n_diag, n), A_check(m, n);
  Hatrix::getrf(A, L, U);

  Hatrix::gemm(L, U, A_check, 'N', 'N', 1, 0);

  // Check result
  for (int i=0; i<A.rows; ++i) {
    for (int j=0; j<A.cols; ++j) {
      EXPECT_DOUBLE_EQ(A(i, j), A_check(i, j));
    }
  }
}


TEST(LAPACKTests, getrf) {
  test_getrf(8, 8, 0.5);
  test_getrf(4, 8, -2.3);
  test_getrf(10, 4, 11);
}