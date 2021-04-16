#include "Hatrix/Hatrix.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <iostream>

void test_lu(int m, int n, double value) {
  Hatrix::Matrix A(m, n);
  A = value;

  //set a large value on the diagonal to avoid pivoting
  int d = m * n;
  int n_diag = A.min_dim();
  for (int i=0; i<n_diag; ++i){
    A(i,i) += d--;
  }
  Hatrix::Matrix A_copy(A);

  Hatrix::Matrix L(m, n_diag), U(n_diag, n), A_rebuilt(m, n);
  Hatrix::lu(A, L, U);

  Hatrix::matmul(L, U, A_rebuilt, false, false, 1, 0);

  // Check result
  for (int i=0; i<A.rows; ++i) {
    for (int j=0; j<A.cols; ++j) {
      EXPECT_DOUBLE_EQ(A_rebuilt(i, j), A_copy(i, j));
    }
  }
}

TEST(LAPACKTests, lu) {
  test_lu(8, 8, 0.5);
  test_lu(4, 8, -2.3);
  test_lu(10, 4, 11);
}
