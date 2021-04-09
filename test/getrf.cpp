#include "Hatrix/Hatrix.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <iostream>


TEST(LAPACKTests, getrf) {
  Hatrix::Matrix A(8, 8);
  A = 0.5;
  //set a large value on the diagonal to avoid pivoting
  int d = 100;
  for (int i=0; i<8; ++i){
    A(i,i) += d--;
  }
  Hatrix::Matrix L(8, 8), U(8, 8), A_check(8, 8);
  Hatrix::Matrix B(A);
  Hatrix::getrf(B);

  int idag = A.cols<A.rows?A.cols:A.rows;
  for (int i=0; i<idag; i++){
    L(i, i) = 1;
  }

  // extract L and U
  for (int i=0; i<A.rows; ++i) {
    for (int j=0; j<A.cols; ++j) {
      if (j<i){
        L(i, j) = B(i, j);
      }
      else {
        U(i, j) = B(i, j);
      }
    }
  }

  Hatrix::gemm(L, U, A_check, 'N', 'N', 1, 0);

  // Check result
  for (int i=0; i<A.rows; ++i) {
    for (int j=0; j<A.cols; ++j) {
      EXPECT_DOUBLE_EQ(A(i, j), A_check(i, j));
    }
  }
}