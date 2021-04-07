#include "Hatrix/Hatrix.h"

#include <algorithm>
#include <iostream>


int main() {
  Hatrix::Matrix A(8, 8);
  //set a large value on the diagonal to avoid pivoting
  int d = 100;
  for (int i=0; i<8; ++i){
    A(i,i) = d--;
  }
  Hatrix::Matrix L(8, 8), U(8, 8), A_check(8, 8);
  Hatrix::Matrix B(A);
  Hatrix::getrf(B);

  for (int i=0; i<A.rows; ++i) {
    for (int j=0; j<A.cols; ++j) {
        L(j,i) = 0;
        U(j,i) = 0;
        A_check(j,i) = 0;
    }
    }

  int idag = A.cols<A.rows?A.cols:A.rows;
  for (int i=0; i<idag; i++){
      L(i,i) = 1;
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

  Hatrix::gemm(L, U, A_check);

  bool correct = true;
  // Check result
  for (int i=0; i<A.rows; ++i) {
    for (int j=0; j<A.cols; ++j) {
      if (std::abs(A(i, j) - A_check(i, j)) > 10e-8) {
        correct = false;
        std::cout << i << " " << j << ": ";
        std::cout << A(i, j) << " vs " << A_check(i, j) << "\n";
      }
      break;
    }
  }
  return correct ? 0 : 1;
}