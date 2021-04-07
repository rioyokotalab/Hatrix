#include "Hatrix/Hatrix.h"

#include <algorithm>
#include <iostream>


int main() {
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

  bool correct = true;
  // Check result
  for (int i=0; i<C.rows; ++i) {
    for (int j=0; j<C.cols; ++j) {
      if (std::abs(C(i, j) - C_check(i, j)) > 10e-8) {
        correct = false;
        std::cout << i << " " << j << ": ";
        std::cout << C(i, j) << " vs " << C_check(i, j) << "\n";
      }
      break;
    }
  }
  return correct ? 0 : 1;
}
