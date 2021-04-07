#include "Hatrix/Hatrix.h"

#include <algorithm>
#include <iostream>

using namespace Hatrix;

int main() {
  //Full QR
  Matrix A(8, 4), Q(8, 4), R(4, 4);
  Matrix QR(8,4);
  A = 4.0;
  Matrix A_copy(A);
  qr(A, Q, R);
  gemm(Q, R, QR);
  
  bool correct = true;
  // Check result
  for (int i=0; i<QR.rows; i++) {
    for (int j=0; j<QR.cols; j++) {
      if (std::abs(QR(i, j) - A_copy(i, j)) > 10e-8) {
        correct = false;
        std::cout << i << " " << j << ": ";
        std::cout << QR(i, j) << " vs " << A_copy(i, j) << "\n";
      }
      break;
    }
  }
  return correct ? 0 : 1;
}
