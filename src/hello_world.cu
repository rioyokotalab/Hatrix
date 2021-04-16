

#include <Hatrix/Hatrix.h>
#include <stdio.h>

int main() {
	Hatrix::init();
	

  int m = 16, n = 16, k = 16;
  Hatrix::Matrix A(m, k), B(k, n), C(m, n);
  A = 2;
  B = 4;
  C = 1;
  Hatrix::Matrix A_check(A), B_check(B), C_check(C);
  Hatrix::matmul(A, B, C, false, false, 1., 1.);

  cudaDeviceSynchronize();

  // Manual matmul
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k_ = 0; k_ < k; ++k_) {
        C_check(i, j) += A_check(i, k_) * B_check(k_, j);
      }
    }
  }

  double err = 0.;
  // Check result
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      err += (C_check(i, j) - C(i, j)) * (C_check(i, j) - C(i, j));
    }
  }

  printf("%f", err);

	Hatrix::terminate();
	return 0;
}
