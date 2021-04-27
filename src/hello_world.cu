

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

  //Manual matmul
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

  printf("%f\n", err);

  C.print();

  Hatrix::Matrix test(3, 3);
  test.data_[0] = 1;
  test.data_[1] = 2;
  test.data_[2] = 3;
  test.data_[3] = 2;
  test.data_[4] = 5;
  test.data_[5] = 10;
  test.data_[6] = 3;
  test.data_[7] = 7;
  test.data_[8] = 1;

  test.print();

  Hatrix::Matrix L(3, 3), U(3, 3), S(3, 3);
  Hatrix::svd(test, L, S, U);

  test.print();
  L.print();
  S.print();
  U.print();

	Hatrix::terminate();
	return 0;
}
