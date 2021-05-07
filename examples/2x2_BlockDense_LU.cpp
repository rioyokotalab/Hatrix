#include <cstdint>
#include <iostream>
#include <vector>

#include "Hatrix/Hatrix.h"

int main() {
  Hatrix::init();
  int64_t block_size = 16;
  std::vector<std::vector<Hatrix::Matrix>> A(2);
  A[0] = std::vector<Hatrix::Matrix>{
      Hatrix::generate_random_matrix(block_size, block_size),
      Hatrix::generate_random_matrix(block_size, block_size)};
  A[1] = std::vector<Hatrix::Matrix>{
      Hatrix::generate_random_matrix(block_size, block_size),
      Hatrix::generate_random_matrix(block_size, block_size)};
  // Add Large values to diagonal to assure no pivoting
  double d = 4 * block_size * block_size;
  for (int64_t i = 0; i < block_size; ++i) {
    A[0][0](i, i) += d--;
  }
  for (int64_t i = 0; i < block_size; ++i) {
    A[1][1](i, i) += d--;
  }

  // b = A*x
  Hatrix::Matrix x0 = Hatrix::generate_random_matrix(block_size, 1);
  Hatrix::Matrix x1 = Hatrix::generate_random_matrix(block_size, 1);
  Hatrix::Matrix b0(block_size, 1), b1(block_size, 1);
  Hatrix::matmul(A[0][0], x0, b0, false, false, 1, 0);
  Hatrix::matmul(A[0][1], x1, b0, false, false, 1, 1);
  Hatrix::matmul(A[1][0], x0, b1, false, false, 1, 0);
  Hatrix::matmul(A[1][1], x1, b1, false, false, 1, 1);

  // Block LU
  Hatrix::Matrix L0(block_size, block_size);
  Hatrix::Matrix U0(block_size, block_size);
  Hatrix::lu(A[0][0], L0, U0);
  Hatrix::solve_triangular(L0, A[0][1], Hatrix::Left, Hatrix::Lower, true);
  Hatrix::solve_triangular(U0, A[1][0], Hatrix::Right, Hatrix::Upper, false);
  Hatrix::matmul(A[1][0], A[0][1], A[1][1], false, false, -1, 1);
  Hatrix::Matrix L1(block_size, block_size);
  Hatrix::Matrix U1(block_size, block_size);
  Hatrix::lu(A[1][1], L1, U1);

  // Forward substitution
  Hatrix::solve_triangular(L0, b0, Hatrix::Left, Hatrix::Lower, true);
  Hatrix::matmul(A[1][0], b0, b1, false, false, -1, 1);
  Hatrix::solve_triangular(L1, b1, Hatrix::Left, Hatrix::Lower, true);
  // Backward substitution
  Hatrix::solve_triangular(U1, b1, Hatrix::Left, Hatrix::Upper, false);
  Hatrix::matmul(A[0][1], b1, b0, false, false, -1, 1);
  Hatrix::solve_triangular(U0, b0, Hatrix::Left, Hatrix::Upper, false);

  // Check accuracy
  double error = (Hatrix::frobenius_norm_diff(b0, x0) +
                  Hatrix::frobenius_norm_diff(b1, x1));
  std::cout << "Solution error: " << error << "\n";
  Hatrix::terminate();
  return 0;
}
