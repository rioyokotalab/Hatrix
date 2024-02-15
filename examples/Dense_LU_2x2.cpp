#include <cstdint>
#include <iostream>
#include <vector>

#include "Hatrix/Hatrix.hpp"

// Simple demo of a 2x2 dense LU factorization using the Hatrix API.
// We initialize a SymmetricSharedBasisMatrix class and only make use
// of the D map within the class.

int main() {
  Hatrix::SymmetricSharedBasisMatrix A;
  int64_t block_size = 16;
  int64_t nblocks = 2;
  double d = 4 * block_size * block_size;

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < nblocks; ++j) {
      Hatrix::Matrix mat = Hatrix::generate_random_matrix(block_size, block_size);
      if (i == j) {
        // Add Large values to diagonal to assure no pivoting
        for (int64_t ii = 0; ii < block_size; ++ii) {
          mat(ii, ii) += d;
        }
      }

      A.D.insert(i, j, 0, std::move(mat));
    }
  }


  // b = A*x
  Hatrix::Matrix x0 = Hatrix::generate_random_matrix(block_size, 1);
  Hatrix::Matrix x1 = Hatrix::generate_random_matrix(block_size, 1);
  Hatrix::Matrix b0(block_size, 1), b1(block_size, 1);
  Hatrix::matmul(A.D(0, 0, 0), x0, b0, false, false, 1, 0);
  Hatrix::matmul(A.D(0, 1, 0), x1, b0, false, false, 1, 1);
  Hatrix::matmul(A.D(1, 0, 0), x0, b1, false, false, 1, 0);
  Hatrix::matmul(A.D(1, 1, 0), x1, b1, false, false, 1, 1);

  // Block LU
  Hatrix::Matrix L0(block_size, block_size);
  Hatrix::Matrix U0(block_size, block_size);
  Hatrix::lu(A.D(0, 0, 0), L0, U0);
  Hatrix::solve_triangular(L0, A.D(0, 1, 0), Hatrix::Left, Hatrix::Lower, true);
  Hatrix::solve_triangular(U0, A.D(1, 0, 0), Hatrix::Right, Hatrix::Upper, false);
  Hatrix::matmul(A.D(1, 0, 0), A.D(0, 1, 0), A.D(1, 1, 0), false, false, -1, 1);
  Hatrix::Matrix L1(block_size, block_size);
  Hatrix::Matrix U1(block_size, block_size);
  Hatrix::lu(A.D(1, 1, 0), L1, U1);

  // Forward substitution
  Hatrix::solve_triangular(L0, b0, Hatrix::Left, Hatrix::Lower, true);
  Hatrix::matmul(A.D(1, 0, 0), b0, b1, false, false, -1, 1);
  Hatrix::solve_triangular(L1, b1, Hatrix::Left, Hatrix::Lower, true);
  // Backward substitution
  Hatrix::solve_triangular(U1, b1, Hatrix::Left, Hatrix::Upper, false);
  Hatrix::matmul(A.D(0, 1, 0), b1, b0, false, false, -1, 1);
  Hatrix::solve_triangular(U0, b0, Hatrix::Left, Hatrix::Upper, false);

  // Check accuracy
  double error = Hatrix::norm(b0 - x0) + Hatrix::norm(b1 - x1);
  std::cout << "Solution error: " << error << "\n";
  return 0;
}
