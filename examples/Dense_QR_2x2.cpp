#include <cstdint>
#include <iostream>
#include <vector>

#include "Hatrix/Hatrix.hpp"
using namespace Hatrix;

// Simple demo of a 2x2 dense QR factorization using the Hatrix API.
// We initialize a SymmetricSharedBasisMatrix class and only make use
// of the D map within the class.


// ( A00 ) = I - ( Y00 ) * T * ( Y00^T Y10^T ) * ( A00 )
// ( A10 )       ( Y10 )                         ( A10 )
void apply_block_trapezoidal_reflector(const Hatrix::Matrix& Y00,
                                       const Hatrix::Matrix& Y10,
                                       const Hatrix::Matrix& T,
                                       Hatrix::Matrix& A00, Hatrix::Matrix& A10,
                                       bool trans) {
  Hatrix::Matrix C(A00);  // C = A00
  Hatrix::triangular_matmul(Y00, C, Hatrix::Left, Hatrix::Lower, true, true,
                            1.);                     // C = Y00^T * A01
  Hatrix::matmul(Y10, A10, C, true, false, 1., 1.);  // C = C + Y10^T * A11
  Hatrix::triangular_matmul(T, C, Hatrix::Left, Hatrix::Upper, trans, false,
                            1.);  // C = T*C
  Hatrix::Matrix YC(C);
  Hatrix::triangular_matmul(Y00, YC, Hatrix::Left, Hatrix::Lower, false, true,
                            1.);  // YC = Y00 * C
  A00 -= YC;
  A10 -= Y10 * C;
}

int main() {
  const int64_t block_size = 4;
  const int64_t nblocks = 2;
  SymmetricSharedBasisMatrix A;

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < nblocks; ++j) {
      A.D.insert(i, j, 0, Hatrix::generate_random_matrix(block_size, block_size));
    }
  }

  // Create big Dense A for accuracy evaluation
  Hatrix::Matrix Dense_A(2 * block_size, 2 * block_size);
  for (int64_t i = 0; i < block_size; ++i) {
    for (int64_t j = 0; j < block_size; ++j) {
      Dense_A(i, j) = A.D(0, 0, 0)(i, j);
      Dense_A(i, j + block_size) = A.D(0, 1, 0)(i, j);
      Dense_A(i + block_size, j) = A.D(1, 0, 0)(i, j);
      Dense_A(i + block_size, j + block_size) = A.D(1, 1, 0)(i, j);
    }
  }

  // Block QR
  // QR(A[*][0]) = Q0 ( R00 )
  //                  (  0  )
  Hatrix::Matrix T0(block_size, block_size);
  Hatrix::Matrix A0(2 * block_size, block_size);
  for (int64_t j = 0; j < block_size; j++) {
    for (int64_t i = 0; i < block_size; i++) {
      A0(i, j) = A.D(0, 0, 0)(i, j);
    }
    for (int64_t i = 0; i < block_size; i++) {
      A0(block_size + i, j) = A.D(1, 0, 0)(i, j);
    }
  }
  Hatrix::householder_qr_compact_wy(A0, T0);
  for (int64_t j = 0; j < block_size; j++) {
    for (int64_t i = 0; i < block_size; i++) {
      A.D(0, 0, 0)(i, j) = A0(i, j);
    }
    for (int64_t i = 0; i < block_size; i++) {
      A.D(1, 0, 0)(i, j) = A0(block_size + i, j);
    }
  }
  // Apply Q0^T to ( A01 ) = ( R01 ) from left
  //               ( A11 )   ( A11 )
  apply_block_trapezoidal_reflector(A.D(0, 0, 0), A.D(1, 0, 0), T0, A.D(0, 1, 0), A.D(1, 1, 0),
                                    true);
  // QR(A11) = Q'11 * R11
  Hatrix::Matrix T1(block_size, block_size);
  Hatrix::householder_qr_compact_wy(A.D(1, 1, 0), T1);

  // Construct Q
  std::vector<std::vector<Hatrix::Matrix>> Q(2);
  Q[0] = std::vector<Hatrix::Matrix>{Hatrix::Matrix(block_size, block_size),
                                     Hatrix::Matrix(block_size, block_size)};
  Q[1] = std::vector<Hatrix::Matrix>{Hatrix::Matrix(block_size, block_size),
                                     Hatrix::Matrix(block_size, block_size)};
  for (int64_t i = 0; i < block_size; ++i) {
    Q[0][0](i, i) = 1.;
    Q[1][1](i, i) = 1.;
  }
  // Apply Q'11 to Q11 from left
  apply_block_reflector(A.D(1, 1, 0), T1, Q[1][1], Hatrix::Left, false);
  // Apply Q0 to ( Q00 ) and ( Q01 ) from left
  //             ( Q10 )     ( Q11 )
  apply_block_trapezoidal_reflector(A.D(0, 0, 0), A.D(1, 0, 0), T0, Q[0][0], Q[1][0],
                                    false);
  apply_block_trapezoidal_reflector(A.D(0, 0, 0), A.D(1, 0, 0), T0, Q[0][1], Q[1][1],
                                    false);

  // Convert Q, R to Dense
  Hatrix::Matrix Dense_Q(2 * block_size, 2 * block_size);
  for (int64_t i = 0; i < block_size; i++) {
    for (int64_t j = 0; j < block_size; j++) {
      Dense_Q(i, j) = Q[0][0](i, j);
      Dense_Q(i, j + block_size) = Q[0][1](i, j);
      Dense_Q(i + block_size, j) = Q[1][0](i, j);
      Dense_Q(i + block_size, j + block_size) = Q[1][1](i, j);
    }
  }
  Hatrix::Matrix Dense_R(2 * block_size, 2 * block_size);
  for (int64_t i = 0; i < block_size; i++) {
    for (int64_t j = 0; j < block_size; j++) {
      if (i <= j) {
        Dense_R(i, j) = A.D(0, 0, 0)(i, j);
        Dense_R(i + block_size, j + block_size) = A.D(1, 1, 0)(i, j);
      }
      Dense_R(i, j + block_size) = A.D(0, 1, 0)(i, j);
    }
  }

  // Check accuracy and orthogonality
  Hatrix::Matrix Dense_QR = Dense_Q * Dense_R;
  std::cout << "norm(A-Q*R) = " << Hatrix::norm(Dense_A - Dense_QR)
            << std::endl;

  Hatrix::Matrix Dense_QTQ(Dense_Q.cols, Dense_Q.cols);
  Hatrix::matmul(Dense_Q, Dense_Q, Dense_QTQ, true, false, 1., 0.);
  Hatrix::Matrix Id =
      Hatrix::generate_identity_matrix(Dense_QTQ.rows, Dense_QTQ.cols);
  std::cout << "norm(I-Q^T*Q) = " << Hatrix::norm(Id - Dense_QTQ) << std::endl;

  return 0;
}
