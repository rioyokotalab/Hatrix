#include "Hatrix/Hatrix.h"

#include "gtest/gtest.h"

#include <iostream>
#include <vector>


// ( A00 ) = I - ( Y00 ) * T * ( Y00^T Y10^T ) * ( A00 )
// ( A10 )       ( Y10 )                         ( A10 )
void apply_block_trapezoidal_reflector(
  const Hatrix::Matrix& Y00, const Hatrix::Matrix& Y10, const Hatrix::Matrix& T,
  Hatrix::Matrix& A00, Hatrix::Matrix& A10, bool trans
) {
  Hatrix::Matrix C(A00); // C = A00
  Hatrix::triangular_matmul(
    Y00, C, Hatrix::Left, Hatrix::Lower, true, true, 1.
  ); //C = Y00^T * A01
  Hatrix::matmul(Y10, A10, C, true, false, 1., 1.); // C = C + Y10^T * A11
  Hatrix::triangular_matmul(
    T, C, Hatrix::Left, Hatrix::Upper, trans, false, 1.
  ); //C = T*C
  Hatrix::Matrix YC(C);
  Hatrix::triangular_matmul(
    Y00, YC, Hatrix::Left, Hatrix::Lower, false, true, 1.
  ); //YC = Y00 * C
  Hatrix::Matrix identity(YC.rows, YC.rows);
  for(int64_t i = 0; i < identity.rows; i++) identity(i, i) = 1.;
  Hatrix::matmul(
    identity, YC, A00, false, false, -1., 1.
  ); // A00 = A00 - YC
  Hatrix::matmul(
    Y10, C, A10, false, false, -1., 1.
  ); // A10 = A10 - Y10 * C
}


TEST(BlockDense, qr) {
  int64_t block_size = 4;
  std::vector<std::vector<Hatrix::Matrix>> A(2);
  A[0] = std::vector<Hatrix::Matrix>{
    Hatrix::Matrix(block_size, block_size),
    Hatrix::Matrix(block_size, block_size)
  };
  A[1] = std::vector<Hatrix::Matrix>{
    Hatrix::Matrix(block_size, block_size),
    Hatrix::Matrix(block_size, block_size)
  };
  Hatrix::Matrix Dense_A(2*block_size, 2*block_size);
  // Initialize to non-singular matrix
  for (int64_t i=0; i<block_size; ++i)
    for (int64_t j=0; j<block_size; ++j) {
      A[0][0](i, j) = 1;
      A[0][1](i, j) = 2;
      A[1][0](i, j) = 3;
      A[1][1](i, j) = 4;
      Dense_A(i, j) = 1;
      Dense_A(i, j+block_size) = 2;
      Dense_A(i+block_size, j) = 3;
      Dense_A(i+block_size, j+block_size) = 4;
    }
  for (int64_t i=0; i<block_size; ++i) {
    A[0][0](i, i) += 100;
    A[1][1](i, i) += 100;
    Dense_A(i, i) += 100;
    Dense_A(i+block_size, i+block_size) += 100;
  }

  // Block QR
  // QR(A[*][0]) = Q0 ( R00 )
  //                  (  0  )
  Hatrix::Matrix T0(block_size, block_size);
  Hatrix::Matrix A0(2*block_size, block_size);
  for(int64_t j=0; j<block_size; j++) {
    for(int64_t i=0; i<block_size; i++) {
      A0(i, j) = A[0][0](i, j);
    }
    for(int64_t i=0; i<block_size; i++) {
      A0(block_size+i, j) = A[1][0](i, j);
    }
  }
  Hatrix::householder_qr_compact_wy(A0, T0);
  for(int64_t j=0; j<block_size; j++) {
    for(int64_t i=0; i<block_size; i++) {
      A[0][0](i, j) = A0(i, j);
    }
    for(int64_t i=0; i<block_size; i++) {
      A[1][0](i, j) = A0(block_size+i, j);
    }
  }
  // Apply Q0^T to ( A01 ) = ( R01 ) from left
  //               ( A11 )   ( A11 )
  apply_block_trapezoidal_reflector(A[0][0], A[1][0], T0, A[0][1], A[1][1], true);
  // QR(A11) = Q'11 * R11
  Hatrix::Matrix T1(block_size, block_size);
  Hatrix::householder_qr_compact_wy(A[1][1], T1);

  // Construct Q
  std::vector<std::vector<Hatrix::Matrix>> Q(2);
  Q[0] = std::vector<Hatrix::Matrix>{
    Hatrix::Matrix(block_size, block_size),
    Hatrix::Matrix(block_size, block_size)
  };
  Q[1] = std::vector<Hatrix::Matrix>{
    Hatrix::Matrix(block_size, block_size),
    Hatrix::Matrix(block_size, block_size)
  };
  for(int64_t i=0; i<block_size; ++i) {
    Q[0][0](i, i) = 1.;
    Q[1][1](i, i) = 1.;
  }
  // Apply Q'11 to Q11 from left
  apply_block_reflector(A[1][1], T1, Q[1][1], Hatrix::Left, false);
  // Apply Q0 to ( Q00 ) and ( Q01 ) from left
  //             ( Q10 )     ( Q11 )
  apply_block_trapezoidal_reflector(A[0][0], A[1][0], T0, Q[0][0], Q[1][0], false);
  apply_block_trapezoidal_reflector(A[0][0], A[1][0], T0, Q[0][1], Q[1][1], false);

  // Convert Q, R to Dense
  Hatrix::Matrix Dense_Q(2*block_size, 2*block_size);
  for(int64_t i=0; i<block_size; i++)
    for(int64_t j=0; j<block_size; j++) {
      Dense_Q(i, j) = Q[0][0](i, j);
      Dense_Q(i, j+block_size) = Q[0][1](i, j);
      Dense_Q(i+block_size, j) = Q[1][0](i, j);
      Dense_Q(i+block_size, j+block_size) = Q[1][1](i, j);
    }
  Hatrix::Matrix Dense_R(2*block_size, 2*block_size);
  for(int64_t i=0; i<block_size; i++)
    for(int64_t j=0; j<block_size; j++) {
      if(i <= j) {
	Dense_R(i, j) = A[0][0](i, j);
	Dense_R(i+block_size, j+block_size) = A[1][1](i, j);
      }
      Dense_R(i, j+block_size) = A[0][1](i, j);      
    }
  
  // Check accuracy and orthogonality
  Hatrix::Matrix Dense_QR(2*block_size, 2*block_size);
  Hatrix::matmul(Dense_Q, Dense_R, Dense_QR, false, false, 1., 0.);
  // Check accuracy
  for (int64_t i=0; i<Dense_QR.rows; i++) {
    for (int64_t j=0; j<Dense_QR.cols; j++) {
      EXPECT_NEAR(Dense_A(i, j), Dense_QR(i, j), 10e-14);
    }
  }
  // Check orthogonality
  Hatrix::Matrix Dense_QTQ(Dense_Q.cols, Dense_Q.cols);
  Hatrix::matmul(Dense_Q, Dense_Q, Dense_QTQ, true, false, 1., 0.);
  for (int64_t i=0; i<Dense_QTQ.rows; i++) {
    for (int64_t j=0; j<Dense_QTQ.cols; j++) {
      if(i == j) EXPECT_NEAR(Dense_QTQ(i, j), 1.0, 10e-14);
      else EXPECT_NEAR(Dense_QTQ(i, j), 0.0, 10e-14);
    }
  }
}
