#include <algorithm>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Hatrix/Hatrix.h"

Hatrix::BLR construct_BLR(int64_t block_size, int64_t n_blocks, int64_t rank) {
  Hatrix::BLR A;
  for (int i = 0; i < n_blocks; ++i) {
    for (int j = 0; j < n_blocks; ++j) {
      if (i == j) {
        Hatrix::Matrix diag =
            Hatrix::generate_random_matrix(block_size, block_size);
        // Prevent pivoting
        for (int64_t i = 0; i < diag.min_dim(); ++i) diag(i, i) += 10;
        A.D.insert(i, j, std::move(diag));
      } else {
        A.D.insert(i, j,
                   Hatrix::generate_low_rank_matrix(block_size, block_size));
      }
    }
  }
  // Also store expected errors to check against later
  std::unordered_map<std::tuple<int64_t, int64_t>, double> expected_err;
  int64_t oversampling = 5;
  Hatrix::Matrix U, S, V;
  double error;
  std::vector<Hatrix::Matrix> Y;
  for (int64_t i = 0; i < n_blocks; ++i) {
    Y.push_back(
        Hatrix::generate_random_matrix(block_size, rank + oversampling));
  }
  for (int64_t i = 0; i < n_blocks; ++i) {
    Hatrix::Matrix AY(block_size, rank + oversampling);
    for (int64_t j = 0; j < n_blocks; ++j) {
      if (i == j) continue;
      Hatrix::matmul(A.D(i, j), Y[j], AY);
    }
    std::tie(U, S, V, error) = Hatrix::truncated_svd(AY, rank);
    A.U.insert(i, std::move(U));
  }
  for (int64_t j = 0; j < n_blocks; ++j) {
    Hatrix::Matrix YtA(rank + oversampling, block_size);
    for (int64_t i = 0; i < n_blocks; ++i) {
      if (j == i) continue;
      Hatrix::matmul(Y[i], A.D(i, j), YtA, true);
    }
    std::tie(U, S, V, error) = Hatrix::truncated_svd(YtA, rank);
    A.V.insert(j, std::move(V));
  }

  error = 0;
  for (int i = 0; i < n_blocks; ++i) {
    for (int j = 0; j < n_blocks; ++j) {
      if (i == j)
        continue;
      else {
        A.S.insert(i, j,
                   Hatrix::matmul(Hatrix::matmul(A.U[i], A.D(i, j), true),
                                  A.V[j], false, true));
        error += Hatrix::norm_diff(A.U[i] * A.S(i, j) * A.V[j], A.D(i, j));
      }
    }
  }
  std::cout << "Total construction error: " << error << "\n";
  return A;
}

void matmul_BLR(Hatrix::BLR& A, Hatrix::BLR& B, Hatrix::BLR& C,
		double alpha, double beta, int64_t n_blocks) {
  Hatrix::BLR C_check(C);
  for (int i = 0; i < n_blocks; i++) {
    for(int j = 0; j < n_blocks; j++) {
      if(i == j) {
	C.D(i, j) *= beta;
	Hatrix::Matrix S_ij(A.U[i].cols, B.V[j].rows);
	for(int k = 0; k < n_blocks; k++) {
	  if(i == k && j == k) {
	    Hatrix::matmul(A.D(k, k), B.D(k, k), C.D(i, j), false, false, alpha);
	  }
	  else {
	    S_ij += A.S(i, k) * (A.V[k] * B.U[k]) * B.S(k, j);
	  }
	  // Multiply dense part for error checking
	  Hatrix::matmul(A.D(i, k), B.D(k, j), C_check.D(i, j),
			 false, false, alpha, k == 0 ? beta : 1.0);
	}
	Hatrix::matmul(A.U[i], S_ij * B.V[j], C.D(i, j), false, false, alpha);
      }
      else {
	C.S(i, j) *= beta;
	for(int k = 0; k < n_blocks; k++) {
	  if(i == k && j != k) { //D x LR
	    Hatrix::matmul(Hatrix::matmul(C.U[i], A.D(i, k) * B.U[k], true),
			   B.S(k, j) * Hatrix::matmul(B.V[j], C.V[j], false, true),
			   C.S(i, j), false, false, alpha);
	  }
	  else if(i != k && j == k) { //LR x D
	    Hatrix::matmul(Hatrix::matmul(C.U[i], A.U[i], true),
			   A.S(i, k) * Hatrix::matmul(A.V[k] * B.D(k, j), C.V[j],
						      false, true),
			   C.S(i, j), false, false, alpha);
	  }
	  else { //LR x LR
	    Hatrix::matmul(Hatrix::matmul(C.U[i], A.U[i], true),
			   A.S(i, k) * (A.V[k] * B.U[k]) * B.S(k, j) *
			   Hatrix::matmul(B.V[j], C.V[j], false, true),
			   C.S(i, j), false, false, alpha);
	  }
	  // Multiply dense part for error checking
	  Hatrix::matmul(A.D(i, k), B.D(k, j), C_check.D(i, j),
			 false, false, alpha, k == 0 ? beta : 1.0);
	}
      }
    }
  }
  double norm = 0, diff = 0;
  for (int i = 0; i < n_blocks; ++i) {
    for (int j = 0; j < n_blocks; ++j) {
      norm += Hatrix::norm(C_check.D(i, j));
      if (i == j)
        diff += Hatrix::norm_diff(C.D(i, j), C_check.D(i, j));
      else {
        diff += Hatrix::norm_diff(C.U[i] * C.S(i, j) * C.V[j], C_check.D(i, j));
      }
    }
  }
  std::cout << "Approximate Matmul error: " << diff/norm << "\n";
}

Hatrix::BLR matmul_BLR_out(Hatrix::BLR& A, Hatrix::BLR& B,
			   double alpha, double beta, int64_t n_blocks) {
  //Generate C as zero matrices with row-bases of A and col-bases of B
  Hatrix::BLR C;
  for(int i = 0; i < n_blocks; i++) {
    C.U.insert(i, Hatrix::Matrix(A.U[i]));
    C.V.insert(i, Hatrix::Matrix(B.V[i]));
  }
  for(int i = 0; i < n_blocks; i++) {
    for(int j = 0; j < n_blocks; j++) {
      C.D.insert(i, j, Hatrix::Matrix(A.D(i, j).rows, B.D(i, j).cols));
      if(i != j) {
	C.S.insert(i, j, Hatrix::Matrix(C.U[i].cols, C.V[j].rows));
      }
    }
  }
  matmul_BLR(A, B, C, alpha, beta, n_blocks);
  return C;
}

int main() {
  int64_t block_size = 32;
  int64_t n_blocks = 4;
  int64_t rank = 8;
  Hatrix::BLR A = construct_BLR(block_size, n_blocks, rank);
  Hatrix::BLR B = construct_BLR(block_size, n_blocks, rank);

  Hatrix::BLR C = construct_BLR(block_size, n_blocks, rank);
  matmul_BLR(A, B, C, 1, 1, n_blocks);

  // Hatrix::BLR C = matmul_BLR_out(A, B, 1, 1, n_blocks);

  return 0;
}
