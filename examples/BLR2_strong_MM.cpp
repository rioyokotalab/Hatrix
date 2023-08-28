#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Hatrix/Hatrix.hpp"

std::vector<double> equally_spaced_vector(int64_t N, double minVal, double maxVal) {
  std::vector<double> res(N, 0.0);
  double rnge = maxVal - minVal;
  for(int64_t i=0; i<N; i++) {
    res[i] = minVal + ((double)i/(double)rnge);
  }
  return res;
}

Hatrix::BLR construct_BLR(int64_t block_size, int64_t n_blocks, int64_t rank, int64_t admis) {
  // Random points for laplace kernel
  std::vector<std::vector<double>> randpts;
  randpts.push_back(equally_spaced_vector(n_blocks * block_size, 0.0, 1.0));
  randpts.push_back(equally_spaced_vector(n_blocks * block_size, 0.0, 1.0)); //2D
  
  Hatrix::BLR A;
  for (int i = 0; i < n_blocks; ++i) {
    for (int j = 0; j < n_blocks; ++j) {
      A.is_admissible.insert(i, j, std::abs(i - j) > admis);
      A.D.insert(i, j,
		 Hatrix::generate_laplacend_matrix(randpts,
						   block_size, block_size,
						   i*block_size, j*block_size));
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
      if (A.is_admissible(i, j)) Hatrix::matmul(A.D(i, j), Y[j], AY);
    }
    std::tie(U, S, V, error) = Hatrix::truncated_svd(AY, rank);
    A.U.insert(i, std::move(U));
  }
  for (int64_t j = 0; j < n_blocks; ++j) {
    Hatrix::Matrix YtA(rank + oversampling, block_size);
    for (int64_t i = 0; i < n_blocks; ++i) {
      if (A.is_admissible(i, j)) Hatrix::matmul(Y[i], A.D(i, j), YtA, true);
    }
    std::tie(U, S, V, error) = Hatrix::truncated_svd(YtA, rank);
    A.V.insert(j, std::move(V));
  }
  for (int i = 0; i < n_blocks; ++i) {
    for (int j = 0; j < n_blocks; ++j) {
      if (A.is_admissible(i, j)) {
        A.S.insert(i, j,
                   Hatrix::matmul(Hatrix::matmul(A.U[i], A.D(i, j), true),
                                  A.V[j], false, true));
      }
    }
  }

  double diff = 0, norm = 0, fnorm, fdiff;
  for (int i = 0; i < n_blocks; ++i) {
    for (int j = 0; j < n_blocks; ++j) {
      fnorm = Hatrix::norm(A.D(i, j));
      norm += fnorm * fnorm;
      if (A.is_admissible(i, j)) {
	fdiff = Hatrix::norm(A.U[i] * A.S(i, j) * A.V[j] - A.D(i, j));
	diff += fdiff * fdiff;
      }
    }
  }
  std::cout << "BLR construction error (rel): " << std::sqrt(diff/norm) << "\n";
  return A;
}

void dense_matmul_BLR(Hatrix::BLR& A, Hatrix::BLR& B, Hatrix::BLR &C,
		      double alpha, double beta, int64_t n_blocks) {
  for (int i = 0; i < n_blocks; i++) {
    for(int j = 0; j < n_blocks; j++) {
      for(int k = 0; k < n_blocks; k++) {
	Hatrix::matmul(A.D(i, k), B.D(k, j), C.D(i, j),
		       false, false, alpha, k == 0 ? beta : 1.0);
      }
    }
  }
}

Hatrix::BLR dense_matmul_BLR(Hatrix::BLR& A, Hatrix::BLR& B, int64_t n_blocks) {
  Hatrix::BLR C;
  for(int i = 0; i < n_blocks; i++) {
    for(int j = 0; j < n_blocks; j++) {
      C.D.insert(i, j, Hatrix::Matrix(A.D(i, j).rows, B.D(i, j).cols));
    }
  }
  dense_matmul_BLR(A, B, C, 1, 0, n_blocks);
  return C;
}

void projected_matmul_BLR(Hatrix::BLR& A, Hatrix::BLR& B, Hatrix::BLR& C,
			  double alpha, double beta, int64_t n_blocks, int64_t admis) {
  //Precompute
  Hatrix::RowColMap<Hatrix::Matrix> UCTxAxUB, VAxBxVCT;
  std::vector<Hatrix::Matrix> UCTxUA, VBxVCT;
  std::vector<Hatrix::Matrix> VAxUB;
  for(int k = 0; k < n_blocks; k++) {
    for(int l = 0; l < n_blocks; l++) {
      if(!A.is_admissible(k, l)) {
	UCTxAxUB.insert(k, l, Hatrix::matmul(C.U[k], A.D(k, l) * B.U[l], true, false));
      }
      if(!B.is_admissible(l, k)) {
	VAxBxVCT.insert(l, k, Hatrix::matmul(A.V[l] * B.D(l, k), C.V[k], false, true));
      }
    }
    
    UCTxUA.push_back(Hatrix::matmul(C.U[k], A.U[k], true, false));
    VBxVCT.push_back(Hatrix::matmul(B.V[k], C.V[k], false, true));

    VAxUB.push_back(A.V[k] * B.U[k]);
  }
  for (int i = 0; i < n_blocks; i++) {
    for(int j = 0; j < n_blocks; j++) {
      if(!C.is_admissible(i, j)) { //Inadmissible C(i, j)
	C.D(i, j) *= beta;
	Hatrix::Matrix S_ij(C.U[i].cols, C.V[j].rows);
	for(int k = 0; k < n_blocks; k++) {
	  if((!A.is_admissible(i, k)) && (!B.is_admissible(k, j))) { //D x D
	    Hatrix::matmul(A.D(i, k), B.D(k, j), C.D(i, j),
			   false, false, alpha);
	  }
	  else if(!A.is_admissible(i, k)) { //D x LR
	    Hatrix::matmul(A.D(i, k) * B.U[k], B.S(k, j) * B.V[j], C.D(i, j),
			   false, false, alpha);
	  }
	  else if(!B.is_admissible(k, j)) { //LR x D
	    Hatrix::matmul(A.U[i] * A.S(i, k), A.V[k] * B.D(k, j), C.D(i, j),
			   false, false, alpha);
	  }
	  else { //LR x LR
	    S_ij += A.S(i, k) * VAxUB[k] * B.S(k, j);
	  }
	}
	Hatrix::matmul(C.U[i], S_ij * C.V[j], C.D(i, j), false, false, alpha);
      }
      else { //Admissible C(i, j)
	C.S(i, j) *= beta;
	for(int k = 0; k < n_blocks; k++) {
	  if((!A.is_admissible(i, k)) && (!B.is_admissible(k, j))) { //D x D
	    Hatrix::matmul(Hatrix::matmul(C.U[i], A.D(i, k), true, false),
			   Hatrix::matmul(B.D(k, j), C.V[j], false, true),
			   C.S(i, j), false, false, alpha);
	  }
	  else if(!A.is_admissible(i, k)) { //D x LR
	    Hatrix::matmul(UCTxAxUB(i, k),
			   B.S(k, j) * VBxVCT[j],
			   C.S(i, j), false, false, alpha);
	  }
	  else if(!B.is_admissible(k, j)) { //LR x D
	    Hatrix::matmul(UCTxUA[i],
			   A.S(i, k) * VAxBxVCT(k, j),
			   C.S(i, j), false, false, alpha);
	  }
	  else { //LR x LR
	    Hatrix::matmul(UCTxUA[i],
			   A.S(i, k) * VAxUB[k] * B.S(k, j) *
			   VBxVCT[j],
			   C.S(i, j), false, false, alpha);
	  }
	}
      }
    }
  }
}

Hatrix::BLR projected_matmul_BLR(Hatrix::BLR& A, Hatrix::BLR& B, int64_t n_blocks, int64_t admis) {
  //Generate C as zero matrices with row-bases of A and col-bases of B
  Hatrix::BLR C;
  for(int i = 0; i < n_blocks; i++) {
    C.U.insert(i, Hatrix::Matrix(A.U[i]));
    C.V.insert(i, Hatrix::Matrix(B.V[i]));
  }
  for(int i = 0; i < n_blocks; i++) {
    for(int j = 0; j < n_blocks; j++) {
      C.D.insert(i, j, Hatrix::Matrix(A.D(i, j).rows, B.D(i, j).cols));
      C.is_admissible.insert(i, j, std::abs(i - j) > admis);
      if(C.is_admissible(i, j)) {
	C.S.insert(i, j, Hatrix::Matrix(C.U[i].cols, C.V[j].rows));
      }
    }
  }
  projected_matmul_BLR(A, B, C, 1, 1, n_blocks, admis);
  return C;
}


int main(int argc, char** argv) {
  int64_t N = argc > 1 ? atoi(argv[1]) : 256;
  int64_t block_size = argc > 2 ? atoi(argv[2]) : 32;
  int64_t rank = argc > 3 ? atoi(argv[3]) : 8;
  int64_t admis = argc > 4 ? atoi(argv[4]) : 0;
  int64_t n_blocks = N/block_size;
  std::cout <<"N=" <<N <<",block_size=" <<block_size <<",rank=" <<rank <<",admis=" <<admis <<"\n";
  
  Hatrix::BLR A = construct_BLR(block_size, n_blocks, rank, admis);
  Hatrix::BLR B = construct_BLR(block_size, n_blocks, rank, admis);
  Hatrix::BLR C_dense = dense_matmul_BLR(A, B, n_blocks);
  Hatrix::BLR C_proj = projected_matmul_BLR(A, B, n_blocks, admis);

  //Projected Matmul Error
  double norm = 0, diff = 0, fnorm, fdiff;
  for (int i = 0; i < n_blocks; ++i) {
    for (int j = 0; j < n_blocks; ++j) {
      fnorm = Hatrix::norm(C_dense.D(i, j));
      norm += fnorm * fnorm;
      if (C_proj.is_admissible(i, j)) {
	fdiff = Hatrix::norm(C_proj.U[i] * C_proj.S(i, j) * C_proj.V[j] - C_dense.D(i, j));
      }
      else {
	fdiff = Hatrix::norm(C_proj.D(i, j) - C_dense.D(i, j));
      }
      diff += fdiff * fdiff;
    }
  }
  std::cout << "Projected Matmul Relative Error: " << std::sqrt(diff/norm) << "\n";
  
  return 0;
}
