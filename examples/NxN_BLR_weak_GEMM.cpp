#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Hatrix/Hatrix.h"

std::vector<double> equallySpacedVector(int64_t N, double minVal, double maxVal) {
  std::vector<double> res(N, 0.0);
  double rnge = maxVal - minVal;
  for(int64_t i=0; i<N; i++) {
    res[i] = minVal + ((double)i/(double)rnge);
  }
  return res;
}

Hatrix::BLR construct_BLR(int64_t block_size, int64_t n_blocks, int64_t rank) {
  // Random points for laplace kernel
  std::vector<std::vector<double>> randpts;
  randpts.push_back(equallySpacedVector(n_blocks * block_size, 0.0, 1.0)); //1D
  
  Hatrix::BLR A;
  for (int i = 0; i < n_blocks; ++i) {
    for (int j = 0; j < n_blocks; ++j) {      
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
  for (int i = 0; i < n_blocks; ++i) {
    for (int j = 0; j < n_blocks; ++j) {
      if (i != j) {
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
      if (i == j)
        continue;
      else {
	fdiff = Hatrix::norm(A.U[i] * A.S(i, j) * A.V[j] - A.D(i, j));
	diff += fdiff * fdiff;
      }
    }
  }
  std::cout << "BLR construction error (rel): " << std::sqrt(diff/norm) << "\n";
  return A;
}

void recompress_BLR(Hatrix::BLR& A, double rank, int64_t n_blocks) {
  //Recompress U bases
  for(int i = 0; i < n_blocks; i++) {
    Hatrix::Matrix u(A.U[i].rows, A.U[i].min_dim());
    Hatrix::Matrix s(A.U[i].min_dim(), A.U[i].min_dim());
    Hatrix::Matrix v(A.U[i].min_dim(), A.U[i].cols);
    Hatrix::truncated_svd(A.U[i], u, s, v, rank);
    A.U[i] = std::move(u);
    //Multiply to the left of coupling matrices
    for(int j = 0; j < n_blocks; j++) {
      if(i != j) {
	A.S(i, j) = s * v * A.S(i, j);
      }
    }
  }
  //Recompress V bases
  for(int j = 0; j < n_blocks; j++) {
    Hatrix::Matrix u(A.V[j].rows, A.V[j].min_dim());
    Hatrix::Matrix s(A.V[j].min_dim(), A.V[j].min_dim());
    Hatrix::Matrix v(A.V[j].min_dim(), A.V[j].cols);
    Hatrix::truncated_svd(A.V[j], u, s, v, rank);
    A.V[j] = std::move(v);
    //Multiply to the right of coupling matrices
    for(int i = 0; i < n_blocks; i++) {
      if(i != j) {
	A.S(i, j) = A.S(i, j) * u * s;
      }
    }
  }
}

void projected_matmul_BLR(Hatrix::BLR& A, Hatrix::BLR& B, Hatrix::BLR& C,
		double alpha, double beta, int64_t n_blocks) {
  Hatrix::BLR C_check(C);
  for (int i = 0; i < n_blocks; i++) {
    for(int j = 0; j < n_blocks; j++) {
      if(i == j) {
	C.D(i, j) *= beta;
	Hatrix::Matrix S_ij(C.U[i].cols, C.V[j].rows);
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
	Hatrix::matmul(C.U[i], S_ij * C.V[j], C.D(i, j), false, false, alpha);
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
  double norm = 0, diff = 0, fnorm, fdiff;
  for (int i = 0; i < n_blocks; ++i) {
    for (int j = 0; j < n_blocks; ++j) {
      fnorm = Hatrix::norm(C_check.D(i, j));
      norm += fnorm * fnorm;
      if (i == j) {
	fdiff = Hatrix::norm(C.D(i, j) - C_check.D(i, j));
      } else {
        fdiff = Hatrix::norm(C.U[i] * C.S(i, j) * C.V[j] - C_check.D(i, j));
      }
      diff += fdiff * fdiff;
    }
  }
  std::cout << "Projected matmul error (Rel): " <<std::sqrt(diff/norm) <<"\n";
}

Hatrix::BLR projected_matmul_BLR(Hatrix::BLR& A, Hatrix::BLR& B, int64_t n_blocks) {
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
  projected_matmul_BLR(A, B, C, 1, 1, n_blocks);
  return C;
}

Hatrix::BLR exact_matmul_BLR(Hatrix::BLR& A, Hatrix::BLR& B, int64_t n_blocks) {
  int row_start, col_start;
  Hatrix::BLR M, tmp;
  double rankA, rankB; //Assume uniform rank
  
  //Construct induced product bases
  rankA = A.U[0].cols;
  rankB = B.U[0].cols;
  //U and V bases
  for(int k = 0; k < n_blocks; k++) {
    M.U.insert(k, Hatrix::Matrix(A.U[k].rows, rankA + rankB));
    tmp.U.insert(k, A.D(k, k) * B.U[k]);
    for(int j = 0; j < M.U[k].cols; j++) {
      for(int i = 0; i < M.U[k].rows; i++) {
	M.U[k](i, j) = (j < rankA ? A.U[k](i, j) : tmp.U[k](i, j-rankA));
      }
    }
    M.V.insert(k, Hatrix::Matrix(rankA + rankB, B.V[k].cols));
    tmp.V.insert(k, A.V[k] * B.D(k, k));
    for(int j = 0; j < M.V[k].cols; j++) {
      for(int i = 0; i < M.V[k].rows; i++) {
	M.V[k](i, j) = (i < rankB ? B.V[k](i, j) : tmp.V[k](i-rankB, j));
      }
    }
  }
  //Coupling matrices and inadmissible blocks
  for(int i = 0; i < n_blocks; i++) {
    for(int j = 0; j < n_blocks; j++) {
      if(i == j) M.D.insert(i, j, Hatrix::Matrix(A.D(i, i).rows, B.D(i, i).cols));
      else M.S.insert(i, j, Hatrix::Matrix(M.U[i].cols, M.V[j].rows));
    }
  }

  //For error checking
  Hatrix::BLR M_check(M);
  for(int i = 0; i < n_blocks; i++) {
    for(int j = 0; j < n_blocks; j++) {
      if(i != j) {
	M_check.D.insert(i, j,
			 Hatrix::Matrix(M_check.U[i].rows, M_check.V[j].cols));
      }
    }
  }

  for (int i = 0; i < n_blocks; i++) {
    for(int j = 0; j < n_blocks; j++) {
      if(i == j) { //M(i, j) is non-admissible
	Hatrix::Matrix S_ij(A.U[i].cols, B.V[j].rows);
	for(int k = 0; k < n_blocks; k++) {
	  if(i == k && j == k) {
	    Hatrix::matmul(A.D(k, k), B.D(k, k), M.D(i, j));
	  }
	  else {
	    S_ij += A.S(i, k) * (A.V[k] * B.U[k]) * B.S(k, j);
	  }
	  // Multiply dense part for error checking
	  Hatrix::matmul(A.D(i, k), B.D(k, j), M_check.D(i, j));
	}
	Hatrix::matmul(A.U[i], S_ij * B.V[j], M.D(i, j));
      }
      else { //M(i, j) is admissible
	Hatrix::Matrix S_ij(rankA, rankB);
	for(int k = 0; k < n_blocks; k++) {
	  if(i == k && j != k) { //D x LR
	    //Fill bottom left of M.S with B.S
	    row_start = rankA; col_start = 0;
	    for(int q = 0; q < B.S(i, j).cols; q++) {
	      for(int p = 0; p < B.S(i, j).rows; p++) {
		M.S(i, j)(row_start+p, col_start+q) = B.S(i, j)(p, q);
	      }
	    }
	  }
	  else if(i != k && j == k) { //LR x D
	    //Fill top right of M.S with A.S
	    row_start = 0; col_start = rankB;
	    for(int q = 0; q < A.S(i, j).cols; q++) {
	      for(int p = 0; p < A.S(i, j).rows; p++) {
		M.S(i, j)(row_start+p, col_start+q) = A.S(i, j)(p, q);
	      }
	    }
	  }
	  else { //LR x LR
	    //Accumulate S
	    Hatrix::matmul(A.S(i, k),
			   (A.V[k] * B.U[k]) * B.S(k, j),
			   S_ij);
	  }
	  // Multiply dense part for error checking
	  Hatrix::matmul(A.D(i, k), B.D(k, j), M_check.D(i, j));
	}
	//Fill topleft of M.S
	row_start = 0; col_start = 0;
	for(int q = 0; q < S_ij.cols; q++) {
	  for(int p = 0; p < S_ij.rows; p++) {
	    M.S(i, j)(row_start+p, col_start+q) = S_ij(p, q);
	  }
	}
      }
    }
  }
  double norm = 0, diff = 0, fnorm, fdiff;
  for (int i = 0; i < n_blocks; ++i) {
    for (int j = 0; j < n_blocks; ++j) {
      fnorm = Hatrix::norm(M_check.D(i, j));
      norm += fnorm * fnorm;
      if (i == j) {
	fdiff = Hatrix::norm(M.D(i, j) - M_check.D(i, j));
      } else {
        fdiff = Hatrix::norm(M.U[i] * M.S(i, j) * M.V[j] - M_check.D(i, j));
      }
      diff += fdiff * fdiff;
    }
  }
  std::cout << "Exact matmul error before recompression (Rel): " <<std::sqrt(diff/norm) <<"\n";
  recompress_BLR(M, rankA, n_blocks);
  norm = 0, diff = 0;
  for (int i = 0; i < n_blocks; ++i) {
    for (int j = 0; j < n_blocks; ++j) {
      fnorm = Hatrix::norm(M_check.D(i, j));
      norm += fnorm * fnorm;
      if (i == j) {
	fdiff = Hatrix::norm(M.D(i, j) - M_check.D(i, j));
      } else {
        fdiff = Hatrix::norm(M.U[i] * M.S(i, j) * M.V[j] - M_check.D(i, j));
      }
      diff += fdiff * fdiff;
    }
  }
  std::cout << "Exact matmul error after recompression (Rel): " <<std::sqrt(diff/norm) <<"\n";
  return M;
}

int main(int argc, char** argv) {
  int64_t N = argc > 1 ? atoi(argv[1]) : 256;
  int64_t block_size = argc > 2 ? atoi(argv[2]) : 32;
  int64_t rank = argc > 3 ? atoi(argv[3]) : 8;
  int64_t n_blocks = N/block_size;
  
  Hatrix::BLR A = construct_BLR(block_size, n_blocks, rank);
  Hatrix::BLR B = construct_BLR(block_size, n_blocks, rank);
  Hatrix::BLR C = projected_matmul_BLR(A, B, n_blocks);
  Hatrix::BLR M = exact_matmul_BLR(A, B, n_blocks);
  
  return 0;
}
