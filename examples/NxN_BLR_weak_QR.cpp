#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Hatrix/Hatrix.h"

using RowColMap = Hatrix::RowColMap<Hatrix::Matrix>;

std::vector<double> equally_spaced_vector(int64_t N, double minVal, double maxVal) {
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
  randpts.push_back(equally_spaced_vector(n_blocks * block_size, 0.0, 1.0));
  randpts.push_back(equally_spaced_vector(n_blocks * block_size, 0.0, 1.0)); //2D
  
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
  std::cout << "BLR2 construction error (rel): " << std::sqrt(diff/norm) << "\n";
  return A;
}

void triangularize_block_column(int64_t k, Hatrix::BLR& A,
				Hatrix::BLR& Y, Hatrix::BLR& T, int64_t n_blocks) {
  Hatrix::Matrix Ak(A.D(k, k).rows + (n_blocks-k-1) * A.V[k].rows, A.D(k, k).cols);
  std::vector<int64_t> split_indices{A.D(k, k).rows};
  for(int64_t i = k+1; i < (n_blocks-1); i++) {
    split_indices.push_back(split_indices[i-k-1] + A.V[k].rows);
  }
  std::vector<Hatrix::Matrix> Ak_parts = Ak.split(split_indices,
						  std::vector<int64_t>{}, false);
  Ak_parts[0] = A.D(k, k);
  for(int64_t i = k+1; i < n_blocks; i++) {
    Hatrix::Matrix SV = A.S(i, k) * A.V[k];
    Ak_parts[i-k] = SV;
    A.S(i, k) = 0;
  }
  T.D.insert(k, k, Hatrix::Matrix(Ak.cols, Ak.cols));
  Hatrix::householder_qr_compact_wy(Ak, T.D(k, k));
  A.D(k, k) = upper_tri(Ak_parts[0]);
  Y.D.insert(k, k, lower_tri(Ak_parts[0], true));
  for(int64_t i = k+1; i < n_blocks; i++) {
    Hatrix::Matrix Yik(Ak_parts[i-k].rows, Ak_parts[i-k].cols);
    Yik = Ak_parts[i-k];
    Y.S.insert(i, k, std::move(Yik));
  }
}

RowColMap compute_YTY(int64_t k, bool transT, const Hatrix::BLR& Y, const Hatrix::BLR& T,
		      int64_t n_blocks) {
  RowColMap YTY;
  for(int64_t i = k; i < n_blocks; i++) {
    Hatrix::Matrix YT = Hatrix::triangular_matmul_out(T.D(k, k),
						      i == k ? Hatrix::matmul(Y.U[k], Y.D(k, k),
									      true) : Y.S(i, k),
						      Hatrix::Right, Hatrix::Upper,
						      transT, false);
    for(int64_t j = k; j < n_blocks; j++) {
      YTY.insert(i, j, Hatrix::matmul(YT, j == k ? transpose(Y.D(k, k)) :
				      Hatrix::matmul(Y.S(j, k), Y.U[j], true, true)));
    }
  }
  return YTY;
}

void apply_block_column_reflector(int64_t k, int64_t j, bool trans,
				  Hatrix::BLR& C, const Hatrix::BLR& Y, const Hatrix::BLR& T,
				  const RowColMap& YTY, int64_t n_blocks) {
  //Precompute for applying onto j-th block-column
  Hatrix::Matrix YT_kk = Hatrix::triangular_matmul_out(T.D(k, k), Y.D(k, k),
						       Hatrix::Right, Hatrix::Upper,
						       trans, false);
  Hatrix::Matrix YTY_kk = Hatrix::matmul(YT_kk, Y.D(k, k), false, true);
  Hatrix::RowMap tmp;
  for(int64_t i = k; i < n_blocks; i++) {
    if(i == j) {
      tmp.insert(i, Hatrix::matmul(C.D(i, j), C.V[j], false, true));
    }
    else {
      tmp.insert(i, Hatrix::matmul(C.U[i], C.S(i, j)));
    }
  }
  //Multiply
  for(int64_t i = k; i < n_blocks; i++) {
    if(i == j) { //Inadmissible C(i, j)
      Hatrix::Matrix Cij(C.D(i, j));
      Hatrix::Matrix Sij(i == k ? YT_kk.cols : Y.U[i].cols, C.V[j].rows);
      for(int64_t l = k; l < n_blocks; l++) {
	if(i == l && l == j) {
	  Hatrix::matmul(i == k ? YTY_kk : Y.U[i] * YTY(i, l),
			 Cij, C.D(i, j), false, false, -1, 1);
	}
	else {
	  Hatrix::matmul(i == k ? Hatrix::matmul(Y.S(l, k), Y.U[l], true, true) : YTY(i, l),
			 tmp[l], Sij, false, false, 1, 1);
	}
      }
      Hatrix::matmul(i == k ? YT_kk : Y.U[i], Sij * C.V[j],
		     C.D(i, j), false, false, -1, 1);
    }
    else { //Admissible C(i, j)
      for(int64_t l = k; l < n_blocks; l++) {
	Hatrix::matmul(YTY(i, l), tmp[l], C.S(i, j), false, false, -1, 1);
      }
    }
  }
}
				
std::tuple<Hatrix::BLR, Hatrix::BLR, Hatrix::BLR> qr_BLR(const Hatrix::BLR& A,
							 int64_t n_blocks) {
  Hatrix::BLR Y, T;
  Hatrix::BLR R(A);

  for(int64_t k = 0; k < n_blocks; k++) {
    Y.U.insert(k, Hatrix::Matrix(A.U[k]));
  }
  for(int64_t k = 0; k < n_blocks; k++) {
    triangularize_block_column(k, R, Y, T, n_blocks);
    //Precompute for applying k-th block column reflector
    RowColMap YTY = compute_YTY(k, true, Y, T, n_blocks);
    for(int64_t j = k+1; j < n_blocks; j++) {
      apply_block_column_reflector(k, j, true, R, Y, T, YTY, n_blocks);
    }
  }
  return {Y, T, R};
}

Hatrix::BLR left_multiply_Q(const Hatrix::BLR& Y, const Hatrix::BLR& T, bool trans,
			    const Hatrix::BLR& A, int64_t n_blocks) {
  Hatrix::BLR C;
  for(int64_t i = 0; i < n_blocks; i++) {
    C.U.insert(i, Hatrix::Matrix(Y.U[i]));
    C.V.insert(i, Hatrix::Matrix(A.V[i]));
  }
  for(int64_t i = 0; i < n_blocks; i++) {
    for(int64_t j = 0; j < n_blocks; j++) {
      if(i == j) C.D.insert(i, j, Hatrix::Matrix(A.D(i, j)));
      else C.S.insert(i, j, Hatrix::Matrix(A.S(i, j)));
    }
  }
  if(trans) {
    for(int64_t k = 0; k < n_blocks; k++) {
      RowColMap YTY = compute_YTY(k, trans, Y, T, n_blocks);
      for(int64_t j = k; j < n_blocks; j++) {
	apply_block_column_reflector(k, j, trans, C, Y, T, YTY, n_blocks);
      }
    }
  }
  else {
    for(int64_t k = n_blocks-1; k >= 0; k--) {
      RowColMap YTY = compute_YTY(k, trans, Y, T, n_blocks);
      for(int64_t j = k; j < n_blocks; j++) {
	apply_block_column_reflector(k, j, trans, C, Y, T, YTY, n_blocks);
      }
    }
  }
  return C;
}

int main(int argc, char** argv) {
  Hatrix::Context::init();
  int64_t N = argc > 1 ? atoi(argv[1]) : 256;
  int64_t block_size = argc > 2 ? atoi(argv[2]) : 32;
  int64_t rank = argc > 3 ? atoi(argv[3]) : 8;
  int64_t n_blocks = N/block_size;
  
  Hatrix::BLR A = construct_BLR(block_size, n_blocks, rank);
  Hatrix::BLR Y, T, R;
  std::tie(Y, T, R) = qr_BLR(A, n_blocks);
  
  // Approximate QR Error
  Hatrix::BLR QR = left_multiply_Q(Y, T, false, R, n_blocks);
  double norm = 0, diff = 0, fnorm, fdiff;
  for (int64_t i = 0; i < n_blocks; i++) {
    for (int64_t j = 0; j < n_blocks; j++) {
      if (i == j) {
	fnorm = Hatrix::norm(A.D(i, j));
	fdiff = Hatrix::norm(QR.D(i, j) - A.D(i, j));
      } else {
	fnorm = Hatrix::norm(A.U[i] * A.S(i, j) * A.V[j]);
        fdiff = Hatrix::norm(QR.U[i] * QR.S(i, j) * QR.V[j] -
			     A.U[i] * A.S(i, j) * A.V[j]);
      }
      norm += fnorm * fnorm;
      diff += fdiff * fdiff;
    }
  }
  std::cout << "BLR2-QR Relative Error: " << std::sqrt(diff/norm) << "\n";

  //Q orthogonality
  //Construct Q
  Hatrix::BLR I;
  for(int64_t i = 0; i < n_blocks; i++) {
    I.U.insert(i, Hatrix::Matrix(Y.U[i]));
    I.V.insert(i, transpose(Y.U[i]));
  }
  for(int64_t i = 0; i < n_blocks; i++) {
    for(int64_t j = 0; j < n_blocks; j++) {
      if(i == j) {
	I.D.insert(i, j, Hatrix::generate_identity_matrix(Y.D(i, j).rows,
							  Y.D(i, j).cols));
      }
      else {
	I.S.insert(i, j, Hatrix::Matrix(I.U[i].cols, I.V[j].rows));
      }
    }
  }
  Hatrix::BLR Q = left_multiply_Q(Y, T, false, I, n_blocks);
  Hatrix::BLR QtQ = left_multiply_Q(Y, T, true, Q, n_blocks);
  for (int64_t i = 0; i < n_blocks; i++) {
    for (int64_t j = 0; j < n_blocks; j++) {
      if (i == j) {
	fdiff = Hatrix::norm(QtQ.D(i, j) - I.D(i, j));
      } else {
        fdiff = Hatrix::norm(QtQ.U[i] * QtQ.S(i, j) * QtQ.V[j]);
      }
      diff += fdiff * fdiff;
    }
  }
  std::cout << "BLR2-QR Orthogonality: " << std::sqrt(diff/N) << "\n";
  
  Hatrix::Context::finalize();
  return 0;
}
