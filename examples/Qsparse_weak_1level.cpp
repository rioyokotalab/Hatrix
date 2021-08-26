#include <iostream>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <cmath>
#include <algorithm>

#ifdef USE_MKL
#include "mkl_cblas.h"
#include "mkl_lapacke.h"
#else
#include "cblas.h"
#include "lapacke.h"
#endif

// TODO: The V here is stored as a horizontal matrix. Should we be storing both
// U and V as (b x rank) matrices? Right now I manually transpose.

#include "Hatrix/Hatrix.h"

using randvec_t = std::vector<std::vector<double> >;

std::vector<double> equally_spaced_vector(int N, double minVal, double maxVal) {
  std::vector<double> res(N, 0.0);
  double rnge = maxVal - minVal;
  for(int i=0; i<N; i++) {
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
      if (std::abs(i - j) <= admis) continue;
      Hatrix::matmul(A.D(i, j), Y[j], AY);
    }
    std::tie(U, S, V, error) = Hatrix::truncated_svd(AY, rank);
    A.U.insert(i, std::move(U));
  }
  for (int64_t j = 0; j < n_blocks; ++j) {
    Hatrix::Matrix YtA(rank + oversampling, block_size);
    for (int64_t i = 0; i < n_blocks; ++i) {
      if (std::abs(i - j) <= admis) continue;
      Hatrix::matmul(Y[i], A.D(i, j), YtA, true);
    }
    std::tie(U, S, V, error) = Hatrix::truncated_svd(YtA, rank);
    A.V.insert(j, std::move(V.transpose()));
  }
  for (int i = 0; i < n_blocks; ++i) {
    for (int j = 0; j < n_blocks; ++j) {
     if (std::abs(i - j) > admis) {
        A.S.insert(i, j,
                   Hatrix::matmul(Hatrix::matmul(A.U[i], A.D(i, j), true),
                                  A.V[j], false, false));
      }
    }
  }

  double diff = 0, norm = 0, fnorm, fdiff;
  for (int i = 0; i < n_blocks; ++i) {
    for (int j = 0; j < n_blocks; ++j) {
      fnorm = Hatrix::norm(A.D(i, j));
      norm += fnorm * fnorm;
      if (std::abs(i - j) <= admis)
        continue;
      else {
	fdiff = Hatrix::norm(A.U[i] * A.S(i, j) * A.V[j].transpose() - A.D(i, j));
	diff += fdiff * fdiff;
      }
    }
  }
  std::cout << "BLR construction error (rel): " << std::sqrt(diff/norm) << "\n";
  return A;
}


Hatrix::Matrix full_qr(Hatrix::Matrix& A) {
  Hatrix::Matrix Q(A.rows, A.rows);
  std::vector<double> tau(std::max(A.rows, A.cols));

  LAPACKE_dgeqrf(LAPACK_COL_MAJOR, A.rows, A.cols, &A, A.stride, tau.data());

  for (int64_t i = 0; i < Q.rows; ++i) {
    Q(i, i) = 1.0;
    for (int j = 0; j < std::min(i, A.cols); ++j) {
      Q(i, j) = A(i, j);
    }
  }

  LAPACKE_dorgqr(LAPACK_COL_MAJOR, Q.rows, Q.cols, Q.cols, &Q,
    Q.stride, tau.data());

  return Q;
}

Hatrix::Matrix make_complement(Hatrix::Matrix& Q) {
  Hatrix::Matrix Q_F(Q.rows, Q.rows);
  Hatrix::Matrix Q_full = full_qr(Q);

  for (int i = 0; i < Q_F.rows; ++i) {
    for (int j = 0; j < Q_F.cols - Q.cols; ++j) {
      Q_F(i, j) = Q_full(i, j + Q.cols);
    }
  }

  for (int i = 0; i < Q_F.rows; ++i) {
    for (int j = 0; j < Q.cols; ++j) {
      Q_F(i, j + (Q_F.cols - Q.cols)) = Q_full(i, j);
    }
  }

  return Q_F;
}

void left_and_right_multiply_dense_block(const Hatrix::Matrix& U_F,
  const Hatrix::Matrix& V_F, Hatrix::Matrix& D) {
  Hatrix::Matrix temp(U_F.cols, D.cols);
  Hatrix::matmul(U_F, D, temp, true, false, 1.0, 0.0);
  Hatrix::matmul(temp, V_F, D, false, false, 1.0, 0.0);
}

void partial_lu(Hatrix::Matrix& D, int rank) {
  int c = D.rows - rank;
  const double alpha = 1.0;
  double * upper_left = &D;
  double * lower_left = upper_left + c;
  double * upper_right = upper_left + c * D.stride;
  double * lower_right = upper_left + c * D.stride + c;

  std::vector<int> ipiv(c);

  LAPACKE_dgetrf(LAPACK_COL_MAJOR, c, c, upper_left, D.stride, ipiv.data());

  cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
              c, rank, alpha, upper_left, D.stride, upper_right, D.stride);

  cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
              rank, c, alpha, upper_left, D.stride, lower_left, D.stride);

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, rank, rank, c, -1.0,
              lower_left, D.stride, upper_right, D.stride, -1.0, lower_right, D.stride);
}

Hatrix::Matrix merge_null_spaces(Hatrix::BLR& A, int nblocks, int rank) {
  Hatrix::Matrix M(rank * nblocks, rank * nblocks);

  for (int i = 0; i < nblocks; ++i) {
    for (int j = 0; j < nblocks; ++j) {
      if (i == j) {
        for (int irow = 0; irow < rank; ++irow) {
          for (int icol = 0; icol < rank; ++icol) {
            int c = A.D(i, j).rows - rank;
            M(i * rank + irow, j * rank + icol) = A.D(i, j)(c + irow, c + icol);
          }
        }
      }
      else {
        for (int irow = 0; irow < rank; ++irow) {
          for (int icol = 0; icol < rank; ++icol) {
            M(i * rank + irow, i * rank + icol) = A.S(i, j)(irow, icol);
          }
        }
      }
    }
  }

  return M;
}

void qsparse_factorize(Hatrix::BLR& A, int N, int nblocks, int rank) {
  for (int node = 0; node < nblocks; ++node) {
    Hatrix::Matrix U_F = make_complement(A.U[node]);
    Hatrix::Matrix V_F = make_complement(A.V[node]);

    left_and_right_multiply_dense_block(U_F, V_F, A.D(node, node));

    partial_lu(A.D(node, node), rank);
  }

  Hatrix::Matrix M = merge_null_spaces(A, nblocks, rank);

}

Hatrix::Matrix qsparse_substitute(Hatrix::BLR& A, Hatrix::Matrix& b) {
  Hatrix::Matrix x(b.rows, 1);

  return x;
}

int main(int argc, char *argv[]) {
  int N = atoi(argv[1]);
  int rank = atoi(argv[2]);
  int block_size = atoi(argv[3]);
  int nblocks = N / block_size;

  Hatrix::Context::init();
  Hatrix::Matrix b = Hatrix::generate_random_matrix(N, 1);
  Hatrix::BLR A = construct_BLR(block_size, nblocks, rank, 0);
  qsparse_factorize(A, N, nblocks, rank);
  //Hatrix::Matrix x = qsparse_substitute(A, b);

  Hatrix::Context::finalize();
}
