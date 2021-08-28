#include <iostream>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <cmath>
#include <algorithm>
#include <cassert>

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

Hatrix::BLR construct_BLR(randvec_t& randpts, int64_t block_size, int64_t n_blocks, int64_t rank, int64_t admis) {
  // Random points for laplace kernel

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

Hatrix::Matrix make_complement(const Hatrix::Matrix& Q) {
  Hatrix::Matrix Q_copy(Q);
  Hatrix::Matrix Q_F(Q.rows, Q.rows);
  Hatrix::Matrix Q_full = full_qr(Q_copy);

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

Hatrix::Matrix left_and_right_multiply_dense_block(const Hatrix::Matrix& U_F,
  const Hatrix::Matrix& V_F, const Hatrix::Matrix& D) {
  Hatrix::Matrix result(D);
  Hatrix::Matrix temp(U_F.cols, D.cols);
  Hatrix::matmul(U_F, D, temp, true, false, 1.0, 0.0);
  Hatrix::matmul(temp, V_F, result, false, false, 1.0, 0.0);

  return result;
}

void partial_lu(Hatrix::Matrix& D, int rank) {
  int c = D.rows - rank;
  double * upper_left = &D;
  double * lower_left = upper_left + c;
  double * upper_right = upper_left + c * D.stride;
  double * lower_right = upper_left + c * D.stride + c;

  std::vector<int> ipiv(c);

  int info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, c, c, upper_left, D.stride, ipiv.data());

  if (info != 0) {
    std::cout << "GETRF ERROR: " << info << std::endl;
  }

  cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
              c, rank, 1.0, upper_left, D.stride, upper_right, D.stride);

  cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
              rank, c, 1.0, upper_left, D.stride, lower_left, D.stride);

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, rank, rank, c, -1.0,
              lower_left, D.stride, upper_right, D.stride, 1.0, lower_right, D.stride);
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
            M(i * rank + irow, j * rank + icol) = A.S(i, j)(irow, icol);
          }
        }
      }
    }
  }

  return M;
}

#ifdef VERIFY

void verify_complement_generation(const Hatrix::Matrix& Q_F, const Hatrix::Matrix& Q) {
  Hatrix::Matrix result = Hatrix::matmul(Q_F, Q, true, false, 1.0);
  result.print();
}

void verify_multiplication(const Hatrix::Matrix& prod, const Hatrix::Matrix& A,
  const Hatrix::Matrix& U_F, const Hatrix::Matrix& V_F) {
  Hatrix::Matrix temp = Hatrix::matmul(U_F, prod, false, false, 1.0);
  Hatrix::Matrix result = Hatrix::matmul(temp, V_F, false, true, 1.0);

  assert(Hatrix::norm(result) - Hatrix::norm(A) < 0.001);
}
#endif

Hatrix::Matrix qsparse_factorize(Hatrix::BLR& A, int N, int nblocks, int rank) {
  for (int node = 0; node < nblocks; ++node) {
    Hatrix::Matrix U_F = make_complement(A.U[node]);
    Hatrix::Matrix V_F = make_complement(A.V[node]);

#ifdef VERIFY
    verify_complement_generation(U_F, A.U[node]);
    verify_complement_generation(V_F, A.V[node]);
#endif

    Hatrix::Matrix prod = left_and_right_multiply_dense_block(U_F, V_F, A.D(node, node));

#ifdef VERIFY
  verify_multiplication(prod, A.D(node, node), U_F, V_F);
#endif

    A.D(node, node) = prod;

    partial_lu(A.D(node, node), rank);
  }

  Hatrix::Matrix M = merge_null_spaces(A, nblocks, rank);
  Hatrix::lu(M);

  return M;
}

void permute_forward(Hatrix::Matrix& x, int rank, int nblocks, int block_size) {
  Hatrix::Matrix x_copy(x.rows, x.cols);
  int c = block_size - rank;
  int offset = c * nblocks;

  for (int block = 0; block < nblocks; ++block) {
    // Copy the c part to the top of the copy vector.
    for (int i = 0; i < c; ++i) {
      x_copy(block * c + i, 0) = x(block_size * block + i, 0);
    }

    // Copy the rank part to the bottom of the copy vector
    for (int i = 0; i < rank; ++i) {
      x_copy(offset + block * rank + i, 0) = x(block_size * block + c + i, 0);
    }
  }

  x = std::move(x_copy);
}

void permute_backward(Hatrix::Matrix& x, int rank, int nblocks, int block_size) {
  Hatrix::Matrix x_copy(x.rows, x.cols);
  int c = block_size - rank;
  int offset = c * nblocks;

  for (int block = 0; block < nblocks; ++block) {
    // Copy the c part from the top of the original vector.
    for (int i = 0; i < c; ++i) {
      x_copy(block * block_size + i, 0) = x(block * c + i, 0);
    }

    // Copy the rank part from the bottom of the original vector.
    for (int i = 0; i < rank; ++i) {
      x_copy(block_size * block + c + i, 0) = x(offset + block * rank + i, 0);
    }
  }

  x = std::move(x_copy);
}

Hatrix::Matrix qsparse_substitute(Hatrix::BLR& A, Hatrix::Matrix& last_lu, const Hatrix::Matrix& b,
  int nblocks, int block_size, int rank) {
  Hatrix::Matrix x(b);

  // Forward substitution.
  for (int node = 0; node < nblocks; ++node) {
    Hatrix::Matrix U_F = make_complement(A.U[node]);
    Hatrix::Matrix& D = A.D(node, node);
    double * x_temp = &x + node * block_size;

    std::vector<double> result(block_size);
    cblas_dgemv(CblasColMajor, CblasTrans, U_F.cols, U_F.rows, 1.0, &U_F, U_F.stride,
      x_temp, 1, 0.0, result.data(), 1);

    for (int i = 0; i < result.size(); ++i) {
      x_temp[i] = result[i];
    }

    int c = block_size - rank;
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
      c, 1, 1.0, &D, D.stride, x_temp, x.stride);

    cblas_dgemv(CblasColMajor, CblasNoTrans, rank, c, -1.0, &D + c, D.stride,
      x_temp, 1, 1.0, x_temp + c, 1);
  }

  int c = block_size - rank;

  permute_forward(x, rank, nblocks, block_size);
  cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
    last_lu.rows, 1, 1.0, &last_lu, last_lu.stride, &x + c * nblocks, x.stride);

  cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
    last_lu.rows, 1, 1.0, &last_lu, last_lu.stride, &x + c * nblocks, x.stride);
  permute_backward(x, rank, nblocks, block_size);

  for (int node = 0; node < nblocks; ++node) {
    Hatrix::Matrix& D = A.D(node, node);
    double *x_temp = &x + node * block_size;
    int c = D.rows - rank;

    // // Perform upper trinagular TRSM on a piece of the vector.
    cblas_dgemv(CblasColMajor, CblasNoTrans, c, rank, -1.0,
                 &D + c * D.stride, D.stride, x_temp + c,
                 1, 1.0, x_temp, 1);

    cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper,
                 CblasNoTrans, CblasNonUnit,
                c, 1, 1.0, &D, D.stride, x_temp, x.stride);

    Hatrix::Matrix V_F = make_complement(A.V[node]);
    std::vector<double> result(block_size);
    cblas_dgemv(CblasColMajor, CblasNoTrans, V_F.rows, V_F.cols, 1.0,
                &V_F, V_F.stride, x_temp, 1, 0.0, result.data(), 1);

    for (int i = 0; i < result.size(); ++i) {
      x_temp[i] = result[i];
    }
  }

  return x;
}

int main(int argc, char *argv[]) {
  int N = atoi(argv[1]);
  int rank = atoi(argv[2]);
  int block_size = atoi(argv[3]);
  int nblocks = N / block_size;

  randvec_t randpts;
  randpts.push_back(equally_spaced_vector(N, 0.0, 1.0)); // 1D

  Hatrix::Context::init();
  const Hatrix::Matrix b = Hatrix::generate_random_matrix(N, 1);
  Hatrix::BLR A = construct_BLR(randpts, block_size, nblocks, rank, 0);
  Hatrix::Matrix last_lu = qsparse_factorize(A, N, nblocks, rank);
  Hatrix::Matrix x = qsparse_substitute(A, last_lu, b, nblocks, block_size, rank);

  Hatrix::Matrix A_dense = Hatrix::generate_laplacend_matrix(randpts, N, N, 0, 0);
  Hatrix::Matrix x_dense = Hatrix::lu_solve(A_dense, b);

  double error = Hatrix::norm(x - x_dense);
  std::cout << "solution error: " << error << std::endl;

  Hatrix::Context::finalize();
}
