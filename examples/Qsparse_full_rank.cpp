#include <iostream>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <fstream>

#ifdef USE_MKL
#include "mkl_cblas.h"
#include "mkl_lapacke.h"
#else
#include "cblas.h"
#include "lapacke.h"
#endif

#include "Hatrix/Hatrix.h"

using randvec_t = std::vector<std::vector<double> >;

double rel_error(const Hatrix::Matrix& A, const Hatrix::Matrix& B) {
  double A_norm = Hatrix::norm(A);
  double B_norm = Hatrix::norm(B);
  double diff = A_norm - B_norm;

  return std::sqrt((diff * diff) / (B_norm * B_norm));
}

std::vector<double> equally_spaced_vector(int N, double minVal, double maxVal) {
  std::vector<double> res(N, 0.0);
  double rnge = maxVal - minVal;
  for(int i=0; i<N; i++) {
    res[i] = minVal + ((double)i/(double)rnge);
  }
  return res;
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


int main(int argc, char *argv[]) {
  int N = atoi(argv[1]);
  randvec_t randpts;
  randpts.push_back(equally_spaced_vector(N, 0.0, 1.0)); // 1D

  Hatrix::Context::init();

  const Hatrix::Matrix b = Hatrix::generate_random_matrix(N, 1);
  Hatrix::Matrix A_dense = Hatrix::generate_laplacend_matrix(randpts, N, N, 0, 0);
  Hatrix::Matrix x_dense = Hatrix::lu_solve(A_dense, b);

  Hatrix::Matrix U, S, V;
  double error;
  Hatrix::Matrix A1 = Hatrix::generate_laplacend_matrix(randpts, N, N, 0, 0);
  std::tie(U, S, V, error) = Hatrix::truncated_svd(A1, A1.cols);
  Hatrix::Matrix Aprod = Hatrix::matmul(Hatrix::matmul(U, A1, true), V);
  Hatrix::lu(Aprod);

  Hatrix::Matrix x0 = Hatrix::matmul(U, b, true);
  Hatrix::Matrix x1(x0);
  Hatrix::solve_triangular(Aprod, x1, Hatrix::Left, Hatrix::Lower,
    true, false, 1.0);
  Hatrix::solve_triangular(Aprod, x1, Hatrix::Left, Hatrix::Upper,
    false, false, 1.0);
  Hatrix::Matrix x_prod = Hatrix::matmul(V, x1);

  x_prod.print();
  x_dense.print();

  double e = rel_error(x_prod, x_dense);
 std::cout << "solution error: " << e << std::endl;

  Hatrix::Context::finalize();

  return 0;
}