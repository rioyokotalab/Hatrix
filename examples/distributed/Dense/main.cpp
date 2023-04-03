#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <iomanip>
#include <random>

#include "Hatrix/Hatrix.h"

#include "distributed/distributed.hpp"

using namespace Hatrix;

static Hatrix::Matrix
generate_rhs_vector(Hatrix::Args& opts) {
  double cmax = 1;
  std::mt19937 gen(0);
  std::uniform_real_distribution<double> dist(0, cmax);
  Matrix x(opts.N, 1);
  double avg = 0;
  double cmax2 = cmax * 2;

  for (int64_t i = 0; i < opts.N; ++i) {
    double c = dist(gen) * cmax2 - cmax;
    x(i, 0) = c;
    avg += c;
  }
  avg /= opts.N;

  if (avg != 0.) {
    for (int64_t i = 0; i < opts.N; ++i) {
      x(i, 0) -= avg;
    }
  }

  return x;
}

void block_lu(Matrix& A, Args& opts) {
  int NB = opts.N / opts.nleaf;
  auto A_splits = A.split(NB, NB);
  for (int d = 0; d < NB; ++d) {
    lu(A_splits[d*NB + d]);
    for (int i = d+1; i < NB; ++i) {
      solve_triangular(A_splits[d*NB + d], A_splits[d*NB + i], Hatrix::Left, Hatrix::Lower,
                       true, false, 1.0);
      solve_triangular(A_splits[d*NB + d], A_splits[i*NB + d], Hatrix::Right, Hatrix::Upper,
                       false, false, 1.0);
    }

    for (int i = d+1; i < NB; ++i) {
      for (int j = d+1; j < NB; ++j) {
        matmul(A_splits[i*NB + d], A_splits[d*NB + j], A_splits[i*NB + j], false, false, -1.0, 1.0);
      }
    }
  }
}


Matrix solve_lu(Matrix& A, const Matrix& b, Args& opts) {
  Matrix x(b, true);
  int64_t NB = opts.N / opts.nleaf;
  auto x_splits = x.split(NB, 1);
  auto A_splits = A.split(NB, NB);

  for (int i = 0; i < NB; ++i) {
    for (int j = 0; j < i; ++j) {
      matmul(A_splits[i * NB + j], x_splits[j], x_splits[i], false, false, -1, 1);
    }
    solve_triangular(A_splits[i * NB + i], x_splits[i], Hatrix::Left, Hatrix::Lower, true, false);
  }

  std::cout << "LU POST FORWARD: " << Hatrix::norm(x) << std::endl;

  for (int i = NB-1; i >= 0; --i) {
    for (int j = NB-1; j > i; --j) {
      matmul(A_splits[i * NB + j], x_splits[j], x_splits[i], false, false, -1, 1);
    }
    solve_triangular(A_splits[i * NB + i], x_splits[i], Hatrix::Left, Hatrix::Upper, false, false);
  }

  std::cout << "LU POST BACK: " << Hatrix::norm(x) << std::endl;

  return x;
}


void block_cholesky(Matrix& A, Args& opts) {
  int NB = opts.N / opts.nleaf;
  auto A_splits = A.split(NB, NB);
  for (int d = 0; d < NB; ++d) {
    cholesky(A_splits[d*NB + d], Hatrix::Lower);
    for (int i = d+1; i < NB; ++i) {
      solve_triangular(A_splits[d*NB + d], A_splits[i*NB + d], Hatrix::Right, Hatrix::Lower,
                       false, true, 1.0);
    }

    for (int i = d+1; i < NB; ++i) {
      syrk(A_splits[i * NB + d], A_splits[i * NB + i], Hatrix::Lower, false, -1, 1);
      for (int j = i+1; j < NB; ++j) {
        matmul(A_splits[j*NB + d], A_splits[i*NB + d], A_splits[j*NB + i], false, true, -1.0, 1.0);
      }
    }
  }
}

Matrix solve_chol(Matrix& A, const Matrix& b, Args& opts) {
  Matrix x(b, true);
  int64_t NB = opts.N / opts.nleaf;
  auto x_splits = x.split(NB, 1);
  auto A_splits = A.split(NB, NB);

  for (int i = 0; i < NB; ++i) {
    for (int j = 0; j < i; ++j) {
      matmul(A_splits[i * NB + j], x_splits[j], x_splits[i], false, false, -1, 1);
    }
    solve_triangular(A_splits[i * NB + i], x_splits[i], Hatrix::Left, Hatrix::Lower, false, false);
  }

  std::cout << "CHOL POST FORWARD: " << Hatrix::norm(x) << std::endl;

  for (int i = NB-1; i >= 0; --i) {
    for (int j = NB-1; j > i; --j) {
      matmul(A_splits[j * NB + i], x_splits[j], x_splits[i], true, false, -1, 1);
    }
    solve_triangular(A_splits[i * NB + i], x_splits[i], Hatrix::Left, Hatrix::Lower, false, true);
  }

  std::cout << "CHOL POST BACK: " << Hatrix::norm(x) << std::endl;

  return x;
}


int main(int argc, char* argv[]) {
  Hatrix::Context::init();
  long long int fp_ops;
  int64_t dense_blocks;
  double construct_time, matvec_time, factor_time,
    solve_time, solve_error, construct_error;
  int64_t construct_max_rank, construct_average_rank,
    post_factor_max_rank, post_factor_average_rank;

  Args opts(argc, argv);

  auto start_domain = std::chrono::system_clock::now();
  Domain domain(opts.N, opts.ndim);
  if (opts.kind_of_geometry == GRID) {
    domain.generate_grid_particles();
  }
  else if (opts.kind_of_geometry == CIRCULAR) {
    domain.generate_circular_particles(0, opts.N);
  }
  else if (opts.kind_of_geometry == COL_FILE && opts.ndim == 3) {
    domain.read_col_file_3d(opts.geometry_file);
  }
  else if (opts.kind_of_geometry == COL_FILE && opts.ndim == 2) {
    domain.read_col_file_2d(opts.geometry_file);
  }

  domain.build_tree(opts.nleaf);
  auto stop_domain = std::chrono::system_clock::now();
  double domain_time = std::chrono::duration_cast<
    std::chrono::milliseconds>(stop_domain - start_domain).count();

  Matrix x = Hatrix::generate_random_matrix(opts.N, 1);

  Matrix Adense = generate_p2p_matrix(domain, opts.kernel);
  Matrix bdense = matmul(Adense, x);
  // block_lu(Adense, opts);
  // Matrix dense_solution = solve_lu(Adense, bdense, opts);

  block_cholesky(Adense, opts);
  Matrix dense_solution = solve_chol(Adense, bdense, opts);

  auto diff = dense_solution - x;

  for (int i = 0; i < 64; ++i) {
    std::cout << std::setprecision(8) << std::setw(15) << diff(i, 0) << " "
              << std::setprecision(8) << std::setw(15) << dense_solution(i, 0) << " "
              << std::setprecision(8) << std::setw(15) << x(i, 0) << std::endl;
  }

  solve_error = Hatrix::norm(dense_solution - x) / Hatrix::norm(x);

  std::cout << "DENSE SOLVER: N->" << opts.N
            << " COND -> " << Hatrix::cond(Adense)
            << " KERNEL -> " << opts.kernel_verbose
            << " SOLVE ERR. -> " << std::scientific << solve_error << std::fixed
            << " PARAMS -> " << std::scientific << opts.param_1 << std::fixed
            << "," << opts.param_2 << "," << opts.param_3 << std::endl;
}
