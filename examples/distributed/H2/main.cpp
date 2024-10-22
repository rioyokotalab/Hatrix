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

#include "h2_construction.hpp"
#include "h2_operations.hpp"
#include "h2_factorize_tests.hpp"

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

static double cond_svd(const Matrix& A) {
  Matrix copy(A, true);
  Matrix _U(A, true), _S(A, true), _V(A, true);
  double error;

  svd(copy, _U, _S, _V);

  return _S(0,0) / _S(_S.rows-1, _S.cols-1);
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

  Matrix b, h2_solution, x_regen;
  Matrix x = Hatrix::generate_random_matrix(opts.N, 1);

  Matrix Adense = generate_p2p_matrix(domain, opts.kernel);

  if (opts.is_symmetric) {
    auto begin_construct = std::chrono::system_clock::now();
    SymmetricSharedBasisMatrix A;
    if (opts.admis_kind == GEOMETRY) {
      init_geometry_admis(A, domain, opts);
    }
    else if (opts.admis_kind == DIAGONAL) {
      init_diagonal_admis(A, domain, opts);
    }
    A.print_structure();

    construct_h2_matrix_miro(A, domain, opts);
    auto stop_construct = std::chrono::system_clock::now();
    construct_time = std::chrono::duration_cast<
      std::chrono::milliseconds>(stop_construct - begin_construct).count();

    construct_max_rank = A.max_rank();
    construct_average_rank = A.average_rank();
    dense_blocks = A.leaf_dense_blocks();

    auto begin_matvec = std::chrono::system_clock::now();
    b = matmul(A, x);
    auto stop_matvec = std::chrono::system_clock::now();
    matvec_time = std::chrono::duration_cast<
      std::chrono::milliseconds>(stop_matvec - begin_matvec).count();

    auto begin_factor = std::chrono::system_clock::now();
    fp_ops = factorize(A, opts);
    auto stop_factor = std::chrono::system_clock::now();
    factor_time = std::chrono::duration_cast<
      std::chrono::milliseconds>(stop_factor - begin_factor).count();

    post_factor_max_rank = A.max_rank();
    post_factor_average_rank = A.average_rank();

    auto begin_solve = std::chrono::system_clock::now();
    h2_solution = solve(A, b);
    auto stop_solve = std::chrono::system_clock::now();
    solve_time = std::chrono::duration_cast<
      std::chrono::milliseconds>(stop_solve - begin_solve).count();
  }
  else {
    std::cerr << "Not implemented for non-symmetric matrices." << std::endl;
    abort();
  }

  // ||x - A * (A^-1 * x)|| / ||x||
  auto diff = h2_solution - x;
  solve_error = Hatrix::norm(diff) / Hatrix::norm(x);
  Matrix bdense = matmul(Adense, x);
  construct_error = Hatrix::norm(bdense - b) / Hatrix::norm(b);

  double h2_norm = Hatrix::norm(h2_solution);
  double x_norm = Hatrix::norm(x);
  std::cout << "CONST. ERR: " << construct_error
            << " MAX RANK: " << construct_max_rank << std::endl;

  std::cout << "RESULT: " << opts.N << "," << opts.ndim << ","
            << opts.accuracy << ","
            << opts.qr_accuracy << ","
            << opts.kind_of_recompression << ","
            << opts.max_rank << ","
            << opts.admis << ","
            << construct_max_rank << ","
            << opts.nleaf <<  ","
            << domain_time <<  ","
            << construct_time  << ","
            << factor_time << ","
            << solve_time << ","
            << construct_error << ","
            << std::scientific << solve_error << ","
            << std::fixed << fp_ops << ","
            << opts.kind_of_geometry << ","
            << opts.use_nested_basis << ","
            << dense_blocks << ","
            << opts.perturbation << ","
            << std::scientific << opts.param_1 << std::fixed  << ","
            << std::scientific << opts.param_2 << std::fixed << ","
            << opts.param_3 << ","
            << opts.kernel_verbose << ",1" // fake MPISIZE
            << std::endl;

  Hatrix::Context::finalize();
  return 0;
}
