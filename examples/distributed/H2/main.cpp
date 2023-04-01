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
  // std::mt19937 gen(0);
  // std::uniform_real_distribution<double> dist(0, 1);
  // Matrix x(opts.N, 1);
  // for (int64_t i = 0; i < opts.N; ++i) {
  //   x(i, 0) = dist(gen);
  // }
  // Matrix x = generate_rhs_vector(opts);
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
    // factorize_raw(A, opts);
    auto stop_factor = std::chrono::system_clock::now();
    factor_time = std::chrono::duration_cast<
      std::chrono::milliseconds>(stop_factor - begin_factor).count();

    post_factor_max_rank = A.max_rank();
    post_factor_average_rank = A.average_rank();

    auto begin_solve = std::chrono::system_clock::now();
    h2_solution = solve(A, b);
    // h2_solution = solve_raw(A, b);

    auto stop_solve = std::chrono::system_clock::now();
    solve_time = std::chrono::duration_cast<
      std::chrono::milliseconds>(stop_solve - begin_solve).count();
  }
  else {
    std::cerr << "Not implemented for non-symmetric matrices." << std::endl;
    abort();
  }

  // ||x - A * (A^-1 * x)|| / ||x||
  // h2_solution.print();
  auto diff = h2_solution - x;
  solve_error = Hatrix::norm(diff) / Hatrix::norm(x);
  Matrix bdense = matmul(Adense, x);
  // Matrix dense_solution = cholesky_solve(Adense, bdense, Hatrix::Lower);
  // construct_error = 0;
  std::cout << "DIFF: 32-48\n";
  for (int i = 32; i < 48; ++i) {
    std::cout << std::setw(12) << diff(i, 0) << " "
              << std::setw(12) << h2_solution(i, 0) << " "
              << std::setw(12) << x(i, 0) << " "
              << std::setw(12)  << b(i, 0) << std::endl;
  }
  // std::cout << "DIFF 48-64:\n";
  // for (int i = 48; i < 64; ++i) {
  //   std::cout << std::setw(12) << diff(i, 0) << " "
  //             << std::setw(12) << h2_solution(i, 0) << " "
  //             << std::setw(12) << x(i, 0) << " "
  //             << std::setw(12)  << b(i, 0) << std::endl;
  // }
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
            << opts.param_2 << ","
            << opts.param_3 << ","
            << opts.kernel_verbose
            << std::endl;

  Hatrix::Context::finalize();
  return 0;
}
