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

  std::cout << "start args.";
  Args opts(argc, argv);

  auto start_domain = std::chrono::system_clock::now();
  Domain domain(opts.N, opts.ndim);
  if (opts.kind_of_geometry == GRID) {
    domain.generate_grid_particles();
  }
  else if (opts.kind_of_geometry == CIRCULAR) {
    domain.generate_circular_particles(0, opts.N);
  }
  else if (opts.kind_of_geometry == COL_FILE_3D) {
    domain.read_col_file_3d(opts.geometry_file);
  }
  domain.build_tree(opts.nleaf);
  auto stop_domain = std::chrono::system_clock::now();
  double domain_time = std::chrono::duration_cast<
    std::chrono::milliseconds>(stop_domain - start_domain).count();

  Matrix b, h2_solution, x_regen;
  double construct_time, matvec_time, factor_time, solve_time;
  int64_t construct_max_rank, construct_average_rank,
    post_factor_max_rank, post_factor_average_rank;

  std::mt19937 gen(0);
  std::uniform_real_distribution<double> dist(0, 1);
  Matrix x(opts.N, 1);
  for (int64_t i = 0; i < opts.N; ++i) {
    x(i, 0) = dist(gen);
  }

  long long int fp_ops;
  int64_t dense_blocks;

  if (opts.is_symmetric) {
    auto begin_construct = std::chrono::system_clock::now();
    SymmetricSharedBasisMatrix A;
    init_geometry_admis(A, domain, opts);
    construct_h2_matrix_miro(A, domain, opts);
    auto stop_construct = std::chrono::system_clock::now();
    construct_time = std::chrono::duration_cast<
      std::chrono::milliseconds>(stop_construct - begin_construct).count();

    construct_max_rank = A.max_rank();
    construct_average_rank = A.average_rank();
    dense_blocks = A.leaf_dense_blocks();

    A.print_structure();

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
    // x_regen = matmul(A_orig, h2_solution);

    auto stop_solve = std::chrono::system_clock::now();
    solve_time = std::chrono::duration_cast<
      std::chrono::milliseconds>(stop_solve - begin_solve).count();
  }
  else {
    std::cerr << "Not implemented for non-symmetric matrices." << std::endl;
    abort();
  }

  // ||x - A * (A^-1 * x)|| / ||x||
  // std::cout << "SOLVE: " << (Hatrix::norm(x - h2_solution) / Hatrix::norm(x));
  double solve_error = Hatrix::norm(x - h2_solution) / opts.N;

  // Matrix Adense = generate_p2p_matrix(domain, opts.kernel);
  // Matrix dense_solution = cholesky_solve(Adense, b, Hatrix::Lower);
  // double solve_error = Hatrix::norm(dense_solution - h2_solution) / opts.N;

  // Matrix Adense = generate_p2p_matrix(domain, opts.kernel);
  // Matrix bdense = matmul(Adense, x);

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
            << solve_error << ","
            << fp_ops << ","
            << opts.kind_of_geometry << ","
            << opts.use_nested_basis << ","
            << dense_blocks
            << std::endl;

  // std::cout << "----------------------------\n";
  // std::cout << "N               : " << opts.N << std::endl;
  // std::cout << "NDIM            : " << opts.ndim << std::endl;
  // std::cout << "ACCURACY        : " << opts.accuracy << std::endl;
  // std::cout << "QR ACCURACY     : " << opts.qr_accuracy << std::endl;
  // std::cout << "RECOMP. KIND    : " << opts.kind_of_recompression << std::endl;
  // std::cout << "OPT MAX RANK    : " << opts.max_rank << std::endl;
  // std::cout << "ADMIS           : " << opts.admis << std::endl;
  // std::cout << "REAL MAX RANK   : " << construct_max_rank << std::endl;
  // std::cout << "POST FACT. RANK : " << post_factor_max_rank << std::endl;
  // std::cout << "NLEAF           : " << opts.nleaf << "\n"
  //           << "Domain(ms)      : " << domain_time << "\n"
  //           << "Contruct(ms)    : " << construct_time << "\n"
  //           << "Factor(ms)      : " << factor_time << "\n"
  //           << "Solve(ms)       : " << solve_time << "\n"
  //           << "Solve error     : " << solve_error << "\n"
  //           << "PAPI_FP_OPS     : " << fp_ops << "\n"
  //           << "Kind of recomp. : " << opts.kind_of_recompression
  //           << std::endl;
  // std::cout << "----------------------------\n";



  Hatrix::Context::finalize();
  return 0;
}
