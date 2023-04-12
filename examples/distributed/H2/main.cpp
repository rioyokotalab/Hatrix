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

void
block_chol(SymmetricSharedBasisMatrix& A, Args& opts, const Domain& domain) {
  int level = A.max_level;
  int64_t nblocks = pow(2, level);

  for (int i = 0; i < nblocks; ++i) {
    for (int j = i + 1; j < nblocks; ++j) {
      if (A.is_admissible.exists(i, j, level)) {
        A.is_admissible.erase(i, j, level);
        A.is_admissible.insert(i, j, level, false);
      }
    }
  }

  Matrix Adense = generate_p2p_matrix(domain, opts.kernel);
  // cholesky(Adense, Hatrix::Lower);
  auto A_splits = Adense.split(nblocks, nblocks);

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j <= i; ++j) {
      A.D.erase(i, j, level);
      Matrix Aij(A_splits[i * nblocks + j], true);
      // A.D.insert(i, j, level, std::move(Aij));
      A.D.insert(i, j, level, generate_p2p_interactions(domain, i, j, level, opts.kernel));
    }
  }


  for (int64_t block = 0; block < nblocks; ++block) {
    auto U_F = make_complement(A.U(block, level));
    for (int64_t j = 0; j <= block; ++j) {
      A.D(block, j, level) = matmul(U_F, A.D(block, j, level), true);
    }

    for (int64_t i = block; i < nblocks; ++i) {
      A.D(i, block, level) = matmul(A.D(i, block, level), U_F);
    }

    std::cout << "cond block -> " << block <<  " " << cond_svd(A.D(block, block, level)) << std::endl;
    cholesky(A.D(block, block, level), Hatrix::Lower);
    for (int64_t i = block+1; i < nblocks; ++i) {
      solve_triangular(A.D(block, block, level), A.D(i, block, level), Hatrix::Right, Hatrix::Lower,
                       false, true, 1);
    }

    for (int64_t i = block+1; i < nblocks; ++i) {
      syrk(A.D(i, block, level), A.D(i, i, level), Hatrix::Lower, false, -1, 1);
      for (int64_t j = i+1; j < nblocks; ++j) {
        matmul(A.D(j, block, level), A.D(i, block, level), A.D(j, i, level), false, true, -1, 1);
      }
    }
  }
}

Matrix
solve_chol(SymmetricSharedBasisMatrix& A, const Matrix& b, Args& opts) {
  Matrix x(b, true);
  int level = A.max_level;
  int64_t nblocks = pow(2, level);
  auto x_splits = x.split(nblocks, 1);

  for (int64_t block = 0; block < nblocks; ++block) {
    auto U_F = make_complement(A.U(block, level));
    auto prod = matmul(U_F, x_splits[block]);
    x_splits[block] = prod;

    solve_triangular(A.D(block, block, level), x_splits[block], Hatrix::Left, Hatrix::Lower,
                     false, false, 1.0);
    for (int64_t i = block+1; i < nblocks; ++i) {
      matmul(A.D(i, block, level), x_splits[block], x_splits[i], false, false, -1, 1);
    }
  }

  // dense solve goes here.

  for (int64_t block = nblocks-1; block >= 0; --block) {
    for (int64_t j = nblocks-1; j > block; --j) {
      matmul(A.D(j, block, level), x_splits[j], x_splits[block], true, false, -1, 1);
    }

    solve_triangular(A.D(block, block, level), x_splits[block], Hatrix::Left, Hatrix::Lower,
                     false, true);

    auto U_F = make_complement(A.U(block, level));
    auto prod = matmul(U_F, x_splits[block], true);
    x_splits[block] = prod;
  }

  return x;
}


Matrix
solve_chol_hor(SymmetricSharedBasisMatrix& A, const Matrix& b, Args& opts) {
  Matrix x(b, true);
  int level = A.max_level;
  int64_t nblocks = pow(2, level);
  auto x_splits = x.split(nblocks, 1);

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < i; ++j) {
      matmul(A.D(i, j, level), x_splits[j], x_splits[i], false, false, -1, 1);
    }
    solve_triangular(A.D(i, i, level), x_splits[i], Hatrix::Left, Hatrix::Lower, false, false);
  }

  for (int64_t i = nblocks-1; i >= 0; --i) {
    for (int64_t j = nblocks-1; j > i; --j) {
      matmul(A.D(j, i, level), x_splits[j], x_splits[i], true, false, -1, 1);
    }
    solve_triangular(A.D(i, i, level), x_splits[i], Hatrix::Left, Hatrix::Lower, false, true);
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
    // block_chol(A, opts, domain);
    auto stop_factor = std::chrono::system_clock::now();
    factor_time = std::chrono::duration_cast<
      std::chrono::milliseconds>(stop_factor - begin_factor).count();

    post_factor_max_rank = A.max_rank();
    post_factor_average_rank = A.average_rank();

    auto begin_solve = std::chrono::system_clock::now();
    h2_solution = solve(A, b);
    // h2_solution = solve_chol(A, b, opts);
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
