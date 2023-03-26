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

  std::mt19937 gen(0);
  std::uniform_real_distribution<double> dist(0, 1);
  Matrix x(opts.N, 1);
  for (int64_t i = 0; i < opts.N; ++i) {
    x(i, 0) = dist(gen);
  }

  Matrix Adense = generate_p2p_matrix(domain, opts.kernel);
  Matrix bdense = matmul(Adense, x);
  Matrix dense_solution = cholesky_solve(Adense, bdense, Hatrix::Lower);
  solve_error = Hatrix::norm(dense_solution - x) / Hatrix::norm(x);

  std::cout << "DENSE SOLVER: N->" << opts.N
            << " COND -> " << Hatrix::cond(Adense)
            << " KERNEL -> " << opts.kernel_verbose
            << " SOLVE ERR. -> " << std::scientific << solve_error << std::fixed
            << " PARAMS -> " << std::scientific << opts.param_1 << std::fixed
            << "," << opts.param_2 << "," << opts.param_3 << std::endl;
}
