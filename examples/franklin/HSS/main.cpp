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

#include "Hatrix/Hatrix.h"

#include "franklin/franklin.hpp"

#include "SymmetricSharedBasisMatrix.hpp"
#include "operations.hpp"
#include "matrix_construction.hpp"


using namespace Hatrix;

int main(int argc, char* argv[]) {
  Hatrix::Context::init();

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
  domain.divide_domain_and_create_particle_boxes(opts.nleaf);
  auto stop_domain = std::chrono::system_clock::now();
  double domain_time = std::chrono::duration_cast<
    std::chrono::milliseconds>(stop_domain - start_domain).count();

  Matrix x = generate_random_matrix(opts.N, 1);
  x *= 1000;
  Matrix b;
  Matrix HSS_solution;

  double construct_time;
  int64_t HSS_max_rank;

  if (opts.is_symmetric) {
    auto begin_construct = std::chrono::system_clock::now();
    SymmetricSharedBasisMatrix A;
    if (opts.admis_kind == DIAGONAL) {
      init_diagonal_admis(A, domain, opts);
    }
    else {
      init_geometry_admis(A, domain, opts);
    }
    construct_h2_matrix_miro(A, domain, opts);
    auto stop_construct = std::chrono::system_clock::now();
    construct_time = std::chrono::duration_cast<
      std::chrono::milliseconds>(stop_construct - begin_construct).count();

    b = matmul(A, x);
    HSS_max_rank = A.max_rank();

    factorize(A);
    HSS_solution = solve(A, x);
  }

  Matrix Adense = generate_p2p_matrix(domain, opts.kernel);
  Matrix bdense = matmul(Adense, x);
  Matrix dense_solution = cholesky_solve(Adense, x, Hatrix::Lower);

  double matvec_error = Hatrix::norm(bdense - b) / Hatrix::norm(bdense);
  double solve_error = Hatrix::norm(dense_solution - HSS_solution) / opts.N;

  std::cout << "-------------------------------\n";
  std::cout << "N               : " << opts.N << std::endl;
  std::cout << "ACCURACY        : " << opts.accuracy << std::endl;
  std::cout << "OPT MAX RANK    : " << opts.max_rank << std::endl;
  std::cout << "REAL MAX RANK   : " << HSS_max_rank << std::endl;
  std::cout << "NLEAF           : " << opts.nleaf << "\n"
            << "Domain(ms)      : " << domain_time << "\n"
            << "Contruct(ms)    : " << construct_time << "\n"
            << "Construct error : " << matvec_error << std::endl;
  std::cout << "Solve error     : " << solve_error << std::endl;
  std::cout << "-------------------------------\n";

  Hatrix::Context::finalize();
  return 0;
}
