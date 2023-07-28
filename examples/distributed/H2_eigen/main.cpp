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

extern "C" {
#include "elses.h"
}

using namespace Hatrix;

void inertia(Matrix& A) {

}

void slicing_the_spectrum(Matrix& A, int64_t m,
                          double eigen_interval_start, double eigen_interval_end) {
}


int main(int argc, char* argv[]) {
  Hatrix::Context::init();
  Args opts(argc, argv);
  const int64_t num_electrons_per_atom = 4, num_atoms_per_molecule = 60;
  const int64_t molecule_size = num_electrons_per_atom * num_atoms_per_molecule;

  init_elses_state();

  Domain domain(opts.N, opts.ndim);

  Matrix A_dense(opts.N, opts.N);
#pragma omp parallel for collapse(2)
  for (long int i = 0; i < opts.N; ++i) {
    for (long int j = 0; j < opts.N; ++j) {
      long int f_i = i + 1;
      long int f_j = j + 1;
      double val;
      get_elses_matrix_value(&f_i, &f_j, &val);
      A_dense(i, j) = val;
    }
  }

  auto A_copy = Matrix(A_dense, true);
  auto dense_eign = Hatrix::get_eigenvalues(A_copy);

  // Compute all the eigen values between m_begin'th and m_end'th.
  int64_t k_begin = opts.N / 2, k_end = opts.N / 2;
  assert(k_begin <= k_end);

  // Determine the interval within which the eigen values can reside.
  double eigen_interval_start = opts.param_1;
  double eigen_interval_end = opts.param_2;

  // Assume that we only want eigen values from a uniformly spaced interval.
  for (int64_t k = k_begin; k <= k_end; ++k) {
    slicing_the_spectrum(A_dense, k, eigen_interval_start, eigen_interval_end);
  }


  Hatrix::Context::finalize();
  return 0;
}
