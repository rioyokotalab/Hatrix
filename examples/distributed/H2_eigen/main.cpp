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


int main(int argc, char* argv[]) {
  Hatrix::Context::init();
  Args opts(argc, argv);
  const int64_t num_electrons_per_atom = 4, num_atoms_per_molecule = 60;
  const int64_t molecule_size = num_electrons_per_atom * num_atoms_per_molecule;

  init_elses_state();



  Domain domain(opts.N, opts.ndim);

  if (opts.kernel_verbose == "elses_c60") {
    domain.read_xyz_chemical_file(opts.geometry_file, num_electrons_per_atom);
  }
  domain.sort_bodies_elses(molecule_size);

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

  auto dense_eign = Hatrix::get_eigenvalues(A_dense);


  // Compute the kth eigen value.

  Hatrix::Context::finalize();
  return 0;
}
