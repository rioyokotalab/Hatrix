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

  // for (long int i = 1; i <= 7680; ++i) {
  //   for (long int j = 1; j <= 7680; ++j) {
  //     get_elses_matrix_value(&i, &j, &val);
  //     std::cout << "i: " << i << " j: " << j << " "  << val << std::endl;
  //   }
  // }

  Domain domain(opts.N, opts.ndim);

  if (opts.kernel_verbose == "elses_c60") {
    domain.read_xyz_chemical_file(opts.geometry_file, num_electrons_per_atom);
  }
  domain.sort_bodies_elses(molecule_size);

  Matrix A_dense;

  Hatrix::Context::finalize();
  return 0;
}
