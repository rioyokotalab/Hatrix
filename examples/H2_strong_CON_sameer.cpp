#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <chrono>
#include <fstream>

#include "Hatrix/Hatrix.hpp"
using namespace Hatrix;
Hatrix::greens_functions::kernel_function_t kernel;

int main(int argc, char ** argv) {
  if (argc == 1) {
    std::cout << "HELP SCREEN FOR H2_strong_CON.cpp" << std::endl;
    std::cout << "Specify arguments as follows: " << std::endl;
    std::cout << "N leaf_size accuracy max_rank random_matrix_size admis kernel_type geom_type ndim matrix_type" << std::endl;
    return 0;
  }

  const int64_t N = argc > 1 ? atol(argv[1]) : 256;
  const int64_t leaf_size = argc > 2 ? atol(argv[2]) : 32;
  const double accuracy = argc > 3 ? atof(argv[3]) : 1.e-5;
  const int64_t max_rank = argc > 4 ? atol(argv[4]) : 30;
  const int64_t random_matrix_size = argc > 5 ? atol(argv[5]) : 100;
  const double admis = argc > 6 ? atof(argv[6]) : 1.0;

  // Specify kernel function
  // 0: Laplace Kernel
  // 1: Yukawa Kernel
  const int64_t kernel_type = argc > 7 ? atol(argv[7]) : 0;

  // Specify underlying geometry
  // 0: Unit Circular
  // 1: Unit Cubical
  // 2: StarsH Uniform Grid
  const int64_t geom_type = argc > 8 ? atol(argv[8]) : 0;
  const int64_t ndim  = argc > 9 ? atol(argv[9]) : 2;

  // Specify compressed representation
  // 0: BLR2
  // 1: H2
  const int64_t matrix_type = argc > 10 ? atol(argv[10]) : 1;

  const double add_diag = 1e-3 / N;
  const double alpha = 1;
  // Setup the kernel.
  switch (kernel_type) {
  case 1:                       // yukawa
    kernel = [&](const std::vector<double>& c_row,
                 const std::vector<double>& c_col) {
      return Hatrix::greens_functions::yukawa_kernel(c_row, c_col, alpha, add_diag);
    };
    break;
  default:                      // laplace
    kernel = [&](const std::vector<double>& c_row,
                 const std::vector<double>& c_col) {
      if (ndim == 1) {
        return Hatrix::greens_functions::laplace_1d_kernel(c_row, c_col, add_diag);
      }
      else if (ndim == 2) {
        return Hatrix::greens_functions::laplace_2d_kernel(c_row, c_col, add_diag);
      }
      else if (ndim == 3) {
        return Hatrix::greens_functions::laplace_3d_kernel(c_row, c_col, add_diag);
      }
    };
  }

  // Setup the Domain
  Hatrix::Domain domain(N, ndim);
  switch(geom_type) {
  case 1:                       // cube mesh
    domain.generate_grid_particles();
  default:                      // circle / sphere mesh
    domain.generate_circular_particles();
  }
}
