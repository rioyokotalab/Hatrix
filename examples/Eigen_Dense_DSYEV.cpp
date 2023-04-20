#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <tuple>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cassert>
#include <random>
#include <string>
#include <iomanip>
#include <functional>
#include <fstream>
#include <chrono>
#include <stdexcept>
#include <cstdio>

#include "Hatrix/Hatrix.h"
#include "Domain.hpp"
#include "functions.hpp"

#define OUTPUT_CSV

int main(int argc, char ** argv) {
  int64_t N = argc > 1 ? atol(argv[1]) : 256;
  // Specify kernel function
  // 0: Laplace Kernel
  // 1: Yukawa Kernel
  // 2: ELSES Dense Matrix
  const int64_t kernel_type = argc > 2 ? atol(argv[2]) : 0;

  // Specify underlying geometry
  // 0: Unit Circular
  // 1: Unit Cubical
  // 2: StarsH Uniform Grid
  // 3: ELSES Geometry (ndim = 3)
  const int64_t geom_type = argc > 3 ? atol(argv[3]) : 0;
  const int64_t ndim  = argc > 4 ? atol(argv[4]) : 1;
  const int64_t print_csv_header = argc > 5 ? atol(argv[5]) : 1;

  // ELSES Input Files
  const std::string file_name = argc > 6 ? std::string(argv[6]) : "";

  Hatrix::Context::init();

  Hatrix::set_kernel_constants(1e-3, 1.);
  std::string kernel_name = "";
  switch (kernel_type) {
    case 0: {
      Hatrix::set_kernel_function(Hatrix::laplace_kernel);
      kernel_name = "laplace";
      break;
    }
    case 1: {
      Hatrix::set_kernel_function(Hatrix::yukawa_kernel);
      kernel_name = "yukawa";
      break;
    }
    case 2: {
      Hatrix::set_kernel_function(Hatrix::ELSES_dense_input);
      kernel_name = "ELSES-kernel";
      break;
    }
    default: {
      Hatrix::set_kernel_function(Hatrix::laplace_kernel);
      kernel_name = "laplace";
    }
  }

  Hatrix::Domain domain(N, ndim);
  std::string geom_name = std::to_string(ndim) + "d-";
  switch (geom_type) {
    case 0: {
      domain.initialize_unit_circular_mesh();
      geom_name += "circular_mesh";
      break;
    }
    case 1: {
      domain.initialize_unit_cubical_mesh();
      geom_name += "cubical_mesh";
      break;
    }
    case 2: {
      domain.initialize_starsh_uniform_grid();
      geom_name += "starsh_uniform_grid";
      break;
    }
    case 3: {
      domain.ndim = 3;
      geom_name = file_name;
      break;
    }
    default: {
      domain.initialize_unit_circular_mesh();
      geom_name += "circular_mesh";
    }
  }
  // Pre-processing step for ELSES geometry
  const bool is_non_synthetic = (geom_type == 3);
  if (is_non_synthetic) {
    domain.read_bodies_ELSES(file_name + ".xyz");
    domain.read_p2p_matrix_ELSES(file_name + ".dat");
    N = domain.N;
  }
  domain.build_tree(N);

  const auto start_construct = std::chrono::system_clock::now();
  Hatrix::Matrix A = Hatrix::generate_p2p_matrix(domain);
  const auto stop_construct = std::chrono::system_clock::now();
  const double construct_time = std::chrono::duration_cast<std::chrono::milliseconds>
                                (stop_construct - start_construct).count();

  // Compute all eigenvalues
  const auto dense_eig_start = std::chrono::system_clock::now();
  const auto dense_eigv = Hatrix::get_eigenvalues(A);
  const auto dense_eig_stop = std::chrono::system_clock::now();
  const double dense_eig_time = std::chrono::duration_cast<std::chrono::milliseconds>
                                (dense_eig_stop - dense_eig_start).count();
  // Also include dsyev work array in memory consumption
  // https://netlib.org/lapack/explore-html/d2/d8a/group__double_s_yeigen_ga442c43fca5493590f8f26cf42fed4044.html
  const auto dense_eig_mem = A.memory_used() + sizeof(double) * (3*N - 1);

#ifndef OUTPUT_CSV
  std::cout << "N=" << N
            << " kernel=" << kernel_name
            << " geometry=" << geom_name
            << " construct_time=" << construct_time
            << " dense_eig_time=" << dense_eig_time
            << " dense_eig_mem=" << dense_eig_mem
            << std::endl;
#else
  if (print_csv_header == 1) {
    // Print CSV header
    std::cout << "N,kernel,geometry,construct_time,dense_eig_time,dense_eig_mem"
              << std::endl;
  }
  std::cout << N
            << "," << kernel_name
            << "," << geom_name
            << "," << construct_time
            << "," << dense_eig_time
            << "," << dense_eig_mem
            << std::endl;
#endif

  Hatrix::Context::finalize();
  return 0;
}

