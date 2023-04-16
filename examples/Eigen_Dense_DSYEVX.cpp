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

#include "nlohmann/json.hpp"

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
  int64_t ndim  = argc > 4 ? atol(argv[4]) : 1;
  // Eigenvalue computation parameters
  const double abs_tol = argc > 5 ? atof(argv[5]) : 1e-3;
  const int64_t k_begin = argc > 6 ? atol(argv[6]) : 1;
  const int64_t k_end = argc > 7 ? atol(argv[7]) : k_begin;
  const bool compute_eig_acc = argc > 8 ? (atol(argv[8]) == 1) : true;
  const int64_t print_csv_header = argc > 9 ? atol(argv[9]) : 1;

  // ELSES Input Files
  const std::string file_name = argc > 10 ? std::string(argv[10]) : "";

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

  std::vector<double> dense_eigv;
  double dense_eig_time = 0;
  if (compute_eig_acc) {
    Hatrix::Matrix Acopy(A);
    const auto dense_eig_start = std::chrono::system_clock::now();
    dense_eigv = Hatrix::get_eigenvalues(Acopy);
    const auto dense_eig_stop = std::chrono::system_clock::now();
    dense_eig_time = std::chrono::duration_cast<std::chrono::milliseconds>
                     (dense_eig_stop - dense_eig_start).count();
  }

  // Compute selected eigenvalues
  const auto actual_eig_start = std::chrono::system_clock::now();
  const auto actual_eigv = Hatrix::get_selected_eigenvalues(A, k_begin, k_end, abs_tol);
  const auto actual_eig_stop = std::chrono::system_clock::now();
  const double actual_eig_time = std::chrono::duration_cast<std::chrono::milliseconds>
                                 (actual_eig_stop - actual_eig_start).count();

#ifndef OUTPUT_CSV
  std::cout << "N=" << N
            << " kernel=" << kernel_name
            << " geometry=" << geom_name
            << " abs_tol=" << abs_tol
            << " k_begin=" << k_begin
            << " k_end=" << k_end
            << " construct_time=" << construct_time
            << " dense_eig_time=" << dense_eig_time
            << " actual_eig_time=" << actual_eig_time
            << std::endl;
#else
  if (print_csv_header == 1) {
    // Print CSV header
    std::cout << "N,kernel,geometry,abs_tol,k_begin,k_end"
              << ",construct_time,dense_eig_time,actual_eig_time"
              << ",k,dense_eigv,actual_eigv,eig_abs_err"
              << std::endl;
  }
#endif

  for (int64_t k = k_begin; k <= k_end; k++) {
    const auto dense_eigv_k = compute_eig_acc ? dense_eigv[k - 1] : -1;
    const auto actual_eigv_k = actual_eigv[k - k_begin];
    const auto eig_abs_err = compute_eig_acc ? std::abs(dense_eigv_k - actual_eigv_k) : -1;
#ifndef OUTPUT_CSV
    std::cout << "k=" << k
              << std::setprecision(8)
              << " dense_eigv=" << dense_eigv_k
              << " actual_eigv=" << actual_eigv_k
              << std::setprecision(3)
              << " eig_abs_err=" << std::scientific << eig_abs_err << std::defaultfloat
              << std::setprecision(1)
              << std::endl;
#else
    std::cout << N
              << "," << kernel_name
              << "," << geom_name
              << "," << abs_tol
              << "," << k_begin
              << "," << k_end
              << "," << construct_time
              << "," << dense_eig_time
              << "," << actual_eig_time
              << "," << k
              << std::setprecision(8)
              << "," << dense_eigv_k
              << "," << actual_eigv_k
              << std::setprecision(3)
              << "," << std::scientific << eig_abs_err << std::defaultfloat
              << std::setprecision(1)
              << std::endl;
#endif
  }

  Hatrix::Context::finalize();
  return 0;
}

