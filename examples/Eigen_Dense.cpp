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

constexpr double EPS = std::numeric_limits<double>::epsilon();
using vec = std::vector<int64_t>;

#define OUTPUT_CSV

void shift_diag(Hatrix::Matrix& A, const double shift) {
  for(int64_t i = 0; i < A.min_dim(); i++) {
    A(i, i) += shift;
  }
}

int64_t inertia(const Hatrix::Matrix& A, const double lambda, bool &singular) {
  // Shift diagonal
  Hatrix::Matrix A_shifted(A);
  shift_diag(A_shifted, -lambda);
  // LDL Factorize
  Hatrix::ldl(A_shifted);
  // Count negative entries in D
  int64_t negative_elements_count = 0;
  for (int64_t i = 0; i < A.min_dim(); i++) {
    negative_elements_count += (A_shifted(i, i) < 0 ? 1 : 0);
    if(std::isnan(A_shifted(i, i)) || std::abs(A_shifted(i, i)) < EPS) singular = true;
  }
  return negative_elements_count;
}

double get_mth_eigenvalue(const Hatrix::Matrix& A,
                          const int64_t m, const double ev_tol,
                          double left, double right) {
  bool singular = false;
  while((right - left) >= ev_tol) {
    const auto mid = (left + right) / 2;
    const auto value = inertia(A, mid, singular);
    if(singular) {
      std::cout << "Shifted matrix became singular (shift=" << mid << ")" << std::endl;
      break;
    }
    if(value >= m) right = mid;
    else left = mid;
  }
  return (left + right) / 2;
}

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
  const double ev_tol = argc > 5 ? atof(argv[5]) : 1.e-3;
  int64_t m_begin = argc > 6 ? atol(argv[6]) : 1;
  int64_t m_end = argc > 7 ? atol(argv[7]) : m_begin;
  double a = argc > 8 ? atof(argv[8]) : 0;
  double b = argc > 9 ? atof(argv[9]) : 0;
  const bool compute_eig_acc = argc > 10 ? (atol(argv[10]) == 1) : false;
  const int64_t print_csv_header = argc > 11 ? atol(argv[11]) : 1;

  // ELSES Input Files
  const std::string file_name = argc > 12 ? std::string(argv[12]) : "";

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
#ifndef OUTPUT_CSV
  std::cout << "N=" << N
            << " kernel=" << kernel_name
            << " geometry=" << geom_name
            << " construct_time=" << construct_time
            << std::endl;
#endif

  std::vector<double> dense_eigv;
  double dense_eig_time = 0;
  if (compute_eig_acc) {
    Hatrix::Matrix Adense = Hatrix::generate_p2p_matrix(domain);
    const auto dense_eig_start = std::chrono::system_clock::now();
    dense_eigv = Hatrix::get_eigenvalues(Adense);
    const auto dense_eig_stop = std::chrono::system_clock::now();
    dense_eig_time = std::chrono::duration_cast<std::chrono::milliseconds>
                     (dense_eig_stop - dense_eig_start).count();
  }
#ifndef OUTPUT_CSV
  std::cout << "dense_eig_time=" << dense_eig_time
            << std::endl;
#endif

  bool s = false;
  if (a == 0 && b == 0) {
    b = N < 10000 || is_non_synthetic ?
        Hatrix::norm(A) : N * (1. / Hatrix::PV);
    a = -b;
  }
  const auto v_a = inertia(A, a, s);
  const auto v_b = inertia(A, b, s);
  if(v_a != 0 || v_b != N) {
    std::cerr << "Warning: starting interval does not contain the whole spectrum "
              << "(v(a)=v(" << a << ")=" << v_a << ","
              << " v(b)=v(" << b << ")=" << v_b << ")"
              << std::endl;
  }
  // Determine which eigenvalue(s) to approximate
  std::vector<int64_t> target_m;
  if (m_begin <= 0) {
    const auto num = m_end;
    if (m_begin == 0) {
      std::mt19937 g(N);
      std::vector<int64_t> random_m(N, 0);
      for (int64_t i = 0; i < N; i++) {
        random_m[i] = i + 1;
      }
      std::shuffle(random_m.begin(), random_m.end(), g);
      for (int64_t i = 0; i < num; i++) {
        target_m.push_back(random_m[i]);
      }
    }
    if (m_begin == -1) {
      const auto linspace = Hatrix::equally_spaced_vector(num, 1, N, true);
      for (int64_t i = 0; i < num; i++) {
        target_m.push_back((int64_t)linspace[i]);
      }
    }
  }
  else {
    for (int64_t m = m_begin; m <= m_end; m++) {
      target_m.push_back(m);
    }
  }
#ifdef OUTPUT_CSV
  if (print_csv_header == 1) {
    // Print CSV header
    std::cout << "N,kernel,geometry"
              << ",construct_time"
              << ",dense_eig_time"
              << ",m,a0,b0,ev_tol,ldl_eig_time,dense_eigv,ldl_eigv,eig_abs_err,success"
              << std::endl;
  }
#endif
  for (int64_t k = 0; k < target_m.size(); k++) {
    const int64_t m = target_m[k];
    double ldl_mth_eigv;
    const auto ldl_eig_start = std::chrono::system_clock::now();
    ldl_mth_eigv = get_mth_eigenvalue(A, m, ev_tol, a, b);
    const auto ldl_eig_stop = std::chrono::system_clock::now();
    const double ldl_eig_time = std::chrono::duration_cast<std::chrono::milliseconds>
                                (ldl_eig_stop - ldl_eig_start).count();
    const double dense_mth_eigv = compute_eig_acc ? dense_eigv[m - 1] : -1;
    const double eig_abs_err = compute_eig_acc ? std::abs(ldl_mth_eigv - dense_mth_eigv) : -1;
    const bool success = compute_eig_acc ? (eig_abs_err < (0.5 * ev_tol)) : true;
#ifndef OUTPUT_CSV
    std::cout << "m=" << m
              << " a0=" << a
              << " b0=" << b
              << " ev_tol=" << ev_tol
              << " ldl_eig_time=" << ldl_eig_time
              << " dense_eigv=" << dense_mth_eigv
              << " ldl_eigv=" << ldl_mth_eigv
              << " eig_abs_err=" << std::scientific << eig_abs_err << std::defaultfloat
              << " success=" << (success ? "TRUE" : "FALSE")
              << std::endl;
#else
    std::cout << N
              << "," << kernel_name
              << "," << geom_name
              << "," << construct_time
              << "," << dense_eig_time
              << "," << m
              << "," << a
              << "," << b
              << "," << ev_tol
              << "," << ldl_eig_time
              << "," << dense_mth_eigv
              << "," << ldl_mth_eigv
              << "," << std::scientific << eig_abs_err << std::defaultfloat
              << "," << (success ? "TRUE" : "FALSE")
              << std::endl;
#endif
  }

  Hatrix::Context::finalize();
  return 0;
}

