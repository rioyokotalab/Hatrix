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

#include "functions.hpp"
#include "internal_types.hpp"
#include "SharedBasisMatrix.hpp"

using namespace Hatrix;

const std::vector<std::string> valid_opts = {
  "--N",                        // 0
  "--nleaf",                    // 1
  "--kernel-func",              // 2
  "--kind-of-geometry",         // 3
  "--ndim",                     // 4
  "--rank",                     // 5
  "--acc",                      // 6
  "--admis",                    // 7
  "--admis_kind",               // 8
  "--construct-algorithm",      // 9
  "--add-diag",                 // 10
  "--nested-basis"              // 11
};

bool help_option_exists(std::vector<std::string>& cmd_options) {
  auto begin = cmd_options.begin();
  auto end = cmd_options.end();
  return (std::find(begin, end, "-h") != end) || (std::find(begin, end, "--help") != end);
}

void validate_cmd(std::vector<std::string>& cmd_options) {
  bool found_rank = false, found_acc = false;
  for (std::string option : cmd_options) {
    if (option.find("--", 0) != std::string::npos) {
      if (std::find(valid_opts.begin(), valid_opts.end(), option) == valid_opts.end()) {
        std::cout << "Found unknown option " << option << std::endl;
        abort();
      }

      if (option == valid_opts[5]) {
        found_rank = true;
      }
      if (option == valid_opts[6]) {
        found_acc = true;
      }
    }
  }

  if (found_rank && found_acc) {
    throw std::invalid_argument("validate_cmd()-> --rank and --acc are mutually exclusive.");
  }
}


int main(int argc, char* argv[]) {
  Hatrix::Context::init();
  std::vector<std::string> cmd_options;

  for (int i = 0; i < argc; ++i) {
    cmd_options.push_back(std::string(argv[i]));
  }

  if (help_option_exists(cmd_options)) {
    std::cout << "FRANKLIN help screen. Thank you for using FRANKLIN! May all your factorizations be linear." << std::endl;
    std::cout << "FRANKLIN accepts the following options: " << std::endl;
    std::cout << " " << valid_opts[0] << ":: Specify problem size. Generates a NxN matrix. Must be integer." << std::endl
              << " " << valid_opts[1] << ":: Specify leaf size. This is the maximum size of the tile in the lowest level of the tree. Alternatively maximum number of particles in the leaf node. Must be integer." << std::endl
              << " --kernel-func :: Specify the kernel function: " << std::endl
              << "    laplace -- Laplace kernel." << std::endl
              << "    sqrexp  -- Sqr. exp. kernel from stars-H." << std::endl
              << "    sin     -- Sine kernel from stars-H." << std::endl
              << " --kind-of-geometry :: Specify the kind of geometry: " << std::endl
              << "    grid     -- Random unit grid with particles on the surface. ndim determines whether line, square or cube." << std::endl
              << "    circular -- Random unit circular surface. ndim determines whether line, circle or sphere." << std::endl
              << " --ndim :: Number of dimensions of the problem. 1, 2 or 3" << std::endl
              << " --rank :: Specify the maximum rank for a fixed rank calculation." << std::endl
              << " --acc :: Specify the accuracy of the calculation. The --rank option is ignored if this is specified." << std::endl
              << " --admis :: Specify the admissibility constant. Double precision number >= 0." << std::endl
              << " --admis-kind :: Specify whether geometry-based admissiblity or diagonal-based: " << std::endl
              << "    geometry :: Admissiblity condition based on the geometry of the problem." << std::endl
              << "    diagonal :: Admissibility condition based on the distance from the diagonal."
              << " --construct-algorithm :: Specify the kind of construction algorithm." << std::endl
              << "    miro       -- From the miro board. Use SVD everywhere." << std::endl
              << "    miro_fast  -- Imporovised miro board with SVD." << std::endl
              << "    id_random  -- From Martinsson2010. Use ID + randomized sampling."
              << " --add-diag :: Specify the value to add to the diagonal."
              << " --nested-basis :: Specify whether this matrix uses shared basis. 0 or 1."
              << std::endl;
  }

  int64_t N = 1000;
  int64_t nleaf = 100;
  KERNEL_FUNC kernel_func = LAPLACE;
  KIND_OF_GEOMETRY kind_of_geometry = GRID;
  int64_t ndim = 1;
  int64_t rank = -1;
  double admis = 0;
  double acc = 1;
  double add_diag = 0;
  ADMIS_KIND admis_kind = DIAGONAL;
  CONSTRUCT_ALGORITHM construct_algorithm = MIRO;
  bool use_nested_basis = true;

  validate_cmd(cmd_options);

  for (auto iter = cmd_options.begin(); iter != cmd_options.end(); ++iter) {
    auto option = *iter;

    if (option == valid_opts[0]) {
      N = std::stol(*(++iter));
    }
    else if (option == valid_opts[1]) {
      nleaf = std::stol(*(++iter));
    }
    else if (option == valid_opts[2]) {
      auto value = *(++iter);
      if (value == "laplace") {
        kernel_func = LAPLACE;
      }
      else if (value == "sqrexp") {
        kernel_func = SQR_EXP;

      }
      else if (value == "sin") {
        kernel_func = SINE;
      }
      else {
        std::cout << "wrong value for --kind-of-problem " << value << std::endl;
        abort();
      }
    }
    else if (option == valid_opts[3]) {
      auto value = *(++iter);
      if (value == "grid") {
        kind_of_geometry = GRID;
      }
      else if (value == "circular") {
        kind_of_geometry = CIRCULAR;
      }
      else {
        std::cout << "wrong value for --kind-of-geometry " << value << std::endl;
        abort();
      }
    }
    else if (option == valid_opts[4]) {
      ndim = std::stol(*(++iter));
    }
    else if (option == valid_opts[5]) {
      rank = std::stol(*(++iter));
      acc = 1;
    }
    else if (option == valid_opts[6]) {
      acc = std::stod(*(++iter));
      rank = -1;
    }
    else if (option == valid_opts[7]) {
      admis = std::stod(*(++iter));
    }
    else if (option == valid_opts[8]) {
      auto value = *(++iter);
      if (value == "geometry") {
        admis_kind = GEOMETRY;
      }
      else if (value == "diagonal") {
        admis_kind = DIAGONAL;
      }
      else {
        std::cout << "wrong value for --admis-kind " << value << std::endl;
        abort();
      }
    }
    else if (option == valid_opts[9]) {
      auto value = *(++iter);
      if (value == "miro") {
        construct_algorithm = MIRO;
      }
      else if (value == "id_random") {
        construct_algorithm = ID_RANDOM;
      }
      else {
        std::cout << "wrong value for --construct-algorithm "
                  << value << std::endl;
        abort();
      }
    }
    else if (option == valid_opts[10]) {
      add_diag = std::stod(*(++iter));
    }
    else if (option == valid_opts[11]) {
      use_nested_basis = bool(std::stoi(*(++iter)));
    }
  }

  kernel_function kernel;
  if (kernel_func == LAPLACE) {
    kernel = [&](const std::vector<double>& c_row,
                 const std::vector<double>& c_col) {
      return laplace_kernel(c_row, c_col, add_diag);
    };
  }

  bool is_symmetric = false;
  if (kernel_func == LAPLACE) {
    is_symmetric = true;
  }

  auto start_domain = std::chrono::system_clock::now();
  Domain domain(N, ndim);
  if (kind_of_geometry == GRID) {
    domain.generate_grid_particles();
  }
  else if (kind_of_geometry == CIRCULAR) {
    domain.generate_circular_particles(0, N);
  }
  domain.divide_domain_and_create_particle_boxes(nleaf);
  auto stop_domain = std::chrono::system_clock::now();
  double domain_time = std::chrono::duration_cast<
    std::chrono::milliseconds>(stop_domain - start_domain).count();

  auto start_construct = std::chrono::system_clock::now();
  SharedBasisMatrix A(N,
                      nleaf,
                      rank,
                      acc,
                      admis,
                      admis_kind,
                      construct_algorithm,
                      use_nested_basis,
                      domain,
                      kernel,
                      is_symmetric);
  auto stop_construct = std::chrono::system_clock::now();
  double construct_time = std::chrono::duration_cast<
    std::chrono::milliseconds>(stop_construct - start_construct).count();

  double construct_error;

  auto start_check = std::chrono::system_clock::now();
  Matrix x = generate_range_matrix(N, 1, 0);
  Matrix b = A.matvec(x);
  Matrix Adense = Hatrix::generate_p2p_matrix(domain, kernel);
  Matrix bdense = matmul(Adense, x);
  construct_error = Hatrix::norm(b - bdense) / Hatrix::norm(bdense);
  auto stop_check = std::chrono::system_clock::now();
  double check_time = std::chrono::duration_cast<
    std::chrono::milliseconds>(stop_check - start_check).count();

  std::cout << "-------------------------------\n";
  std::cout << "N               : " << N << std::endl;
  if (rank < 0) {
    std::cout << "ACCURACY      : " << acc << std::endl;
    std::cout << "MAX RANK      : " << A.max_rank() << std::endl;
  }
  else {
    std::cout << "RANK            : " << rank << std::endl;
  }
  std::cout << "NLEAF           : " << nleaf << "\n"
            << "Domain(ms)      : " << domain_time << "\n"
            << "Contruct(ms)    : " << construct_time << "\n"
            << "Construct error : " << construct_error << std::endl;
  std::cout << "-------------------------------\n";

  Hatrix::Context::finalize();
  return 0;
}
