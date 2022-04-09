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
#include "HSS.hpp"

using namespace Hatrix;

bool help_option_exists(std::vector<std::string>& cmd_options) {
  auto begin = cmd_options.begin();
  auto end = cmd_options.end();
  return (std::find(begin, end, "-h") != end) || (std::find(begin, end, "--help") != end);
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
    std::cout << " --N :: Specify problem size. Generates a NxN matrix. Must be integer." << std::endl
              << " --nleaf :: Specify leaf size. This is the maximum size of the tile in the lowest level of the tree. Alternatively maximum number of particles in the leaf node. Must be integer." << std::endl
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
              << " --admis_kind :: Specify whether geometry-based admissiblity or diagonal-based: " << std::endl
              << "    geometry :: Admissiblity condition based on the geometry of the problem." << std::endl
              << "    diagonal :: Admissibility condition based on the distance from the diagonal."
              << " --construct-algorithm :: Specify the kind of construction algorithm." << std::endl
              << "    miro       -- From the miro board. Use SVD everywhere." << std::endl
              << "    id_random  -- From Martinsson2010. Use ID + randomized sampling."
              << " --add-diag :: Specify the value to add to the diagonal."
              << std::endl;
  }

  int64_t N = 1000;
  int64_t nleaf = 100;
  KERNEL_FUNC kernel_func = LAPLACE;
  KIND_OF_GEOMETRY kind_of_geometry = GRID;
  int64_t ndim = 1;
  int64_t rank = 10;
  double admis = 1;
  double acc = -1;
  double add_diag = 0;
  ADMIS_KIND admis_kind = DIAGONAL;
  CONSTRUCT_ALGORITHM construct_algorithm = MIRO;

  for (auto iter = cmd_options.begin(); iter != cmd_options.end(); ++iter) {
    auto option = *iter;

    if (option == "--N") {
      N = std::stol(*(++iter));
    }
    else if (option == "--nleaf") {
      nleaf = std::stol(*(++iter));
    }
    else if (option == "--kind-of-problem") {
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
    else if (option == "--kind-of-geometry") {
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
    else if (option == "--ndim") {
      ndim = std::stol(*(++iter));
    }
    else if (option == "--rank") {
      rank = std::stol(*(++iter));
    }
    else if (option == "--acc") {
      acc = std::stod(*(++iter));
    }
    else if (option == "--admis") {
      admis = std::stod(*(++iter));
    }
    else if (option == "--admis-kind") {
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
    else if (option == "--construct-algorithm") {
      auto value = *(++iter);
      if (value == "miro") {
        construct_algorithm = MIRO;
      }
      else if (value == "id_random") {
        construct_algorithm = ID_RANDOM;
      }
      else {
        std::cout << "wrong value for --construct-algorithm " << value << std::endl;
        abort();
      }
    }
    else if (option == "--add-diag") {
      add_diag = std::stod(*(++iter));
    }
  }

  kernel_function kernel;
  if (kernel_func == LAPLACE) {
    kernel = [&](const std::vector<double>& c_row,
                 const std::vector<double>& c_col) {
      return laplace_kernel(c_row, c_col, add_diag);
    };
  }


  Hatrix::Context::finalize();
  return 0;
}
