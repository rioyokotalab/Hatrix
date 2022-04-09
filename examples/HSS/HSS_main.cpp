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

bool help_option_exists(std::vector<std::string>& cmd_options) {
  auto begin = cmd_options.begin();
  auto end = cmd_options.end();
  return (std::find(begin, end, "-h") != end) || (std::find(begin, end, "--help") != end);
}

constexpr int INIT_VALUE = -1;
enum KIND_OF_PROBLEM {LAPLACE, SQR_EXP, SINE};
enum KIND_OF_GEOMETRY {GRID, CIRCULAR};
enum ADMIS_KIND {DIAGONAL, GEOMETRY};
enum CONSTRUCT_ALGORITHM {MIRO, ID_RANDOM};

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
              << " --kind-of-problem :: Specify the kind of problem: " << std::endl
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
              << std::endl;
  }

  long long N = 1000;
  long long nleaf = 100;
  KIND_OF_PROBLEM kind_of_problem = LAPLACE;
  KIND_OF_GEOMETRY kind_of_geometry = GRID;
  int ndim = 1;
  int rank = 10;
  double admis = 1;
  double acc = -1;
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
        kind_of_problem = LAPLACE;
      }
      else if (value == "sqrexp") {
        kind_of_problem = SQR_EXP;
      }
      else if (value == "sin") {
        kind_of_problem = SINE;
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
      ndim = std::stoi(*(++iter));
    }
    else if (option == "--rank") {
      rank = std::stoi(*(++iter));
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
  }

  std::cout << "N: " << N << " leaf: " << nleaf << std::endl;

  Hatrix::Context::finalize();
  return 0;
}
