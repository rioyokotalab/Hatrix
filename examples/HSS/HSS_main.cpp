#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
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


int main(int argc, char* argv[]) {
  Hatrix::Context::init();
  std::vector<std::string> cmd_options;

  for (int i = 0; i < argc; ++i) {
    cmd_options.push_back(std::string(argv[i]));
  }
  const int width = 40;
  const char separator = '*';

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
              << " --admis :: Specify the admissibility constant. Double precision number >= 0." << std::endl
              << " --admis_kind :: Specify whether geometry-based admissiblity or diagonal-based: " << std::endl
              << "    geometry :: Admissiblity condition based on the geometry of the problem." << std::endl
              << "    diagonal :: Admissibility condition based on the distance from the diagonal."
              << std::endl;
  }

  Hatrix::Context::finalize();
  return 0;
}
