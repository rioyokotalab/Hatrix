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
  Args opts(argc, argv);

  init_elses_state();

  int m = 7680;
  double val;

  // for (long int i = 1; i <= 7680; ++i) {
  //   for (long int j = 1; j <= 7680; ++j) {
  //     get_elses_matrix_value(&i, &j, &val);
  //     std::cout << "i: " << i << " j: " << j << " "  << val << std::endl;
  //   }
  // }

  return 0;
}
