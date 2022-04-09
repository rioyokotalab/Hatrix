#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <chrono>
#include <fstream>

#include "Hatrix/Hatrix.h"

int main(int argc, char* argv[]) {
  Hatrix::Context::init();

  Hatrix::Context::finalize();

  return 0;
}
