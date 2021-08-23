#include <iostream>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "Hatrix/Hatrix.h"

void check_HSS_matrix_generation(Hatrix::HSS& hss) {

}

int main(int argc, char *argv[]) {
  int64_t N = atoi(argv[1]);
  int64_t rank = atoi(argv[2]);
  int64_t height = atoi(argv[3]);

  Hatrix::Context::init();
  Hatrix::Matrix b = Hatrix::generate_random_matrix(N, 1);
  Hatrix::HSS A(N, rank, height);

  Hatrix::Context::finalize();
}
