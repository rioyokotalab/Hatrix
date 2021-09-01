#include <vector>

#include "Hatrix/Hatrix.h"

using randvec_t = std::vector<std::vector<double> >;

namespace Hatrix {
  class HSS {
    RowLevelMap U, V;
    RowColLevelMap D, S;
  };
}

std::vector<double> equally_spaced_vector(int N, double minVal, double maxVal) {
  std::vector<double> res(N, 0.0);
  double rnge = maxVal - minVal;
  for(int i=0; i<N; i++) {
    res[i] = minVal + ((double)i/(double)rnge);
  }
  return res;
}

int main(int argc, char *argv[]) {
  int N = atoi(argv[1]);
  int rank = atoi(argv[2]);
  int levels = 2;

  Hatrix::Context::init();
  randvec_t randvec;
  randvec.push_back(equally_spaced_vector(N, 0.0, 1.0)); // 1D


  Hatrix::Context::finalize();

}
