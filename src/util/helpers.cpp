#include "Hatrix/util/matrix_generators.h"

#include <vector>

namespace Hatrix {
  std::vector<double> equally_spaced_vector(int N, double minVal, double maxVal) {
    std::vector<double> res(N, 0.0);
    double rnge = maxVal - minVal;
    for(int i=0; i<N; i++) {
      res[i] = minVal + ((double)i/(double)rnge);
    }
    return res;
  }
}
