#include <vector>
#include <cmath>

#include "Hatrix/Hatrix.h"

using randvec_t = std::vector<std::vector<double> >;

namespace Hatrix {
  class HSS {
    RowLevelMap U;
    ColLevelMap V;
    RowColLevelMap D, S;
    int N, rank, levels;

    std::tuple<RowLevelMap, ColLevelMap> generate_leaf_nodes(randvec_t& randvec) {
      int nblocks = pow(levels, 2);
      int leaf_size = N / nblocks;
      RowLevelMap Ugen, Vgen;

      for (int block = 0; block < nblocks; ++block) {
        D.insert(block, block, levels,
                 Hatrix::generate_laplacend_matrix(randvec, leaf_size, leaf_size,
                                                   block * leaf_size, block * leaf_size));
      }

      return {Ugen, Vgen};
    }

  public:

    HSS(randvec_t& randpts, int _N, int _rank, int _levels) :
      N(_N), rank(_rank), levels(_levels) {
      RowLevelMap Ugen;
      ColLevelMap Vgen;

      std::tie(Ugen, Vgen) = generate_leaf_nodes(randpts);

    }
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

  Hatrix::HSS(randvec, N, rank, levels);

  Hatrix::Context::finalize();

}
