#include <vector>
#include <cmath>

#include "Hatrix/Hatrix.h"

using randvec_t = std::vector<std::vector<double> >;

namespace Hatrix {
  class HSS {
    ColLevelMap U;
    RowLevelMap V;
    RowColLevelMap D, S;
    int N, rank, levels;

    std::tuple<Matrix, Matrix> generate_column_bases(int block, int leaf_size, randvec_t& randvec) {
      Matrix U_temp(leaf_size, rank);
      Matrix Ugen_temp(rank, N - leaf_size);

      return {U_temp, Ugen_temp};
    }

    std::tuple<Matrix, Matrix> generate_row_bases(int block, int leaf_size, randvec_t& randvec) {
      Matrix V_temp(leaf_size, rank);
      Matrix Vgen_temp(rank, N - leaf_size);

      return {V_temp, Vgen_temp};
    }

    std::tuple<RowLevelMap, ColLevelMap> generate_leaf_nodes(randvec_t& randvec) {
      int nblocks = pow(levels, 2);
      int leaf_size = N / nblocks;
      ColLevelMap Ugen;
      RowLevelMap Vgen;

      for (int block = 0; block < nblocks; ++block) {
        D.insert(block, block, levels,
                 Hatrix::generate_laplacend_matrix(randvec, leaf_size, leaf_size,
                                                   block * leaf_size, block * leaf_size));
        Matrix U_temp, Ugen_temp;
        std::tie(U_temp, Ugen_temp) = generate_column_bases(block, leaf_size, randvec);
        U.insert(block, levels, std::move(U_temp));
        Ugen.insert(block, levels, std::move(Ugen_temp));

        Matrix V_temp, Vgen_temp;
        std::tie(V_temp, Vgen_temp) = generate_row_bases(block, leaf_size, randvec);
        V.insert(block, levels, std::move(V_temp));
        Vgen.insert(block, levels, std::move(Vgen_temp));
      }

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {

        }
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
