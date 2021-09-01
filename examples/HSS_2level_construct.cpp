#include <vector>
#include <cmath>

#include "Hatrix/Hatrix.h"

using randvec_t = std::vector<std::vector<double> >;

namespace Hatrix {
  class HSS {
    ColLevelMap U;
    RowLevelMap V;
    RowColLevelMap D, S;
    int N, rank, height;

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

    Matrix generate_coupling_matrix(int row, int col, int level) {
      Matrix S(rank, rank);

      return S;
    }

    std::tuple<RowLevelMap, ColLevelMap> generate_leaf_nodes(randvec_t& randvec) {
      int nblocks = pow(height, 2);
      int leaf_size = N / nblocks;
      ColLevelMap Ugen;
      RowLevelMap Vgen;

      for (int block = 0; block < nblocks; ++block) {
        D.insert(block, block, height,
                 Hatrix::generate_laplacend_matrix(randvec, leaf_size, leaf_size,
                                                   block * leaf_size, block * leaf_size));
        Matrix U_temp, Ugen_temp;
        std::tie(U_temp, Ugen_temp) = generate_column_bases(block, leaf_size, randvec);
        U.insert(block, height, std::move(U_temp));
        Ugen.insert(block, height, std::move(Ugen_temp));

        Matrix V_temp, Vgen_temp;
        std::tie(V_temp, Vgen_temp) = generate_row_bases(block, leaf_size, randvec);
        V.insert(block, height, std::move(V_temp));
        Vgen.insert(block, height, std::move(Vgen_temp));

        int s_col = block % 2 == 0 ? block + 1 : block - 1;
        S.insert(block, s_col, height, generate_coupling_matrix(block, s_col, height));
      }

      return {Ugen, Vgen};
    }

  public:

    HSS(randvec_t& randpts, int _N, int _rank, int _height) :
      N(_N), rank(_rank), height(_height) {
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
  int height = 2;

  Hatrix::Context::init();
  randvec_t randvec;
  randvec.push_back(equally_spaced_vector(N, 0.0, 1.0)); // 1D

  Hatrix::HSS(randvec, N, rank, height);

  Hatrix::Context::finalize();

}
