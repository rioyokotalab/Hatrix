#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Hatrix/Hatrix.h"

using randvec_t = std::vector<std::vector<double> >;

namespace Hatrix {
  class H2 {
  private:
    ColLevelMap U;
    RowLevelMap V;
    RowColLevelMap<Matrix> D, S;
    RowColLevelMap<bool> is_admissible;

    int64_t N, rank, admis, height;

    void generate_transfer_matrices(const randvec_t& randvec, RowLevelMap& Ugen, ColLevelMap& Vgen) {
    }

    std::tuple<RowLevelMap, ColLevelMap> generate_leaf_nodes(const randvec_t& randpts) {
      int nblocks = pow(height, 2);
      int leaf_size = N / nblocks;
      ColLevelMap Ugen;
      RowLevelMap Vgen;

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          is_admissible.insert(i, j, height, std::abs(i - j) < admis);
          if (is_admissible(i, j, height)) {
            D.insert(i, j, height, Hatrix::generate_laplacend_matrix(randpts, leaf_size, leaf_size,
                                                                     i * leaf_size, j * leaf_size));
          }
        }
      }

      return {Ugen, Vgen};
    }

  public:

    H2(const randvec_t& randpts, int64_t N, int64_t rank, int64_t admis, int64_t height) :
      N(N), rank(rank), admis(admis), height(height) {
      RowLevelMap Ugen; ColLevelMap Vgen;

      std::tie(Ugen, Vgen) = generate_leaf_nodes(randpts);
      generate_transfer_matrices(randpts, Ugen, Vgen);
    }

    double construction_relative_error(const randvec_t& randpts) {
      double error = 0;

      return error;
    }
  };
}

int main(int argc, char* argv[]) {
  int64_t N = atoi(argv[1]);
  int64_t rank = atoi(argv[2]);
  int64_t admis = atoi(argv[3]);

  if (admis > 1) {
    std::cout << "This program only supports admis with 0 or 1.\n";
    abort();
  }

  Hatrix::Context::init();
  randvec_t randpts;
  randpts.push_back(Hatrix::equally_spaced_vector(N, 0.0, 1.0 * N)); // 1D
  randpts.push_back(Hatrix::equally_spaced_vector(N, 0.0, 1.0 * N)); // 2D
  randpts.push_back(Hatrix::equally_spaced_vector(N, 0.0, 1.0 * N)); // 3D

  Hatrix::H2 A(randpts, N, rank, admis, 2);
  double error = A.construction_relative_error(randpts);

  Hatrix::Context::finalize();

  std::cout << "N=" << N << " rank=" << rank << " admis="
            << admis <<  " construction error=" << error << std::endl;
}
