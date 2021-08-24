#include <iostream>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <algorithm>
#include <cmath>

#include "Hatrix/Hatrix.h"

// Generate an HSS matrix represented by a 3 level tree. There is a layer of leaf nodes
// at the bottom most level and another layer of transfer matrices above it.

using randvec_t = std::vector<std::vector<double> >;

std::vector<double> equally_spaced_vector(int N, double minVal, double maxVal) {
  std::vector<double> res(N, 0.0);
  double rnge = maxVal - minVal;
  for(int i=0; i<N; i++) {
    res[i] = minVal + ((double)i/(double)rnge);
  }
  return res;
}

std::pair<Hatrix::RowLevelMap, Hatrix::RowLevelMap>
generate_leaf_nodes(
                    Hatrix::HSS& A, randvec_t& randvec, int N, int rank, int height) {
  int level = height - 1;
  int nleaf_nodes = pow(2, level);
  int block = std::floor(N/nleaf_nodes);
  Hatrix::RowLevelMap U_generators, V_generators;

  for (int node = 0; node < nleaf_nodes; ++node) {
    int leaf_size = node == nleaf_nodes - 1 ?
      std::min(block, N - block * (nleaf_nodes-1)) : block;

    // Generate diagonal dense blocks
    A.D.insert(node, node, level,
               Hatrix::generate_laplacend_matrix(randvec, leaf_size, leaf_size,
                                                 block * node, block * node));

  }

  return {U_generators, V_generators};
}

Hatrix::HSS construct_HSS_matrix(int N, int rank, int height, randvec_t& randvec) {
  Hatrix::HSS A;
  Hatrix::RowLevelMap U_generators, V_generators;
  std::tie(U_generators, V_generators) = generate_leaf_nodes(A, randvec, N, rank, height);
  return A;
}

Hatrix::Matrix construct_dense_matrix(int N, randvec_t& randvec) {
  return Hatrix::generate_laplacend_matrix(randvec, N, N, 0, 0);
}

void verify_hss_construction(Hatrix::HSS& A, Hatrix::Matrix& A_dense) {

}

int main(int argc, char *argv[]) {
  int N = atoi(argv[1]);
  int rank = atoi(argv[2]);
  int height = 3;

  Hatrix::Context::init();
  randvec_t randvec;
  randvec.push_back(equally_spaced_vector(N, 0.0, 1.0));

  Hatrix::Matrix b = Hatrix::generate_random_matrix(N, 1);
  Hatrix::HSS A = construct_HSS_matrix(N, rank, height, randvec);
  Hatrix::Matrix A_dense = construct_dense_matrix(N, randvec);

  verify_hss_construction(A, A_dense);

  Hatrix::Context::finalize();
}
