#include <iostream>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <cmath>

#include "Hatrix/Hatrix.h"

// Generate an HSS matrix represented by a 3 level tree. There is a layer of leaf nodes
// at the bottom most level and another layer of transfer matrices above it.

std::vector<double> equally_spaced_vector(int N, double minVal, double maxVal) {
  std::vector<double> res(N, 0.0);
  double rnge = maxVal - minVal;
  for(int i=0; i<N; i++) {
    res[i] = minVal + ((double)i/(double)rnge);
  }
  return res;
}

void generate_leaf_nodes(Hatrix::HSS& A, int N, int rank, int height) {
  int level = height - 1;
  int nleaf_nodes = pow(2, level);
  for (int node = 0; node < nleaf_nodes; ++node) {
    // Generate diagonal dense blocks

  }
}

Hatrix::HSS construct_HSS_matrix(int N, int rank, int height) {
  Hatrix::HSS A;

  std::vector<std::vector<double> > randvec;
  randvec.push_back(equally_spaced_vector(N, 0.0, 1.0));
  generate_leaf_nodes(A, N, rank, height);
  return A;
}

Hatrix::Matrix construct_dense_matrix(int N) {
  std::vector<std::vector<double> > randvec;
  randvec.push_back(equally_spaced_vector(N, 0.0, 1.0));
  return Hatrix::generate_laplacend_matrix(randvec, N, N, 0, 0);
}

void verify_hss_construction(Hatrix::HSS& A, Hatrix::Matrix& A_dense) {

}

int main(int argc, char *argv[]) {
  int64_t N = atoi(argv[1]);
  int64_t rank = atoi(argv[2]);
  int64_t height = atoi(argv[3]);

  Hatrix::Context::init();
  Hatrix::Matrix b = Hatrix::generate_random_matrix(N, 1);
  Hatrix::HSS A = construct_HSS_matrix(N, rank, height);
  Hatrix::Matrix A_dense = construct_dense_matrix(N);

  verify_hss_construction(A, A_dense);

  Hatrix::Context::finalize();
}
