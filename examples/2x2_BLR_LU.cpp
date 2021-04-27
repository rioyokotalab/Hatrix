#include "Hatrix/Hatrix.h"

#include <cstdint>
using std::int64_t;
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <utility>


namespace std {

template<>
struct hash<std::tuple<int64_t, int64_t>> {
  size_t operator()(const std::tuple<int64_t, int64_t>& pair) const {
    int64_t first, second;
    std::tie(first, second) = pair;
    size_t first_hash = hash<int64_t>()(first);
    first_hash ^= (
      hash<int64_t>()(second) + 0x9e3779b97f4a7c17
       + (first_hash << 6) + (first_hash >> 2)
    );
    return first_hash;
  }
};

} // namespace std


struct BLR_2x2 {
  // BLR stored in set of maps
  std::unordered_map<std::tuple<int64_t, int64_t>, Hatrix::Matrix> A;
  std::unordered_map<std::tuple<int64_t, int64_t>, Hatrix::Matrix> S;
  std::unordered_map<int64_t, Hatrix::Matrix> U;
  std::unordered_map<int64_t, Hatrix::Matrix> V;
};

BLR_2x2 construct_2x2_BLR(int64_t N, int64_t rank) {
  BLR_2x2 blr;
  // Also store tolerances to check against later
  std::unordered_map<std::tuple<int64_t, int64_t>, double> tolerances;
  std::unordered_map<std::tuple<int64_t, int64_t>, Hatrix::Matrix> A_lr_blocks;
  for (int64_t i=0; i<2; ++i) for (int64_t j=0; j<2; ++j) {
    if (i == j) {
      blr.A[{i, j}] = Hatrix::generate_random_matrix(N, N);
    } else {
      A_lr_blocks[{i, j}] = Hatrix::generate_low_rank_matrix(N, N);
      blr.S[{i, j}] = Hatrix::Matrix(N, N);
      blr.U[i] = Hatrix::Matrix(N, N);
      blr.V[j] = Hatrix::Matrix(N, N);
      // Make copy so we can compare norms later
      Hatrix::Matrix A_work(A_lr_blocks[{i, j}]);
      tolerances[{i, j}] = Hatrix::truncated_svd(
        A_work, blr.U[i], blr.S[{i, j}], blr.V[j], rank
      );
    }
  }

  for (int64_t i=0; i<2; ++i) for (int64_t j=0; j<2; ++j) {
    if (i == j) {
      // TODO: Check something for dense blocks?
      continue;
    } else {
      double norm_diff = Hatrix::frobenius_norm_diff(
        blr.U[i] * blr.S[{i, j}] * blr.V[j], A_lr_blocks[{i, j}]
      );
      std::cout << tolerances[{i, j}] << " = " << norm_diff << " ?\n";
    }
  }
  return blr;
}


int main() {
  int64_t N = 16;
  int64_t rank = 6;

  // Build 2x2 BLR and check result
  BLR_2x2 blr = construct_2x2_BLR(N, rank);

  return 0;
}
