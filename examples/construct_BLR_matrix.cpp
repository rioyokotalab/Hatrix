#include "Hatrix/Hatrix.h"

#include <cstdint>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <iostream>


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


int main() {
  int64_t N = 16;
  int64_t rank = 6;

  // BLR stored in set of maps
  std::unordered_map<std::tuple<int64_t, int64_t>, Hatrix::Matrix> A;
  std::unordered_map<std::tuple<int64_t, int64_t>, Hatrix::Matrix> S;
  std::unordered_map<int64_t, Hatrix::Matrix> U;
  std::unordered_map<int64_t, Hatrix::Matrix> V;

  // Also store tolerances to check against later
  std::unordered_map<std::tuple<int64_t, int64_t>, double> tolerances;
  for (int64_t i=0; i<2; ++i) for (int64_t j=0; j<2; ++j) {
    if (i == j) {
      A[{i, j}] = Hatrix::generate_random_matrix(N, N);
    } else {
      A[{i, j}] = Hatrix::generate_low_rank_matrix(N, N);
      S[{i, j}] = Hatrix::Matrix(N, N);
      U[i] = Hatrix::Matrix(N, N);
      V[j] = Hatrix::Matrix(N, N);
      // Make copy so we can compare norms later
      Hatrix::Matrix A_work(A[{i, j}]);
      tolerances[{i, j}] = Hatrix::truncated_svd(
        A_work, U[i], S[{i, j}], V[j], rank
      );
    }
  }

  for (int64_t i=0; i<2; ++i) for (int64_t j=0; j<2; ++j) {
    if (i == j) {
      // TODO: Check something for dense blocks?
      continue;
    } else {
      double norm_diff = Hatrix::frobenius_norm_diff(
        U[i] * S[{i, j}] * V[j], A[{i, j}]);
      std::cout << tolerances[{i, j}] << " = " << norm_diff << " ?\n";
    }
  }

  return 0;
}
