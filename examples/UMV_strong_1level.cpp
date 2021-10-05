#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Hatrix/Hatrix.h"

Hatrix::BLR construct_BLR(int64_t block_size, int64_t n_blocks, int64_t rank) {
  Hatrix::BLR A;
  for (int i = 0; i < n_blocks; ++i) {
    for (int j = 0; j < n_blocks; ++j) {
      if (i == j) {
        Hatrix::Matrix diag =
            Hatrix::generate_random_matrix(block_size, block_size);
        // Prevent pivoting
        for (int64_t i = 0; i < diag.min_dim(); ++i) diag(i, i) += 15;
        A.D.insert(i, j, std::move(diag));
      } else if (std::abs(i - j) == 1) {
        A.D.insert(i, j,
                   Hatrix::generate_random_matrix(block_size, block_size));
      } else {
        A.D.insert(i, j,
                   Hatrix::generate_low_rank_matrix(block_size, block_size));
      }
    }
  }
  // Also store expected errors to check against later
  std::unordered_map<std::tuple<int64_t, int64_t>, double> expected_err;
  int64_t oversampling = 5;
  Hatrix::Matrix U, S, V;
  double error;
  std::vector<Hatrix::Matrix> Y;
  for (int64_t i = 0; i < n_blocks; ++i) {
    Y.push_back(
        Hatrix::generate_random_matrix(block_size, rank + oversampling));
  }
  for (int64_t i = 0; i < n_blocks; ++i) {
    Hatrix::Matrix AY(block_size, rank + oversampling);
    for (int64_t j = 0; j < n_blocks; ++j) {
      if (std::abs(i - j) <= 1) continue;
      Hatrix::matmul(A.D(i, j), Y[j], AY);
    }
    std::tie(U, S, V, error) = Hatrix::truncated_svd(AY, rank);
    A.U.insert(i, std::move(U));
  }
  for (int64_t j = 0; j < n_blocks; ++j) {
    Hatrix::Matrix YtA(rank + oversampling, block_size);
    for (int64_t i = 0; i < n_blocks; ++i) {
      if (std::abs(i - j) <= 1) continue;
      Hatrix::matmul(Y[i], A.D(i, j), YtA, true);
    }
    std::tie(U, S, V, error) = Hatrix::truncated_svd(YtA, rank);
    A.V.insert(j, std::move(V));
  }

  double norm_diff = 0;
  double norm = 0;
  for (int64_t i = 0; i < n_blocks; ++i) {
    for (int64_t j = 0; j < n_blocks; ++j) {
      norm += std::pow(Hatrix::norm(A.D(i, j)), 2);
      if (std::abs(i - j) <= 1)
        continue;
      else {
        A.S.insert(i, j,
                   Hatrix::matmul(Hatrix::matmul(A.U[i], A.D(i, j), true),
                                  A.V[j], false, true));
        norm_diff +=
            std::pow(Hatrix::norm(A.U[i] * A.S(i, j) * A.V[j] - A.D(i, j)), 2);
        // std::cout << i << ", " << j << ": "
        //           << std::pow(
        //                  Hatrix::norm(A.U[i] * A.S(i, j) * A.V[j] - A.D(i,
        //                  j)), 2)
        //           << "\n";
      }
    }
  }
  double l2_error = std::sqrt(norm_diff / norm);
  std::cout << "Total construction L2 error: " << l2_error << "\n";
  return A;
}

double BLR_error()


int main() {
  int64_t block_size = 32;
  int64_t n_blocks = 8;
  int64_t rank = 12;
  bool multiply_compressed = false;
  Hatrix::BLR A = construct_BLR(block_size, n_blocks, rank);

  std::cout << "Solution error: " << error << "\n";
}
