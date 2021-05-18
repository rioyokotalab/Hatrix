#include <algorithm>
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
        for (int64_t i = 0; i < diag.min_dim(); ++i) diag(i, i) += 10;
        A.D.insert(i, j, std::move(diag));
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
      if (i == j) continue;
      Hatrix::matmul(A.D(i, j), Y[j], AY);
    }
    std::tie(U, S, V, error) = Hatrix::truncated_svd(AY, rank);
    A.U.insert(i, std::move(U));
  }
  for (int64_t j = 0; j < n_blocks; ++j) {
    Hatrix::Matrix YtA(rank + oversampling, block_size);
    for (int64_t i = 0; i < n_blocks; ++i) {
      if (j == i) continue;
      Hatrix::matmul(Y[i], A.D(i, j), YtA, true);
    }
    std::tie(U, S, V, error) = Hatrix::truncated_svd(YtA, rank);
    A.V.insert(j, std::move(V));
  }

  error = 0;
  for (int i = 0; i < n_blocks; ++i) {
    for (int j = 0; j < n_blocks; ++j) {
      if (i == j)
        continue;
      else {
        A.S.insert(i, j,
                   Hatrix::matmul(Hatrix::matmul(A.U[i], A.D(i, j), true),
                                  A.V[j], false, true));
        error += Hatrix::norm_diff(A.U[i] * A.S(i, j) * A.V[j], A.D(i, j));
      }
    }
  }
  std::cout << "Total construction error: " << error << "\n";
  return A;
}

std::vector<Hatrix::Matrix> multiply_BLR(const Hatrix::BLR& A,
                                         const std::vector<Hatrix::Matrix>& x,
                                         int64_t n_blocks,
                                         bool compare_compressed) {
  std::vector<Hatrix::Matrix> b(n_blocks);
  if (compare_compressed) {
    std::vector<Hatrix::Matrix> Vx;
    for (int64_t i = 0; i < n_blocks; ++i) {
      Vx.push_back(A.V[i] * x[i]);
    }
    for (int64_t i = 0; i < n_blocks; ++i) {
      b[i] = A.D(i, i) * x[i];
      Hatrix::Matrix SVx_sum(A.S(i, 0).rows, 1);
      for (int64_t j = 0; j < n_blocks; ++j) {
        if (i == j) continue;
        Hatrix::matmul(A.S(i, j), Vx[j], SVx_sum);
      }
      Hatrix::matmul(A.U[i], SVx_sum, b[i]);
    }
  } else {
    for (int64_t i = 0; i < n_blocks; ++i) {
      b[i] = Hatrix::Matrix(x[0].rows, 1);
      for (int64_t j = 0; j < n_blocks; ++j) {
        Hatrix::matmul(A.D(i, j), x[j], b[i]);
      }
    }
  }
  return b;
}

void factorize_BLR(Hatrix::BLR& A, Hatrix::BLR& L, Hatrix::BLR& U,
                   int64_t n_blocks) {
  Hatrix::BLR A_check(A);

  for (int64_t diag = 0; diag < n_blocks; ++diag) {
    // Initialize diagonal blocks of L, U
    Hatrix::Matrix& A_diag = A.D(diag, diag);
    L.D.insert(diag, diag, Hatrix::Matrix(A_diag.rows, A_diag.cols));
    U.D.insert(diag, diag, Hatrix::Matrix(A_diag.rows, A_diag.cols));
    // Initialize off-diagonal blocks of L, U
    // Copy basis to L and U (using move instead of copy where possible)
    if (diag == 0) {
      L.V.insert(diag, std::move(A.V[diag]));
      U.U.insert(diag, std::move(A.U[diag]));
    } else if (diag == n_blocks - 1) {
      L.U.insert(diag, std::move(A.U[diag]));
      U.V.insert(diag, std::move(A.V[diag]));
    } else {
      L.U.insert(diag, Hatrix::Matrix(A.U[diag]));
      L.V.insert(diag, Hatrix::Matrix(A.V[diag]));
      U.U.insert(diag, std::move(A.U[diag]));
      U.V.insert(diag, std::move(A.V[diag]));
    }
    for (int64_t i_c = diag + 1; i_c < n_blocks; ++i_c) {
      L.S.insert(i_c, diag, std::move(A.S(i_c, diag)));
    }
    for (int64_t j = diag + 1; j < n_blocks; ++j) {
      U.S.insert(diag, j, std::move(A.S(diag, j)));
    }

    // Left looking LU
    // TODO We can be more efficient here by storing V*U somehow!
    for (int64_t i = 0; i < diag; ++i) {
      Hatrix::Matrix VU = L.V[i] * U.U[i];
      Hatrix::matmul(L.U[diag], L.S(diag, i) * VU * U.S(i, diag) * U.V[diag],
                     A_diag, false, false, -1, 1);
      for (int64_t i_c = diag + 1; i_c < n_blocks; ++i_c) {
        Hatrix::matmul(L.S(i_c, i), VU * U.S(i, diag), L.S(i_c, diag), false,
                       false, -1, 1);
      }
      for (int64_t j = diag + 1; j < n_blocks; ++j) {
        Hatrix::matmul(L.S(diag, i) * VU, U.S(i, j), U.S(diag, j), false, false,
                       -1, 1);
      }
    }
    Hatrix::lu(A_diag, L.D(diag, diag), U.D(diag, diag));
    if (diag < n_blocks - 1) {
      Hatrix::solve_triangular(L.D(diag, diag), U.U[diag], Hatrix::Left,
                               Hatrix::Lower, true);
      Hatrix::solve_triangular(U.D(diag, diag), L.V[diag], Hatrix::Right,
                               Hatrix::Upper, false);
    }
  }

  double error = 0;
  for (int64_t i = 0; i < n_blocks; ++i) {
    for (int64_t j = 0; j < n_blocks; ++j) {
      Hatrix::Matrix result;
      if (i > j)
        result = L.U[i] * L.S(i, j) * L.V[j] * U.D(j, j);
      else if (j > i)
        result = L.D(i, i) * U.U[i] * U.S(i, j) * U.V[j];
      else
        result = L.D(i, i) * U.D(i, i);
      for (int64_t k = 0; k < std::min(i, j); ++k) {
        result += L.U[i] * L.S(i, k) * L.V[k] * U.U[k] * U.S(k, j) * U.V[j];
      }
      error += Hatrix::norm_diff(result, A_check.D(i, j));
    }
  }
  std::cout << "Total factorization error: " << error << "\n";
}

void solve_BLR(const Hatrix::BLR& L, const Hatrix::BLR& U,
               std::vector<Hatrix::Matrix>& b, int64_t n_blocks) {
  // Foward substitution
  for (int64_t i = 0; i < n_blocks; ++i) {
    for (int64_t j = 0; j < i; ++j) {
      Hatrix::matmul(L.U[i] * L.S(i, j) * L.V[j], b[j], b[i], false, false, -1,
                     1);
    }
    Hatrix::solve_triangular(L.D(i, i), b[i], Hatrix::Left, Hatrix::Lower,
                             true);
  }
  // Backward substitution
  for (int64_t i = n_blocks - 1; i >= 0; --i) {
    for (int64_t j = n_blocks - 1; j > i; --j) {
      Hatrix::matmul(U.U[i] * U.S(i, j) * U.V[j], b[j], b[i], false, false, -1,
                     1);
    }
    Hatrix::solve_triangular(U.D(i, i), b[i], Hatrix::Left, Hatrix::Upper,
                             false);
  }
}

int main() {
  int64_t block_size = 32;
  int64_t n_blocks = 4;
  int64_t rank = 8;
  bool multiply_compressed = false;
  Hatrix::BLR A = construct_BLR(block_size, n_blocks, rank);

  std::vector<Hatrix::Matrix> x;
  for (int64_t i = 0; i < n_blocks; ++i) {
    x.push_back(Hatrix::generate_random_matrix(block_size, 1));
  }

  std::vector<Hatrix::Matrix> b =
      multiply_BLR(A, x, n_blocks, multiply_compressed);

  Hatrix::BLR L, U;
  factorize_BLR(A, L, U, n_blocks);

  solve_BLR(L, U, b, n_blocks);

  double error = 0;
  for (int64_t i = 0; i < n_blocks; ++i) {
    error += Hatrix::norm_diff(b[i], x[i]);
  }
  std::cout << "Solution error: " << error << "\n";
}
