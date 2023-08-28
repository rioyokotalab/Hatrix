#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Hatrix/Hatrix.hpp"

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

std::vector<Hatrix::Matrix> multiply_BLR(const Hatrix::BLR& A,
                                         const std::vector<Hatrix::Matrix>& x,
                                         int64_t n_blocks,
                                         bool multiply_compressed) {
  std::vector<Hatrix::Matrix> b(n_blocks);
  if (multiply_compressed) {
    std::vector<Hatrix::Matrix> Vx;
    for (int64_t i = 0; i < n_blocks; ++i) {
      Vx.push_back(A.V[i] * x[i]);
    }
    for (int64_t i = 0; i < n_blocks; ++i) {
      b[i] = A.D(i, i) * x[i];
      Hatrix::Matrix SVx_sum(A.S(i, 0).rows, 1);
      for (int64_t j = 0; j < n_blocks; ++j) {
        if (i == j)
          continue;
        else if (std::abs(i - j) == 1)
          Hatrix::matmul(A.D(i, j), x[j], b[i]);
        else
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
                   int64_t n_blocks, int64_t rank) {
  Hatrix::BLR A_check(A);

  for (int64_t diag = 0; diag < n_blocks; ++diag) {
    // Initialize diagonal blocks of L, U
    Hatrix::Matrix& A_diag = A.D(diag, diag);
    L.D.insert(diag, diag, Hatrix::Matrix(A_diag.rows, A_diag.cols));
    U.D.insert(diag, diag, Hatrix::Matrix(A_diag.rows, A_diag.cols));
    // Move over the off-diagonal unless on last diagonal block
    if (diag < n_blocks - 1) {
      L.D.insert(diag + 1, diag, A.D.extract(diag + 1, diag));
      U.D.insert(diag, diag + 1, A.D.extract(diag, diag + 1));
    }
    // Initialize off-diagonal blocks of L, U
    // Copy basis to L
    L.U.insert(diag, Hatrix::Matrix(A.U[diag]));
    for (int64_t i_c = diag + 2; i_c < n_blocks; ++i_c) {
      L.S.insert(i_c, diag, A.S.extract(i_c, diag));
    }
    L.V.insert(diag, Hatrix::Matrix(A.V[diag]));
    // Copy basis to U (using extract instead of copy)
    U.U.insert(diag, A.U.extract(diag));
    for (int64_t j = diag + 2; j < n_blocks; ++j) {
      U.S.insert(diag, j, A.S.extract(diag, j));
    }
    U.V.insert(diag, A.V.extract(diag));

    // Left looking LU
    // Schur complement for diagonal and LU
    for (int64_t i = 0; i < diag; ++i) {
      if (i == diag - 1)
        Hatrix::matmul(L.D(diag, i), U.D(i, diag), A_diag, false, false, -1, 1);
      else
        Hatrix::matmul(
            L.U[diag],
            L.S(diag, i) * L.V[i] * U.U[i] * U.S(i, diag) * U.V[diag], A_diag,
            false, false, -1, 1);
    }
    Hatrix::lu(A_diag, L.D(diag, diag), U.D(diag, diag));
    // Schur complement for off-diagonal and triangular solve
    if (diag < n_blocks - 1) {
      if (diag > 0) {
        for (int64_t i = 0; i < diag; ++i) {
          if (i == diag - 1) {
            Hatrix::matmul(L.D(diag, i) * U.U[i] * U.S(i, diag + 1),
                           A.V[diag + 1], U.D(diag, diag + 1), false, false, -1,
                           1);
            Hatrix::matmul(A.U[diag + 1],
                           L.S(diag + 1, i) * L.V[i] * U.D(i, diag),
                           L.D(diag + 1, diag), false, false, -1, 1);
          } else {
            Hatrix::matmul(L.U[diag],
                           L.S(diag, i) * L.V[i] * U.U[i] * U.S(i, diag + 1) *
                               A.V[diag + 1],
                           U.D(diag, diag + 1), false, false, -1, 1);
            Hatrix::matmul(A.U[diag + 1] * L.S(diag + 1, i) * L.V[i] * U.U[i] *
                               U.S(i, diag),
                           U.V[diag], L.D(diag + 1, diag), false, false, -1, 1);
          }
        }
      }
      Hatrix::solve_triangular(L.D(diag, diag), U.D(diag, diag + 1),
                               Hatrix::Left, Hatrix::Lower, true);
      Hatrix::solve_triangular(U.D(diag, diag), L.D(diag + 1, diag),
                               Hatrix::Right, Hatrix::Upper, false);
    }
    // Schur complement for far off-diagonal, recompression and triangular solve
    if (diag < n_blocks - 2) {
      // Simple shared basis low-rank addition
      for (int64_t i = 0; i < diag - 1; ++i) {
        Hatrix::Matrix VU = L.V[i] * U.U[i];
        for (int64_t i_c = diag + 2; i_c < n_blocks; ++i_c) {
          Hatrix::matmul(L.S(i_c, i), VU * U.S(i, diag), L.S(i_c, diag), false,
                         false, -1, 1);
        }
        for (int64_t j = diag + 2; j < n_blocks; ++j) {
          Hatrix::matmul(L.S(diag, i) * VU, U.S(i, j), U.S(diag, j), false,
                         false, -1, 1);
        }
      }
      // Recompression
      if (diag > 0) {
        Hatrix::Matrix recompU, recompS, recompV;
        double error;
        // Column recompression for L
        Hatrix::Matrix VD = L.V[diag - 1] * U.D(diag - 1, diag);
        Hatrix::Matrix SVD((n_blocks - diag - 2) * rank, VD.cols);
        std::vector<Hatrix::Matrix> SVDsplit =
            SVD.split(n_blocks - diag - 2, 1);
        for (int64_t i = diag + 2; i < n_blocks; ++i) {
          Hatrix::matmul(L.S(i, diag), L.V[diag], SVDsplit[i - diag - 2], false,
                         false, 1, 0);
          Hatrix::matmul(L.S(i, diag - 1), VD, SVDsplit[i - diag - 2], false,
                         false, -1, 1);
        }
        std::tie(recompU, recompS, L.V[diag], error) =
            Hatrix::truncated_svd(SVD, rank);
        std::vector<Hatrix::Matrix> recompUsplit =
            recompU.split(n_blocks - diag - 2, 1);
        for (int64_t i = diag + 2; i < n_blocks; ++i) {
          Hatrix::matmul(recompUsplit[i - diag - 2], recompS, L.S(i, diag),
                         false, false, 1, 0);
        }

        // Row recompression for U
        Hatrix::Matrix DU = L.D(diag, diag - 1) * U.U[diag - 1];
        Hatrix::Matrix DUS(DU.rows, (n_blocks - diag - 2) * rank);
        std::vector<Hatrix::Matrix> DUSsplit =
            DUS.split(1, n_blocks - diag - 2);
        for (int64_t j = diag + 2; j < n_blocks; ++j) {
          Hatrix::matmul(U.U[diag], U.S(diag, j), DUSsplit[j - diag - 2], false,
                         false, 1, 0);
          Hatrix::matmul(DU, U.S(diag - 1, j), DUSsplit[j - diag - 2], false,
                         false, -1, 1);
        }
        std::tie(U.U[diag], recompS, recompV, error) =
            Hatrix::truncated_svd(DUS, rank);
        std::vector<Hatrix::Matrix> recompVsplit =
            recompV.split(1, n_blocks - diag - 2);
        for (int64_t j = diag + 2; j < n_blocks; ++j) {
          Hatrix::matmul(recompS, recompVsplit[j - diag - 2], U.S(diag, j),
                         false, false, 1, 0);
        }
      }
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
      if (i > j + 1)
        result = L.U[i] * L.S(i, j) * L.V[j] * U.D(j, j);
      else if (j > i + 1)
        result = L.D(i, i) * U.U[i] * U.S(i, j) * U.V[j];
      else
        result = L.D(i, std::min(i, j)) * U.D(std::min(i, j), j);
      for (int64_t k = 0; k < std::min(i, j); ++k) {
        // Off-diagonal dense
        if (k == std::min(i, j) - 1) {
          if (i > j)
            result += L.U[i] * L.S(i, k) * L.V[k] * U.D(k, j);
          else if (j > i)
            result += L.D(i, k) * U.U[k] * U.S(k, j) * U.V[j];
          else
            result += L.D(i, k) * U.D(k, j);
        } else {
          result += L.U[i] * L.S(i, k) * L.V[k] * U.U[k] * U.S(k, j) * U.V[j];
        }
      }
      double norm_diff = Hatrix::norm(result - A_check.D(i, j));
      std::cout << i << ", " << j << ": " << norm_diff << "\n";
      if (norm_diff > 0.005) {
        for (int i_c = 0; i_c < result.rows; ++i_c) {
          for (int j_c = 0; j_c < result.cols; ++j_c) {
            std::cout << A_check.D(i, j)(i_c, j_c) << "\t";
          }
          std::cout << "\n";
        }
        std::cout << "\n\n";
        for (int i_c = 0; i_c < result.rows; ++i_c) {
          for (int j_c = 0; j_c < result.cols; ++j_c) {
            std::cout << result(i_c, j_c) << "\t";
          }
          std::cout << "\n";
        }
        std::cout << "\n\n";
      }
      error += norm_diff;
    }
  }
  std::cout << "Total factorization error: " << error << "\n";
}

void solve_BLR(const Hatrix::BLR& L, const Hatrix::BLR& U,
               std::vector<Hatrix::Matrix>& b, int64_t n_blocks) {
  // Foward substitution
  for (int64_t i = 0; i < n_blocks; ++i) {
    for (int64_t j = 0; j < i; ++j) {
      if (j == i - 1) {
        Hatrix::matmul(L.D(i, j), b[j], b[i], false, false, -1, 1);
      } else {
        Hatrix::matmul(L.U[i] * L.S(i, j) * L.V[j], b[j], b[i], false, false,
                       -1, 1);
      }
    }
    Hatrix::solve_triangular(L.D(i, i), b[i], Hatrix::Left, Hatrix::Lower,
                             true);
  }
  // Backward substitution
  for (int64_t i = n_blocks - 1; i >= 0; --i) {
    for (int64_t j = n_blocks - 1; j > i; --j) {
      if (j == i + 1) {
        Hatrix::matmul(U.D(i, j), b[j], b[i], false, false, -1, 1);
      } else {
        Hatrix::matmul(U.U[i] * U.S(i, j) * U.V[j], b[j], b[i], false, false,
                       -1, 1);
      }
    }
    Hatrix::solve_triangular(U.D(i, i), b[i], Hatrix::Left, Hatrix::Upper,
                             false);
  }
}

int main() {
  int64_t block_size = 32;
  int64_t n_blocks = 8;
  int64_t rank = 12;
  bool multiply_compressed = false;
  Hatrix::BLR A = construct_BLR(block_size, n_blocks, rank);

  std::vector<Hatrix::Matrix> x;
  for (int64_t i = 0; i < n_blocks; ++i) {
    x.push_back(Hatrix::generate_random_matrix(block_size, 1));
  }

  std::vector<Hatrix::Matrix> b =
      multiply_BLR(A, x, n_blocks, multiply_compressed);

  Hatrix::BLR L, U;
  factorize_BLR(A, L, U, n_blocks, rank);

  solve_BLR(L, U, b, n_blocks);

  double error = 0;
  for (int64_t i = 0; i < n_blocks; ++i) {
    error += Hatrix::norm(b[i] - x[i]);
  }
  std::cout << "Solution error: " << error << "\n";

  return 0;
}
