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


class BLR_2x2 {
 private:
  // BLR stored in set of maps
  std::unordered_map<std::tuple<int64_t, int64_t>, Hatrix::Matrix> D_;
  std::unordered_map<std::tuple<int64_t, int64_t>, Hatrix::Matrix> S_;
  std::unordered_map<int64_t, Hatrix::Matrix> U_;
  std::unordered_map<int64_t, Hatrix::Matrix> V_;
 public:
  void insert_S(int64_t row, int64_t col, Hatrix::Matrix&& S) {
    S_.emplace(std::make_tuple(row, col), std::move(S));
  }
  Hatrix::Matrix& S(int64_t row, int64_t col) { return S_.at({row, col}); }
  const Hatrix::Matrix& S(int64_t row, int64_t col) const {
    return S_.at({row ,col});
  }

  void insert_D(int64_t row, int64_t col, Hatrix::Matrix&& D) {
    D_.emplace(std::make_tuple(row, col), std::move(D));
  }
  Hatrix::Matrix& D(int64_t row, int64_t col) { return D_.at({row, col}); }
  const Hatrix::Matrix& D(int64_t row, int64_t col) const {
    return D_.at({row ,col});
  }

  void insert_U(int64_t row, Hatrix::Matrix&& U) {
    U_.emplace(row, std::move(U));
  }
  Hatrix::Matrix& U(int64_t row) { return U_.at(row); }
  const Hatrix::Matrix& U(int64_t row) const { return U_.at(row); }

  void insert_V(int64_t col, Hatrix::Matrix&& V) {
    V_.emplace(col, std::move(V));
  }
  Hatrix::Matrix& V(int64_t col) { return V_.at(col); }
  const Hatrix::Matrix& V(int64_t col) const { return V_.at(col); }
};

BLR_2x2 construct_2x2_BLR(int64_t N, int64_t rank) {
  BLR_2x2 A;
  // Also store expected errors to check against later
  std::unordered_map<std::tuple<int64_t, int64_t>, double> expected_err;
  for (int64_t i=0; i<2; ++i) for (int64_t j=0; j<2; ++j) {
    if (i == j) {
      Hatrix::Matrix diag = Hatrix::generate_random_matrix(N, N);
      // Prevent pivoting
      for (int64_t i=0; i<diag.min_dim(); ++i) diag(i, i) += 2;
      A.insert_D(i, j, std::move(diag));
    } else {
      A.insert_D(i, j, Hatrix::generate_low_rank_matrix(N, N));
      A.insert_S(i, j, Hatrix::Matrix(N, N));
      A.insert_U(i, Hatrix::Matrix(N, N));
      A.insert_V(j, Hatrix::Matrix(N, N));
      // Make copy so we can compare norms later
      Hatrix::Matrix A_work(A.D(i, j));
      expected_err[{i, j}] = Hatrix::truncated_svd(
        A_work, A.U(i), A.S(i, j), A.V(j), rank
      );
    }
  }

  double error = 0, expected = 0;
  for (int64_t i=0; i<2; ++i) for (int64_t j=0; j<2; ++j) {
    if (i == j) {
      // TODO: Check something for dense blocks?
      continue;
    } else {
      error += Hatrix::frobenius_norm_diff(
        A.U(i) * A.S(i, j) * A.V(j), A.D(i, j)
      );
      expected += expected_err[{i, j}];
    }
  }
  std::cout << "Construction error: " << error << "  (expected: ";
  std::cout << expected << ")\n\n";
  return A;
}

void factorize_2x2_BLR(BLR_2x2& A, BLR_2x2& L, BLR_2x2& U) {
  BLR_2x2 A_check(A);

  // Initialize dense diagonal blocks of L and U
  L.insert_D(0, 0, Hatrix::Matrix(A.D(0, 0).rows, A.D(0, 0).cols));
  U.insert_D(0, 0, Hatrix::Matrix(A.D(0, 0).rows, A.D(0, 0).cols));
  L.insert_D(1, 1, Hatrix::Matrix(A.D(1, 1).rows, A.D(1, 1).cols));
  U.insert_D(1, 1, Hatrix::Matrix(A.D(1, 1).rows, A.D(1, 1).cols));
  // Move over off-diagonal blocks from A
  L.insert_U(1, std::move(A.U(1)));
  L.insert_S(1, 0, std::move(A.S(1, 0)));
  L.insert_V(0, std::move(A.V(0)));
  U.insert_U(0, std::move(A.U(0)));
  U.insert_S(0, 1, std::move(A.S(0, 1)));
  U.insert_V(1, std::move(A.V(1)));

  // Factorize input A into L and U
  // LU of top left
  Hatrix::lu(A.D(0, 0), L.D(0, 0), U.D(0, 0));
  // TRSMs
  Hatrix::solve_triangular(
    L.D(0, 0), U.U(0), Hatrix::Left, Hatrix::Lower, true
  );
  Hatrix::solve_triangular(
    U.D(0, 0), L.V(0), Hatrix::Right, Hatrix::Upper, false
  );
  // Schur complement into bottom right
  Hatrix::Matrix SxVxUxSxV = (
    (L.S(1, 0) * (L.V(0) * U.U(0)) * U.S(0, 1)) * U.V(1)
  );
  // Use matmul here for fused-multiply-add
  Hatrix::matmul(L.U(1), SxVxUxSxV, A.D(1, 1), false, false, -1, 1);
  // LU of bottom right
  Hatrix::lu(A.D(1, 1), L.D(1, 1), U.D(1, 1));
  // Factorization finished

  // Check result by multiplying L and U and comparing with the copy we made
  std::cout << "Factorization errors: \n";
  double top_left_diff = Hatrix::frobenius_norm_diff(
    L.D(0, 0) * U.D(0, 0), A_check.D(0, 0)
  );
  std::cout << "Top left error: " << top_left_diff << "\n";

  double top_right_diff = Hatrix::frobenius_norm_diff(
    L.D(0, 0) * U.U(0) * U.S(0, 1) * U.V(1), A_check.D(0, 1)
  );
  std::cout << "Top right error: " << top_right_diff << "\n";

  double bottom_left_diff = Hatrix::frobenius_norm_diff(
    L.U(1) * L.S(1, 0) * L.V(0) * U.D(0, 0), A_check.D(1, 0)
  );
  std::cout << "Bottom left error: " << bottom_left_diff << "\n";

  double bottom_right = Hatrix::frobenius_norm_diff(
    L.U(1) * L.S(1, 0) * L.V(0) * U.U(0) * U.S(0, 1) * U.V(1)
    + L.D(1, 1) * U.D(1, 1),
    A_check.D(1, 1)
  );
  std::cout << "Bottom right error: " << bottom_right << "\n\n";
}

void solve_2x2_BLR(
  const BLR_2x2& L, const BLR_2x2& U,
  Hatrix::Matrix& x0, Hatrix::Matrix& x1,
  const Hatrix::Matrix& b0, const Hatrix::Matrix& b1
) {
  // Forward substitution
  Hatrix::solve_triangular(L.D(0, 0), x0, Hatrix::Left, Hatrix::Lower, true);
  Hatrix::matmul(L.U(1) * L.S(1, 0) * L.V(0), x0, x1, false, false, -1, 1);
  Hatrix::solve_triangular(L.D(1, 1), x1, Hatrix::Left, Hatrix::Lower, true);
  // Backward substitution
  Hatrix::solve_triangular(U.D(1, 1), x1, Hatrix::Left, Hatrix::Upper, false);
  Hatrix::matmul(U.U(0) * U.S(0, 1) * U.V(1), x1, x0, false, false, -1, 1);
  Hatrix::solve_triangular(U.D(0, 0), x0, Hatrix::Left, Hatrix::Upper, false);

  double error = (
    Hatrix::frobenius_norm_diff(b0, x0) + Hatrix::frobenius_norm_diff(b1, x1)
  );
  std::cout << "Solution error: " << error << "\n";
}


int main() {
  int64_t N = 16;
  int64_t rank = 6;

  // Build 2x2 BLR and check result
  BLR_2x2 A = construct_2x2_BLR(N, rank);

  // Apply 2x2 BLR to vector for later error checking
  Hatrix::Matrix b0 = Hatrix::generate_random_matrix(N, 1);
  Hatrix::Matrix b1 = Hatrix::generate_random_matrix(N, 1);
  Hatrix::Matrix x0 = A.D(0, 0) * b0 + A.U(0) * A.S(0, 1) * A.V(1) * b1;
  Hatrix::Matrix x1 = A.U(1) * A.S(1, 0) * A.V(0) * b0 + A.D(1, 1) * b1;

  // Factorize 2x2 BLR
  BLR_2x2 L, U;
  factorize_2x2_BLR(A, L, U);

  solve_2x2_BLR(L, U, x0, x1, b0, b1);

  return 0;
}
