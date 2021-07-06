#include <cstdint>
#include <cmath>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "Hatrix/Hatrix.h"

class BLR_2x2 {
 private:
  // BLR stored in set of maps
  /*
    ＊ ＊ ＊ ＊ ＊ | ＊ 　 ＊ ＊ ＊
    ＊ ＊ ＊ ＊ ＊ |
    ＊ ＊ ＊ ＊ ＊ | ＊
    ＊ ＊ ＊ ＊ ＊ | ＊
    ＊ ＊ ＊ ＊ ＊ | ＊
    ーーーーーーーー ーーーーーーーー
    ＊  　＊ ＊ ＊ | ＊ ＊ ＊ ＊ ＊
    　  　        | ＊ ＊ ＊ ＊ ＊
    ＊  　        | ＊ ＊ ＊ ＊ ＊
    ＊  　        | ＊ ＊ ＊ ＊ ＊
    ＊  　        | ＊ ＊ ＊ ＊ ＊
  */
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
    return S_.at({row, col});
  }

  void insert_D(int64_t row, int64_t col, Hatrix::Matrix&& D) {
    D_.emplace(std::make_tuple(row, col), std::move(D));
  }
  Hatrix::Matrix& D(int64_t row, int64_t col) { return D_.at({row, col}); }
  const Hatrix::Matrix& D(int64_t row, int64_t col) const {
    return D_.at({row, col});
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
  for (int64_t i = 0; i < 2; ++i) {
    for (int64_t j = 0; j < 2; ++j) {
      if (i == j) {
        Hatrix::Matrix diag = Hatrix::generate_random_matrix(N, N);
        // Prevent pivoting
        for (int64_t i = 0; i < diag.min_dim(); ++i) diag(i, i) += 2;
        A.insert_D(i, j, std::move(diag));
      } else {
        A.insert_D(i, j, Hatrix::generate_low_rank_matrix(N, N));
        A.insert_S(i, j, Hatrix::Matrix(N, N));
        A.insert_U(i, Hatrix::Matrix(N, N));
        A.insert_V(j, Hatrix::Matrix(N, N));
        // Make copy so we can compare norms later
        Hatrix::Matrix A_work(A.D(i, j));
        expected_err[{i, j}] =
            Hatrix::truncated_svd(A_work, A.U(i), A.S(i, j), A.V(j), rank);
      }
    }
  }

  double error = 0, expected = 0;
  for (int64_t i = 0; i < 2; ++i) {
    for (int64_t j = 0; j < 2; ++j) {
      if (i == j) {
        // TODO: Check something for dense blocks?
        continue;
      } else {
        error += Hatrix::norm_diff(A.U(i) * A.S(i, j) * A.V(j), A.D(i, j));
        expected += expected_err[{i, j}];
      }
    }
  }
  std::cout << "Construction error: " << error << "  (expected: ";
  std::cout << expected << ")\n\n";
  return A;
}

// Approximate multiplication of A and B in the bases of C
void matmul_2x2_BLR(BLR_2x2& A, BLR_2x2& B, BLR_2x2& C,
		    double alpha, double beta) {
  BLR_2x2 C_check(C);
  
  C.D(0, 0) *= beta;
  C.D(0, 0) += alpha * A.D(0, 0) * B.D(0, 0);
  Hatrix::Matrix SC_00 = A.S(0, 1) * A.V(1) * B.U(1) * B.S(1, 0);
  C.D(0, 0) += alpha * A.U(0) * SC_00 * B.V(0);

  C.S(0, 1) *= beta;
  C.S(0, 1) += alpha *
    transpose(C.U(0)) *
    A.D(0, 0) * B.U(0) * B.S(0, 1) * B.V(1) *
    transpose(C.V(1));
  C.S(0, 1) += alpha *
    transpose(C.U(0)) *
    A.U(0) * A.S(0, 1) * A.V(1) * B.D(1, 1) *
    transpose(C.V(1));

  C.S(1, 0) *= beta;
  C.S(1, 0) += alpha *
    transpose(C.U(1)) *
    A.U(1) * A.S(1, 0) * A.V(0) * B.D(0, 0) *
    transpose(C.V(0));
  C.S(1, 0) += alpha *
    transpose(C.U(1)) *
    A.D(1, 1) * B.U(1) * B.S(1, 0) * B.V(0) *
    transpose(C.V(0));

  C.D(1, 1) *= beta;
  Hatrix::Matrix SC_11 = A.S(1, 0) * A.V(0) * B.U(0) * B.S(0, 1);
  C.D(1, 1) += alpha * A.U(1) * SC_11 * B.V(1);
  C.D(1, 1) += alpha * A.D(1, 1) * B.D(1, 1);

  //Error check
  C_check.D(0, 0) *= beta;
  C_check.D(0, 0) += alpha * ((A.D(0, 0) * B.D(0, 0)) + (A.D(0, 1) * B.D(1, 0)));
  C_check.D(0, 1) *= beta;
  C_check.D(0, 1) += alpha * ((A.D(0, 0) * B.D(0, 1)) + (A.D(0, 1) * B.D(1, 1)));
  C_check.D(1, 0) *= beta;
  C_check.D(1, 0) += alpha * ((A.D(1, 0) * B.D(0, 0)) + (A.D(1, 1) * B.D(1, 0)));
  C_check.D(1, 1) *= beta;
  C_check.D(1, 1) += alpha * ((A.D(1, 0) * B.D(0, 1)) + (A.D(1, 1) * B.D(1, 1)));
  double norm = 0, diff = 0;
  for (int64_t i = 0; i < 2; ++i) {
    for (int64_t j = 0; j < 2; ++j) {
      norm += Hatrix::norm(C_check.D(i, j));
      if (i == j) {
	diff += Hatrix::norm_diff(C.D(i, j), C_check.D(i, j));
      } else {
        diff += Hatrix::norm_diff(C.U(i) * C.S(i, j) * C.V(j), C_check.D(i, j));
      }
    }
  }
  std::cout << "Approximate Matmul error: " <<diff/norm <<"\n";
}

int main() {
  int64_t N = 16;
  int64_t rank = 8;

  // Build 2x2 BLR and check accuracy
  BLR_2x2 A = construct_2x2_BLR(N, rank);
  BLR_2x2 B = construct_2x2_BLR(N, rank);
  BLR_2x2 C = construct_2x2_BLR(N, rank);
  
  matmul_2x2_BLR(A, B, C, 1, 1);

  return 0;
}
