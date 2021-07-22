#include <cstdint>
#include <cmath>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

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

std::vector<double> equallySpacedVector(int64_t N, double minVal, double maxVal) {
  std::vector<double> res(N, 0.0);
  double rnge = maxVal - minVal;
  for(int64_t i=0; i<N; i++) {
    res[i] = minVal + ((double)i/(double)rnge);
  }
  return res;
}

BLR_2x2 construct_2x2_BLR(int64_t N, int64_t rank) {
  // Random points for laplace kernel
  std::vector<std::vector<double>> randpts;
  randpts.push_back(equallySpacedVector(2*N, 0.0, 1.0)); //1D
  
  BLR_2x2 A;
  // Also store expected errors to check against later
  std::unordered_map<std::tuple<int64_t, int64_t>, double> expected_err;  
  for (int64_t i = 0; i < 2; ++i) {
    for (int64_t j = 0; j < 2; ++j) {
      A.insert_D(i, j, Hatrix::generate_laplacend_matrix(randpts, N, N, i*N, j*N));
      if (i != j) {
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

  double norm = 0, diff = 0, fnorm, fdiff;
  for (int64_t i = 0; i < 2; ++i) {
    for (int64_t j = 0; j < 2; ++j) {
      fnorm = Hatrix::norm(A.D(i, j));
      norm += fnorm * fnorm;
      if (i == j) {
        continue;
      } else {
	fdiff = Hatrix::norm(A.U(i) * A.S(i, j) * A.V(j) - A.D(i, j));
        diff += fdiff * fdiff;
      }
    }
  }
  std::cout <<"BLR construction error (rel): " <<std::sqrt(diff/norm) <<"\n";
  return A;
}

// Approximate multiplication of A and B
BLR_2x2 matmul_2x2_BLR(BLR_2x2& A, BLR_2x2& B) {
  //Initialize C with row-bases of A and col-bases of B
  BLR_2x2 C;
  C.insert_D(0, 0, Hatrix::Matrix(A.D(0, 0).rows, B.D(0, 0).cols));
  C.insert_D(1, 1, Hatrix::Matrix(A.D(1, 1).rows, B.D(1, 1).cols));
  C.insert_U(0, Hatrix::Matrix(A.U(0)));
  C.insert_U(1, Hatrix::Matrix(A.U(1)));
  C.insert_V(0, Hatrix::Matrix(B.V(0)));
  C.insert_V(1, Hatrix::Matrix(B.V(1)));
  C.insert_S(0, 1, Hatrix::Matrix(A.U(0).cols, B.V(1).rows));
  C.insert_S(1, 0, Hatrix::Matrix(A.U(1).cols, B.V(0).rows));
  
  //For error checking
  BLR_2x2 C_check(C);
  C_check.insert_D(0, 1, Hatrix::Matrix(C.U(0).rows, C.V(1).cols));
  C_check.insert_D(1, 0, Hatrix::Matrix(C.U(1).rows, C.V(0).cols));

  //C = A x B
  //C(0, 0)
  Hatrix::matmul(A.D(0, 0), B.D(0, 0), C.D(0, 0));
  Hatrix::Matrix SC_00 = A.S(0, 1) * A.V(1) * B.U(1) * B.S(1, 0);
  Hatrix::matmul(A.U(0), SC_00 * B.V(0), C.D(0, 0));
  //C_check(0, 0)
  Hatrix::matmul(A.D(0, 0), B.D(0, 0), C_check.D(0, 0));
  Hatrix::matmul(A.D(0, 1), B.D(1, 0), C_check.D(0, 0));

  //C(0, 1)
  Hatrix::matmul(A.U(0), A.D(0, 0) * B.U(0) * B.S(0, 1), C.S(0, 1), true, false);
  Hatrix::matmul(A.S(0, 1) * A.V(1) * B.D(1, 1), B.V(1), C.S(0, 1), false, true);
  //C_check(0, 1)
  Hatrix::matmul(A.D(0, 0), B.D(0, 1), C_check.D(0, 1));
  Hatrix::matmul(A.D(0, 1), B.D(1, 1), C_check.D(0, 1));

  //C(1, 0)
  Hatrix::matmul(A.S(1, 0) * A.V(0) * B.D(0, 0), B.V(0), C.S(1, 0), false, true);
  Hatrix::matmul(A.U(1), A.D(1, 1) * B.U(1) * B.S(1, 0), C.S(1, 0), true, false);
  //C_check(1, 0)
  Hatrix::matmul(A.D(1, 0), B.D(0, 0), C_check.D(1, 0));
  Hatrix::matmul(A.D(1, 1), B.D(1, 0), C_check.D(1, 0));

  //C(1, 1)
  Hatrix::Matrix SC_11 = A.S(1, 0) * A.V(0) * B.U(0) * B.S(0, 1);
  Hatrix::matmul(A.U(1), SC_11 * B.V(1), C.D(1, 1));
  Hatrix::matmul(A.D(1, 1), B.D(1, 1), C.D(1, 1));
  //C_check(1, 1)
  Hatrix::matmul(A.D(1, 0), B.D(0, 1), C_check.D(1, 1));
  Hatrix::matmul(A.D(1, 1), B.D(1, 1), C_check.D(1, 1));
  
  double norm = 0, diff = 0, fnorm, fdiff;
  for (int64_t i = 0; i < 2; ++i) {
    for (int64_t j = 0; j < 2; ++j) {
      fnorm = Hatrix::norm(C_check.D(i, j));
      norm += fnorm * fnorm;
      if (i == j) {
	fdiff = Hatrix::norm(C.D(i, j) - C_check.D(i, j));
      } else {
        fdiff = Hatrix::norm(C.U(i) * C.S(i, j) * C.V(j) - C_check.D(i, j));
      }
      diff += fdiff * fdiff;
      std::cout <<"diff(" <<i <<"," <<j <<"): " <<fdiff * fdiff <<"\n";
    }
  }
  std::cout << "Approximate matmul error (Rel): " <<std::sqrt(diff/norm) <<"\n";

  return C;
}

int main(int argc, char** argv) {
  int64_t N = argc > 1 ? atoi(argv[1]) : 32;
  int64_t rank = argc > 2 ? atoi(argv[2]) : 4;
  int64_t block_size = N/2;

  // Build 2x2 BLR and check accuracy
  BLR_2x2 A = construct_2x2_BLR(block_size, rank);
  BLR_2x2 B = construct_2x2_BLR(block_size, rank);
  BLR_2x2 C = matmul_2x2_BLR(A, B);
  
  return 0;
}
