#include <cstdint>
#include <cmath>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Hatrix/Hatrix.hpp"

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

std::vector<double> equally_spaced_vector(int64_t N, double minVal, double maxVal) {
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
  randpts.push_back(equally_spaced_vector(2*N, 0.0, 1.0)); //1D

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

// Projected multiplication of A and B in the given bases of C
void projected_matmul_2x2_BLR(BLR_2x2& A, BLR_2x2& B, BLR_2x2& C) {
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
    }
  }
  std::cout << "Projected matmul error (Rel): " <<std::sqrt(diff/norm) <<"\n";
}

void construct_induced_product_basis(BLR_2x2& A, BLR_2x2& B, BLR_2x2& M, BLR_2x2& tmp) {
  //Assume uniform rank
  //U bases
  double rankA = A.U(0).cols;
  double rankB = B.U(0).cols;
  tmp.insert_U(0, A.D(0, 0) * B.U(0));
  tmp.insert_U(1, A.D(1, 1) * B.U(1));
  M.insert_U(0, Hatrix::Matrix(A.U(0).rows, rankA + rankB));
  M.insert_U(1, Hatrix::Matrix(A.U(1).rows, rankA + rankB));
  for(int64_t j=0; j<M.U(0).cols; j++) {
    for(int64_t i=0; i<M.U(0).rows; i++) {
      M.U(0)(i, j) = (j < rankA ? A.U(0)(i, j) : tmp.U(0)(i, j-rankA));
      M.U(1)(i, j) = (j < rankA ? A.U(1)(i, j) : tmp.U(1)(i, j-rankA));
    }
  }
  //V bases
  tmp.insert_V(0, A.V(0) * B.D(0, 0));
  tmp.insert_V(1, A.V(1) * B.D(1, 1));
  M.insert_V(0, Hatrix::Matrix(rankA + rankB, B.V(0).cols));
  M.insert_V(1, Hatrix::Matrix(rankA + rankB, B.V(1).cols));
  for(int64_t j=0; j<M.V(0).cols; j++) {
    for(int64_t i=0; i<M.V(0).rows; i++) {
      M.V(0)(i, j) = (i < rankA ? tmp.V(0)(i, j) : B.V(0)(i-rankA, j));
      M.V(1)(i, j) = (i < rankA ? tmp.V(1)(i, j) : B.V(1)(i-rankA, j));
    }
  }
  //Initialize coupling matrices
  M.insert_S(0, 1, Hatrix::Matrix(M.U(0).cols, M.V(1).rows));
  M.insert_S(1, 0, Hatrix::Matrix(M.U(1).cols, M.V(0).rows));
  //Inadmissible blocks
  M.insert_D(0, 0, Hatrix::Matrix(A.D(0, 0).rows, B.D(0, 0).cols));
  M.insert_D(1, 1, Hatrix::Matrix(A.D(1, 1).rows, B.D(1, 1).cols));
}

void recompress(BLR_2x2& A, double rank) {
  //Recompress U bases
  for(int64_t i=0; i<2; i++) {
    Hatrix::Matrix u(A.U(i).rows, A.U(i).min_dim());
    Hatrix::Matrix s(A.U(i).min_dim(), A.U(i).min_dim());
    Hatrix::Matrix v(A.U(i).min_dim(), A.U(i).cols);
    Hatrix::truncated_svd(A.U(i), u, s, v, rank);
    A.U(i) = std::move(u);
    //Multiply to the left of coupling matrices
    for(int64_t j=0; j<2; j++) {
      if(i != j) {
	A.S(i, j) = s * v * A.S(i, j);
      }
    }
  }
  //Recompress V bases
  for(int64_t j=0; j<2; j++) {
    Hatrix::Matrix u(A.V(j).rows, A.V(j).min_dim());
    Hatrix::Matrix s(A.V(j).min_dim(), A.V(j).min_dim());
    Hatrix::Matrix v(A.V(j).min_dim(), A.V(j).cols);
    Hatrix::truncated_svd(A.V(j), u, s, v, rank);
    A.V(j) = std::move(v);
    //Multiply to the right of coupling matrices
    for(int64_t i=0; i<2; i++) {
      if(i != j) {
	A.S(i, j) = A.S(i, j) * u * s;
      }
    }
  }
}

BLR_2x2 exact_matmul_2x2_BLR(BLR_2x2& A, BLR_2x2& B) {
  BLR_2x2 M, tmp;
  construct_induced_product_basis(A, B, M, tmp);
  //Assume uniform rank
  double rankA = A.U(0).cols;
  double rankB = B.U(0).cols;

  //For error checking
  BLR_2x2 M_check(M);
  M_check.insert_D(0, 1, Hatrix::Matrix(M_check.U(0).rows, M_check.V(1).cols));
  M_check.insert_D(1, 0, Hatrix::Matrix(M_check.U(1).rows, M_check.V(0).cols));

  //M(0, 0)
  Hatrix::matmul(A.D(0, 0), B.D(0, 0), M.D(0, 0));
  Hatrix::Matrix SM_00 = A.S(0, 1) * A.V(1) * B.U(1) * B.S(1, 0);
  Hatrix::matmul(A.U(0), SM_00 * B.V(0), M.D(0, 0));
  //M_check(0, 0)
  Hatrix::matmul(A.D(0, 0), B.D(0, 0), M_check.D(0, 0));
  Hatrix::matmul(A.D(0, 1), B.D(1, 0), M_check.D(0, 0));

  //M(0, 1)
  for(int64_t i=0; i<rankA; i++) {
    for(int64_t j=0; j<rankA; j++) {
      M.S(0, 1)(i, j) = A.S(0, 1)(i, j);
    }
  }
  for(int64_t i=0; i<rankB; i++) {
    for(int64_t j=0; j<rankB; j++) {
      M.S(0, 1)(i+rankA, j+rankA) = B.S(0, 1)(i, j);
    }
  }
  //M_check(0, 1)
  Hatrix::matmul(A.D(0, 0), B.D(0, 1), M_check.D(0, 1));
  Hatrix::matmul(A.D(0, 1), B.D(1, 1), M_check.D(0, 1));

  //M(1, 0)
  for(int64_t i=0; i<rankA; i++) {
    for(int64_t j=0; j<rankA; j++) {
      M.S(1, 0)(i, j) = A.S(1, 0)(i, j);
    }
  }
  for(int64_t i=0; i<rankB; i++) {
    for(int64_t j=0; j<rankB; j++) {
      M.S(1, 0)(i+rankA, j+rankA) = B.S(1, 0)(i, j);
    }
  }
  //M_check(1, 0)
  Hatrix::matmul(A.D(1, 0), B.D(0, 0), M_check.D(1, 0));
  Hatrix::matmul(A.D(1, 1), B.D(1, 0), M_check.D(1, 0));

  //M(1, 1)
  Hatrix::Matrix SM_11 = A.S(1, 0) * A.V(0) * B.U(0) * B.S(0, 1);
  Hatrix::matmul(A.U(1), SM_11 * B.V(1), M.D(1, 1));
  Hatrix::matmul(A.D(1, 1), B.D(1, 1), M.D(1, 1));
  //M_check(1, 1)
  Hatrix::matmul(A.D(1, 0), B.D(0, 1), M_check.D(1, 1));
  Hatrix::matmul(A.D(1, 1), B.D(1, 1), M_check.D(1, 1));

  double norm = 0, diff = 0, fnorm, fdiff;
  for (int64_t i = 0; i < 2; ++i) {
    for (int64_t j = 0; j < 2; ++j) {
      fnorm = Hatrix::norm(M_check.D(i, j));
      norm += fnorm * fnorm;
      if (i == j) {
	fdiff = Hatrix::norm(M.D(i, j) - M_check.D(i, j));
      } else {
        fdiff = Hatrix::norm(M.U(i) * M.S(i, j) * M.V(j) - M_check.D(i, j));
      }
      diff += fdiff * fdiff;
    }
  }
  std::cout << "Exact matmul error before recompress (Rel): " <<std::sqrt(diff/norm) <<"\n";
  recompress(M, rankA);
  norm = 0;
  diff = 0;
  for (int64_t i = 0; i < 2; ++i) {
    for (int64_t j = 0; j < 2; ++j) {
      fnorm = Hatrix::norm(M_check.D(i, j));
      norm += fnorm * fnorm;
      if (i == j) {
	fdiff = Hatrix::norm(M.D(i, j) - M_check.D(i, j));
      } else {
        fdiff = Hatrix::norm(M.U(i) * M.S(i, j) * M.V(j) - M_check.D(i, j));
      }
      diff += fdiff * fdiff;
    }
  }
  std::cout << "Exact matmul error after recompress (Rel): " <<std::sqrt(diff/norm) <<"\n";

  return M;
}

int main(int argc, char** argv) {
  int64_t N = argc > 1 ? atoi(argv[1]) : 32;
  int64_t rank = argc > 2 ? atoi(argv[2]) : 4;
  int64_t block_size = N/2;

  // Build 2x2 BLR and check accuracy
  BLR_2x2 A = construct_2x2_BLR(block_size, rank);
  BLR_2x2 B = construct_2x2_BLR(block_size, rank);
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

  projected_matmul_2x2_BLR(A, B, C);
  BLR_2x2 M = exact_matmul_2x2_BLR(A, B);

  return 0;
}
