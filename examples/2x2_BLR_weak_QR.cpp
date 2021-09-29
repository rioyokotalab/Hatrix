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
  std::cout <<"BLR2 construction error (rel): " <<std::sqrt(diff/norm) <<"\n";
  return A;
}

Hatrix::RowColMap<Hatrix::Matrix> compute_YTY(BLR_2x2& A, BLR_2x2& T, bool transT) {
  Hatrix::RowColMap<Hatrix::Matrix> out;
  for(int i=0; i<2; i++) {
    Hatrix::Matrix YT = Hatrix::triangular_matmul_out(T.D(0, 0),
						      i == 0 ? lower_tri(A.D(0, 0), true) :
						      T.S(i, 0), Hatrix::Right, Hatrix::Upper,
						      transT, false);
    for(int j=0; j<2; j++) {
      out.insert(i, j, Hatrix::matmul(YT,
				      j == 0 ? lower_tri(A.D(0, 0), true) : T.S(j, 0),
				      false, true));
    }
  }
  return out;
}

void qr_2x2_BLR(BLR_2x2& A, BLR_2x2& T) {
  //Triangularize first block column
  Hatrix::Matrix SV0 = Hatrix::matmul(A.S(1, 0), A.V(0));
  Hatrix::Matrix A0(A.D(0, 0).rows + SV0.rows, A.D(0, 0).cols);
  std::vector<Hatrix::Matrix> A0_parts = A0.split(std::vector<int64_t>{A.D(0, 0).rows},
						  std::vector<int64_t>{},
						  false);
  A0_parts[0] = A.D(0, 0);
  A0_parts[1] = SV0;
  T.insert_D(0, 0, Hatrix::Matrix(A0.cols, A0.cols));
  Hatrix::householder_qr_compact_wy(A0, T.D(0, 0));
  A.D(0, 0) = A0_parts[0];
  T.insert_S(1, 0, Hatrix::Matrix(A0_parts[1].rows, A0_parts[1].cols));
  T.S(1, 0) = A0_parts[1];

  //Apply block householder reflector
  Hatrix::RowColMap<Hatrix::Matrix> YTY = compute_YTY(A, T, true);
  //Update A(0, 1)
  Hatrix::Matrix S01 = A.S(0, 1);
  Hatrix::matmul(Hatrix::matmul(A.U(0), YTY(0, 0), true), A.U(0) * S01,
		 A.S(0, 1), false, false, -1, 1);
  Hatrix::matmul(Hatrix::matmul(A.U(0), YTY(0, 1), true),
		 Hatrix::matmul(Hatrix::matmul(A.U(1), A.D(1, 1), true), A.V(1), false, true),
		 A.S(0, 1), false, false, -1, 1);
  //Update A(1, 1)
  Hatrix::Matrix D11 = A.D(1, 1);
  Hatrix::matmul(A.U(1) * YTY(1, 0), A.U(0) * S01 * A.V(1),
		 A.D(1, 1), false, false, -1, 1);  
  Hatrix::matmul(A.U(1) * YTY(1, 1), Hatrix::matmul(A.U(1), D11, true),
		 A.D(1, 1), false, false, -1, 1);

  //Triangularize second block column (bottom right block)
  T.insert_D(1, 1, Hatrix::Matrix(A.D(1, 1).cols, A.D(1, 1).cols));
  Hatrix::householder_qr_compact_wy(A.D(1, 1), T.D(1, 1));
}

BLR_2x2 left_multiply_Q(BLR_2x2& A, BLR_2x2& T, BLR_2x2& C, bool trans) {
  BLR_2x2 out;
  for(int i=0; i<2; i++) {
    out.insert_U(i, Hatrix::Matrix(A.U(i)));
    out.insert_V(i, Hatrix::Matrix(C.V(i)));
  }
  for(int i=0; i<2; i++) {
    for(int j=0; j<2; j++) {
      if(i == j)
	out.insert_D(i, j, Hatrix::Matrix(A.D(i, i).rows, C.D(i, i).cols));
      else
	out.insert_S(i, j, Hatrix::Matrix(out.U(i).cols, out.V(j).rows));
    }
  }
  if(trans) {
    //Apply Q1
    Hatrix::RowColMap<Hatrix::Matrix> YTY = compute_YTY(A, T, trans);
    //Update C(0, 1)
    out.S(0, 1) = C.S(0, 1);
    Hatrix::matmul(Hatrix::matmul(A.U(0), YTY(0, 0) * C.U(0), true), C.S(0, 1),
		   out.S(0, 1), false, false, -1, 1);
    Hatrix::matmul(Hatrix::matmul(A.U(0), YTY(0, 1), true),
		   Hatrix::matmul(Hatrix::matmul(A.U(1), C.D(1, 1), true), C.V(1), false, true),
		   out.S(0, 1), false, false, -1, 1);
    //Update C(1, 1)
    out.D(1, 1) = C.D(1, 1);
    Hatrix::matmul(A.U(1) * YTY(1, 0), C.U(0) * C.S(0, 1) * C.V(1),
		   out.D(1, 1), false, false, -1, 1);
    Hatrix::matmul(A.U(1) * YTY(1, 1), Hatrix::matmul(A.U(1), C.D(1, 1), true),
		   out.D(1, 1), false, false, -1, 1);
    //Update C(0, 0)
    out.D(0, 0) = C.D(0, 0);
    Hatrix::matmul(YTY(0, 0), C.D(0, 0),
		   out.D(0, 0), false, false, -1, 1);
    Hatrix::matmul(YTY(0, 1), Hatrix::matmul(A.U(1), C.U(1), true) * C.S(1, 0) * C.V(0),
		   out.D(0, 0), false, false, -1, 1);
    //Update C(1, 0)
    out.S(1, 0) = C.S(1, 0);
    Hatrix::matmul(YTY(1, 0), Hatrix::matmul(C.D(0, 0), C.V(0), false, true),
		   out.S(1, 0), false, false, -1, 1);
    Hatrix::matmul(YTY(1, 1), Hatrix::matmul(A.U(1), C.U(1), true) * C.S(1, 0),
		   out.S(1, 0), false, false, -1, 1);
    
    //Apply Q2
    //Update C(1, 1)
    Hatrix::apply_block_reflector(A.D(1, 1), T.D(1, 1), out.D(1, 1),
				  Hatrix::Left, trans);
  }
  else {
    //Apply Q2
    //Update C(1, 1)
    out.D(1, 1) = C.D(1, 1);
    Hatrix::apply_block_reflector(A.D(1, 1), T.D(1, 1), out.D(1, 1),
				  Hatrix::Left, trans);
    C.D(1, 1) = out.D(1, 1);

    //Apply Q1
    Hatrix::RowColMap<Hatrix::Matrix> YTY = compute_YTY(A, T, trans);
    //Update C(0, 1)
    out.S(0, 1) = C.S(0, 1);
    Hatrix::matmul(Hatrix::matmul(A.U(0), YTY(0, 0) * C.U(0), true), C.S(0, 1),
		   out.S(0, 1), false, false, -1, 1);
    Hatrix::matmul(Hatrix::matmul(A.U(0), YTY(0, 1), true),
		   Hatrix::matmul(Hatrix::matmul(A.U(1), C.D(1, 1), true), C.V(1), false, true),
		   out.S(0, 1), false, false, -1, 1);
    //Update C(1, 1)
    out.D(1, 1) = C.D(1, 1);
    Hatrix::matmul(A.U(1) * YTY(1, 0), C.U(0) * C.S(0, 1) * C.V(1),
		   out.D(1, 1), false, false, -1, 1);
    Hatrix::matmul(A.U(1) * YTY(1, 1), Hatrix::matmul(A.U(1), C.D(1, 1), true),
		   out.D(1, 1), false, false, -1, 1);
    //Update C(0, 0)
    out.D(0, 0) = C.D(0, 0);
    Hatrix::matmul(YTY(0, 0), C.D(0, 0),
		   out.D(0, 0), false, false, -1, 1);
    Hatrix::matmul(YTY(0, 1), Hatrix::matmul(A.U(1), C.U(1), true) * C.S(1, 0) * C.V(0),
		   out.D(0, 0), false, false, -1, 1);
    //Update C(1, 0)
    out.S(1, 0) = C.S(1, 0);
    Hatrix::matmul(YTY(1, 0), Hatrix::matmul(C.D(0, 0), C.V(0), false, true),
		   out.S(1, 0), false, false, -1, 1);
    Hatrix::matmul(YTY(1, 1), Hatrix::matmul(A.U(1), C.U(1), true) * C.S(1, 0),
		   out.S(1, 0), false, false, -1, 1);
  }
  return out;
}

int main(int argc, char** argv) {
  int64_t N = argc > 1 ? atoi(argv[1]) : 32;
  int64_t rank = argc > 2 ? atoi(argv[2]) : 4;
  int64_t block_size = N/2;

  // Build 2x2 BLR and check accuracy
  BLR_2x2 A = construct_2x2_BLR(block_size, rank);
  BLR_2x2 A_copy(A);
  BLR_2x2 T;
  qr_2x2_BLR(A, T);
  
  // Approximate QR Error
  // Construct R
  BLR_2x2 R;
  R.insert_D(0, 0, upper_tri(A.D(0, 0)));
  R.insert_D(1, 1, upper_tri(A.D(1, 1)));
  R.insert_U(0, Hatrix::Matrix(A.U(0)));
  R.insert_U(1, Hatrix::Matrix(A.U(1)));
  R.insert_V(0, Hatrix::Matrix(A.V(0)));
  R.insert_V(1, Hatrix::Matrix(A.V(1)));
  R.insert_S(0, 1, Hatrix::Matrix(A.S(0, 1)));
  R.insert_S(1, 0, Hatrix::Matrix(A.U(1).cols, A.V(0).rows));
  BLR_2x2 QR = left_multiply_Q(A, T, R, false);
  
  double norm = 0, diff = 0, fnorm, fdiff;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      if (i == j) {
	fnorm = Hatrix::norm(A_copy.D(i, j));
	fdiff = Hatrix::norm(QR.D(i, j) - A_copy.D(i, j));
      } else {
	fnorm = Hatrix::norm(A_copy.U(i) * A_copy.S(i, j) * A_copy.V(j));
        fdiff = Hatrix::norm(QR.U(i) * QR.S(i, j) * QR.V(j) -
			     A_copy.U(i) * A_copy.S(i, j) * A_copy.V(j));
      }
      norm += fnorm * fnorm;
      diff += fdiff * fdiff;
    }
  }
  std::cout << "BLR2-QR Relative Error: " << std::sqrt(diff/norm) << "\n";
  
  //Approximate Q Orthogonality
  //Construct Q
  BLR_2x2 I;
  I.insert_D(0, 0, Hatrix::generate_identity_matrix(A.D(0, 0).rows, A.D(0, 0).cols));
  I.insert_D(1, 1, Hatrix::generate_identity_matrix(A.D(1, 1).rows, A.D(1, 1).cols));
  I.insert_U(0, Hatrix::Matrix(A.U(0)));
  I.insert_U(1, Hatrix::Matrix(A.U(1)));
  I.insert_V(0, transpose(A.U(0)));
  I.insert_V(1, transpose(A.U(1)));
  I.insert_S(0, 1, Hatrix::Matrix(I.U(0).cols, I.V(1).rows));
  I.insert_S(1, 0, Hatrix::Matrix(I.U(1).cols, I.V(0).rows));
  BLR_2x2 Q = left_multiply_Q(A, T, I, false);
  BLR_2x2 QtQ = left_multiply_Q(A, T, Q, true);
  
  norm = 0, diff = 0;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      if (i == j) {
	fdiff = Hatrix::norm(QtQ.D(i, j) - Hatrix::generate_identity_matrix(QtQ.D(i, j).rows,
									    QtQ.D(i, j).cols));
      } else {
        fdiff = Hatrix::norm(QtQ.U(i) * QtQ.S(i, j) * QtQ.V(j));
      }
      diff += fdiff * fdiff;
    }
  }
  std::cout << "BLR2-QR Orthogonality: " << std::sqrt(diff/N) << "\n";
  
  return 0;
}
