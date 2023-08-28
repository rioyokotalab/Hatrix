#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cassert>

#include "Hatrix/Hatrix.h"

using vec = std::vector<int64_t>;
using randvec_t = std::vector<std::vector<double> >;

std::vector<double> equally_spaced_vector(int64_t N, double minVal, double maxVal) {
  std::vector<double> res(N, 0.0);
  double rnge = maxVal - minVal;
  for(int64_t i=0; i<N; i++) {
    res[i] = minVal + ((double)i/(double)rnge);
  }
  return res;
}

class BLR2_SPD {
 public:
  Hatrix::RowColMap<Hatrix::Matrix> D, S;
  Hatrix::RowMap<Hatrix::Matrix> U, Uc;
  int64_t N, block_size, n_blocks, rank, admis;
  double construct_error;

  BLR2_SPD(const randvec_t& randpts, int64_t N, int64_t block_size, int64_t rank,
           int64_t admis):
      N(N), block_size(block_size), n_blocks(N/block_size), rank(rank), admis(admis)
  {
    for (int64_t i = 0; i < n_blocks; ++i) {
      for (int64_t j = 0; j < n_blocks; ++j) {
        D.insert(i, j,
                 Hatrix::generate_laplacend_matrix(randpts,
                                                   block_size, block_size,
                                                   i*block_size, j*block_size));
      }
    }
    // Also store expected errors to check against later
    std::unordered_map<std::tuple<int64_t, int64_t>, double> expected_err;
    int64_t oversampling = 5;
    Hatrix::Matrix Ui, Sij, Vj;
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
        Hatrix::matmul(D(i, j), Y[j], AY);
      }
      std::tie(Ui, Sij, Vj, error) = Hatrix::truncated_svd(AY, rank);
      U.insert(i, std::move(Ui));
    }
    for (int i = 0; i < n_blocks; ++i) {
      for (int j = 0; j < n_blocks; ++j) {
        if (i != j) {
          S.insert(i, j,
                   Hatrix::matmul(Hatrix::matmul(U[i], D(i, j), true), U[j]));
        }
      }
    }
    double diff = 0, norm = 0, fnorm, fdiff;
    for (int i = 0; i < n_blocks; ++i) {
      for (int j = 0; j < n_blocks; ++j) {
        fnorm = Hatrix::norm(D(i, j));
        norm += fnorm * fnorm;
        if (i == j)
          continue;
        else {
          fdiff = Hatrix::norm(U[i] * S(i, j) * transpose(U[j]) - D(i, j));
          diff += fdiff * fdiff;
        }
      }
    }
    construct_error = std::sqrt(diff/norm);
  }

  Hatrix::Matrix Uf(int row) {
    Hatrix::Matrix Uf(block_size, block_size);
    int c_size = block_size - rank;
    if(c_size == 0) return U[row];

    vec col_split_indices = vec{c_size};
    auto Uf_splits = Uf.split(vec(), col_split_indices);
    Uf_splits[0] = Uc[row];
    Uf_splits[1] = U[row];
    return Uf;
  }

};

Hatrix::Matrix make_complement(const Hatrix::Matrix& _U) {
  Hatrix::Matrix U(_U);
  int c_size = U.rows - U.cols;
  if(c_size == 0) return Hatrix::Matrix(0, 0);

  Hatrix::Matrix Q(U.rows, U.rows);
  Hatrix::Matrix R(U.rows, U.cols);
  Hatrix::qr(U, Q, R);

  auto Q_splits = Q.split(vec(), vec{U.cols});
  Hatrix::Matrix Uc(U.rows, c_size);
  Uc = Q_splits[1];
  return Uc;
}

void partial_ldl_diag(BLR2_SPD& A, int i) {
  int c_size = A.block_size - A.rank;
  auto D_splits = A.D(i, i).split(vec{c_size}, vec{c_size});
  Hatrix::ldl(D_splits[0]);
  Hatrix::solve_triangular(D_splits[0], D_splits[2], Hatrix::Right,
                           Hatrix::Lower, true, true, 1.);
  Hatrix::solve_diagonal(D_splits[0], D_splits[2], Hatrix::Right, 1.);
  //Compute Schur's complement
  Hatrix::Matrix L_oc(D_splits[2].rows, D_splits[2].cols);
  L_oc = D_splits[2]; // Hatrix::Matrix L_oc(D_splits[2]) doesn't work due to current copy-constructor
  column_scale(L_oc, D_splits[0]); //L*D
  Hatrix::matmul(L_oc, D_splits[2], D_splits[3], false, true, -1., 1.);
}

Hatrix::Matrix factorize(BLR2_SPD& A) {
  int c_size = A.block_size - A.rank;
  for(int64_t i = 0; i < A.n_blocks; i++) {
    A.Uc.insert(i, make_complement(A.U[i]));
    //Transform diagonal
    A.D(i, i) = Hatrix::matmul(Hatrix::matmul(A.Uf(i), A.D(i, i), true), A.Uf(i));
    //Partial factorization
    if(c_size > 0) partial_ldl_diag(A, i);
  }

  Hatrix::Matrix last(A.n_blocks * A.rank, A.n_blocks * A.rank);
  auto last_splits = last.split(A.n_blocks, A.n_blocks);

  for(int64_t i = 0; i < A.n_blocks; i++) {
    for(int64_t j = 0; j < A.n_blocks; j++) {
      if(i == j) {
        auto D_split = A.D(i, j).split(vec{c_size}, vec{c_size});
        last_splits[i * A.n_blocks + j] = D_split[3];
      }
      else {
        last_splits[i * A.n_blocks + j] = A.S(i, j);
      }
    }
  }
  Hatrix::ldl(last);
  return last;
}

void substitute(BLR2_SPD& A, Hatrix::Matrix& root, Hatrix::Matrix& b) {
  //Split b
  int c_size = A.block_size - A.rank;
  vec local_split_indices{c_size};
  vec global_split_indices;
  for(int i = 0; i < A.n_blocks; i++) {
    global_split_indices.push_back(i*A.block_size + c_size);
    if(i < (A.n_blocks - 1))
      global_split_indices.push_back(i*A.block_size + c_size + A.rank);
  }
  auto b_block_splits = b.split(A.n_blocks, 1);
  auto b_block_co_splits = b.split(global_split_indices, vec());

  //---Forward Substitution---
  for(int i = 0; i < A.n_blocks; i++) {
    //Multiply with orthogonal matrices
    Hatrix::Matrix bi(b_block_splits[i].rows, b_block_splits[i].cols);
    bi = b_block_splits[i];
    Hatrix::matmul(A.Uf(i), bi, b_block_splits[i], true, false, 1, 0);

    //Solve triangular from diagonal partial factorizations
    if(c_size > 0) {
      auto Li_splits = A.D(i, i).split(local_split_indices,
                                       local_split_indices);
      Hatrix::solve_triangular(Li_splits[0], b_block_co_splits[2*i],
                               Hatrix::Left, Hatrix::Lower, true);
      Hatrix::matmul(Li_splits[2], b_block_co_splits[2*i], b_block_co_splits[2*i+1],
                     false, false, -1, 1);
    }
  }
  //Gather o parts
  Hatrix::Matrix bo(A.n_blocks * A.rank, 1);
  auto bo_splits = bo.split(A.n_blocks, 1);
  for(int i = 0; i < A.n_blocks; i++) {
    bo_splits[i] = b_block_co_splits[2*i+1];
  }
  Hatrix::solve_triangular(root, bo, Hatrix::Left, Hatrix::Lower, true);

  //---Solve Diagonal---
  //Solve in c parts
  if(c_size > 0) {
    for(int i = 0; i < A.n_blocks; i++) {
      auto Li_splits = A.D(i, i).split(local_split_indices,
                                       local_split_indices);
      Hatrix::solve_diagonal(Li_splits[0], b_block_co_splits[2*i], Hatrix::Left);
    }
  }
  //Solve in o parts
  Hatrix::solve_diagonal(root, bo, Hatrix::Left);

  //---Backward substitution---
  Hatrix::solve_triangular(root, bo, Hatrix::Left, Hatrix::Lower, true, true);
  //Scatter o parts
  for(int i = 0; i < A.n_blocks; i++) {
    b_block_co_splits[2*i+1] = bo_splits[i];
  }
  for(int i = 0; i < A.n_blocks; i++) {
    if(c_size > 0) {
      auto Li_splits = A.D(i, i).split(local_split_indices,
                                       local_split_indices);
      Hatrix::matmul(Li_splits[2], b_block_co_splits[2*i+1], b_block_co_splits[2*i],
                     true, false, -1., 1.);
      Hatrix::solve_triangular(Li_splits[0], b_block_co_splits[2*i],
                               Hatrix::Left, Hatrix::Lower, true, true);
    }
    //Multiply with orthogonal matrix
    Hatrix::Matrix bi(b_block_splits[i].rows, b_block_splits[i].cols);
    bi = b_block_splits[i];
    Hatrix::matmul(A.Uf(i), bi, b_block_splits[i], false, false, 1., 0);
  }
}

double rel_error(const Hatrix::Matrix& A, const Hatrix::Matrix& B) {
  double norm = Hatrix::norm(A);
  double diff = Hatrix::norm(A - B);
  return diff/norm;
}

int main(int argc, char** argv) {
  int64_t N = argc > 1 ? atoi(argv[1]) : 256;
  int64_t block_size = argc > 2 ? atoi(argv[2]) : 32;
  int64_t rank = argc > 3 ? atoi(argv[3]) : 8;

  randvec_t randpts;
  randpts.push_back(equally_spaced_vector(N, 0.0, 1.0)); // 1D

  Hatrix::Context::init();

  BLR2_SPD A(randpts, N, block_size, rank, 0);
  Hatrix::Matrix A_dense = Hatrix::generate_laplacend_matrix(randpts, N, N, 0, 0);
  Hatrix::Matrix x = Hatrix::generate_random_matrix(N, 1);
  Hatrix::Matrix b = A_dense * x;
  std::cout <<"BLR construction error: " <<A.construct_error <<"\n";

  Hatrix::Matrix last = factorize(A);
  substitute(A, last, b);
  double substitution_error = rel_error(x, b);
  std::cout <<"Solve Error : " <<substitution_error <<"\n";

  Hatrix::Context::finalize();
  return 0;
}
