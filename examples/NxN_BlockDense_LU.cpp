#include <algorithm>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Hatrix/Hatrix.h"

class BlockDense {
 private:
  std::unordered_map<std::tuple<int64_t, int64_t>, Hatrix::Matrix> data;

 public:
  void insert(int64_t row, int64_t col, Hatrix::Matrix&& D) {
    data.emplace(std::make_tuple(row, col), std::move(D));
  }

  Hatrix::Matrix& operator()(int64_t row, int64_t col) {
    return data.at({row, col});
  }
  const Hatrix::Matrix& operator()(int64_t row, int64_t col) const {
    return data.at({row, col});
  }
};

BlockDense build_matrix(int64_t block_size, int64_t n_blocks) {
  BlockDense A;
  for (int64_t i = 0; i < n_blocks; ++i) {
    for (int64_t j = 0; j < n_blocks; ++j) {
      Hatrix::Matrix mat =
          Hatrix::generate_random_matrix(block_size, block_size);
      // Avoid pivoting
      if (i == j)
        for (int64_t i_c = 0; i_c < mat.min_dim(); ++i_c) mat(i_c, i_c) += 10;
      A.insert(i, j, std::move(mat));
    }
  }
  return A;
}

std::vector<Hatrix::Matrix> multiply(const BlockDense& A,
                                     const std::vector<Hatrix::Matrix>& x,
                                     int64_t n_blocks) {
  std::vector<Hatrix::Matrix> b;
  for (int64_t i = 0; i < n_blocks; ++i) {
    b.push_back(Hatrix::Matrix(x[i].rows, 1));
    for (int64_t j = 0; j < n_blocks; ++j) {
      Hatrix::matmul(A(i, j), x[j], b[i]);
    }
  }
  return b;
}

void factorize(BlockDense& A, BlockDense& L, BlockDense& U, int64_t n_blocks) {
  BlockDense A_check(A);
  for (int64_t diag = 0; diag < n_blocks; ++diag) {
    // Initialize L, U
    Hatrix::Matrix& A_diag = A(diag, diag);
    L.insert(diag, diag, Hatrix::Matrix(A_diag.rows, A_diag.cols));
    U.insert(diag, diag, Hatrix::Matrix(A_diag.rows, A_diag.cols));
    for (int64_t i = diag + 1; i < n_blocks; ++i) {
      L.insert(i, diag, std::move(A(i, diag)));
      U.insert(diag, i, std::move(A(diag, i)));
    }

    // Left looking LU
    for (int64_t k = 0; k < diag; ++k) {
      Hatrix::matmul(L(diag, k), U(k, diag), A_diag, false, false, -1, 1);
    }
    Hatrix::lu(A_diag, L(diag, diag), U(diag, diag));
    Hatrix::Context::join();

    for (int64_t j = diag + 1; j < n_blocks; ++j) {
      Hatrix::Context::critical();
      for (int64_t k = 0; k < diag; ++k) {
        Hatrix::matmul(L(diag, k), U(k, j), U(diag, j), false, false, -1, 1);
      }
      Hatrix::Context::fork();
      Hatrix::solve_triangular(L(diag, diag), U(diag, j), Hatrix::Left,
                               Hatrix::Lower, true);
    }

    for (int64_t i = diag + 1; i < n_blocks; ++i) {
      Hatrix::Context::critical();
      for (int64_t k = 0; k < diag; ++k) {
        Hatrix::matmul(L(i, k), U(k, diag), L(i, diag), false, false, -1, 1);
      }
      Hatrix::Context::fork();
      Hatrix::solve_triangular(U(diag, diag), L(i, diag), Hatrix::Right,
                               Hatrix::Upper, false);
    }

    Hatrix::Context::join();
  }

  // Check result
  double error = 0;
  for (int64_t i = 0; i < n_blocks; ++i) {
    for (int64_t j = 0; j < n_blocks; ++j) {
      Hatrix::Matrix LU(A(i, i).rows, A(i, i).cols);
      for (int64_t k = 0; k <= std::min(i, j); ++k) {
        Hatrix::matmul(L(i, k), U(k, j), LU);
      }
      error += Hatrix::norm_diff(LU, A_check(i, j));
    }
  }
  std::cout << "Total factorization error: " << error << "\n";
}

void solve(const BlockDense& L, const BlockDense& U,
           const std::vector<Hatrix::Matrix>& x, std::vector<Hatrix::Matrix>& b,
           int64_t n_blocks) {
  // Forward substitution
  for (int64_t i = 0; i < n_blocks; ++i) {
    for (int64_t j = 0; j < i; ++j) {
      Hatrix::matmul(L(i, j), b[j], b[i], false, false, -1, 1);
    }
    Hatrix::solve_triangular(L(i, i), b[i], Hatrix::Left, Hatrix::Lower, true);
  }
  // Backward substitution
  for (int64_t i = n_blocks - 1; i >= 0; --i) {
    for (int64_t j = n_blocks - 1; j > i; --j) {
      Hatrix::matmul(U(i, j), b[j], b[i], false, false, -1, 1);
    }
    Hatrix::solve_triangular(U(i, i), b[i], Hatrix::Left, Hatrix::Upper, false);
  }
  // Check result
  double error = 0;
  for (int64_t i = 0; i < n_blocks; ++i) {
    error += Hatrix::norm_diff(x[i], b[i]);
  }
  std::cout << "Total solution error: " << error << "\n";
}

#include <sys/time.h>

int main(int argc, const char** argv) {
  int64_t block_size = 32;
  int64_t n_blocks = 16;
  Hatrix::Context::init(argc, argv);

  BlockDense A = build_matrix(block_size, n_blocks);

  std::vector<Hatrix::Matrix> x;
  for (int64_t i = 0; i < n_blocks; ++i) {
    x.push_back(Hatrix::generate_random_matrix(block_size, 1));
  }
  std::vector<Hatrix::Matrix> b = multiply(A, x, n_blocks);

  BlockDense L, U;

  struct timeval start, end;

  gettimeofday(&start, NULL);
  factorize(A, L, U, n_blocks);
  gettimeofday(&end, NULL);

  long long int elapse = end.tv_sec * 1000000 + end.tv_usec - start.tv_sec * 1000000 + start.tv_usec;
  

  std::cout << "fac time: " << (double)elapse / 1.e3 << " ms."<< std::endl;

  Hatrix::Context::join();
  solve(L, U, x, b, n_blocks);
  Hatrix::Context::finalize();
}
