#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Hatrix/Hatrix.h"

// BLR2 compression scheme using randomization and multiplying each row/col block
// into another to generate the shared bases.
// UMV factorization using Miamiao Ma's method.

using randvec_t = std::vector<std::vector<double> >;

std::vector<double> equally_spaced_vector(int N, double minVal, double maxVal) {
  std::vector<double> res(N, 0.0);
  double rnge = maxVal - minVal;
  for(int i=0; i<N; i++) {
    res[i] = minVal + ((double)i/(double)rnge);
  }
  return res;
}

double rel_error(const Hatrix::Matrix& A, const Hatrix::Matrix& B) {
  return Hatrix::norm(A - B) / Hatrix::norm(B);
}

namespace Hatrix {
  class BLR2 {
  private:
    RowMap U;
    ColMap V;
    RowColMap<bool> is_admissible;
    RowColMap<Matrix> D, S;
    int64_t N, nblocks, rank, admis;

    Hatrix::Matrix make_complement(const Hatrix::Matrix& Q) {
      Hatrix::Matrix Q_F(Q.rows, Q.rows - Q.cols);
      Hatrix::Matrix Q_full, R;
      std::tie(Q_full, R) = qr(Q, Hatrix::Lapack::QR_mode::Full, Hatrix::Lapack::QR_ret::OnlyQ);

      for (int i = 0; i < Q_F.rows; ++i) {
        for (int j = 0; j < Q_F.cols; ++j) {
          Q_F(i, j) = Q_full(i, j + Q.cols);
        }
      }
      return Q_F;
    }

  public:
    BLR2(const randvec_t& randpts, int64_t N, int64_t nblocks, int64_t rank, int64_t admis) :
      N(N), nblocks(nblocks), rank(rank), admis(admis) {
      int block_size = N / nblocks;

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          is_admissible.insert(i, j, std::abs(i - j) > admis);

          if (!is_admissible(i, j)) {
            D.insert(i, j,
                     Hatrix::generate_laplacend_matrix(randpts,
                                                       block_size, block_size,
                                                       i*block_size, j*block_size));
          }
        }
      }

      int64_t oversampling = 5;
      Hatrix::Matrix Utemp, Stemp, Vtemp;
      double error;
      std::vector<Hatrix::Matrix> Y;

      // Generate a bunch of random matrices.
      for (int64_t i = 0; i < nblocks; ++i) {
        Y.push_back(
                    Hatrix::generate_random_matrix(block_size, rank + oversampling));
      }

      for (int64_t i = 0; i < nblocks; ++i) {
        Hatrix::Matrix AY(block_size, rank + oversampling);
        for (int64_t j = 0; j < nblocks; ++j) {
          if (is_admissible(i, j)) {
            Hatrix::Matrix dense = Hatrix::generate_laplacend_matrix(randpts,
                                                                     block_size, block_size,
                                                                     i*block_size, j*block_size);
            Hatrix::matmul(dense, Y[j], AY);
          }
        }
        std::tie(Utemp, Stemp, Vtemp, error) = Hatrix::truncated_svd(AY, rank);
        U.insert(i, std::move(Utemp));
      }

      for (int64_t j = 0; j < nblocks; ++j) {
        Hatrix::Matrix YtA(rank + oversampling, block_size);
        for (int64_t i = 0; i < nblocks; ++i) {
          if (is_admissible(i, j)) {
            Hatrix::Matrix dense = Hatrix::generate_laplacend_matrix(randpts,
                                                                     block_size, block_size,
                                                                     i*block_size, j*block_size);
            Hatrix::matmul(Y[i], dense, YtA, true);
          }
        }
        std::tie(Utemp, Stemp, Vtemp, error) = Hatrix::truncated_svd(YtA, rank);
        V.insert(j, std::move(transpose(Vtemp)));
      }

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          if (is_admissible(i, j)) {
            Hatrix::Matrix dense = Hatrix::generate_laplacend_matrix(randpts,
                                                                     block_size, block_size,
                                                                     i*block_size, j*block_size);
            S.insert(i, j,
                     Hatrix::matmul(Hatrix::matmul(U[i], dense, true), V[j]));
          }
        }
      }
    }

    void factorize() {
      for (int block = 0; block < nblocks; ++block) {
        Hatrix::Matrix& diagonal = D(block, block);

        Hatrix::Matrix U_F = make_complement(U(block));
        Hatrix::Matrix V_F = make_complement(V(block));

        diagonal = matmul(matmul(U_F, diagonal, true, false), V_F);

        // in case of full rank, dont perform partial LU
        if (rank == diagonal.rows) { continue; }

        int c_size = diagonal.rows - rank;
        std::vector<Hatrix::Matrix> diagonal_splits = diagonal.split(std::vector<int64_t>(1, c_size),
                                                                     std::vector<int64_t>(1, c_size));
        Hatrix::Matrix& Dcc = diagonal_splits[0];
        Hatrix::Matrix& Dco = diagonal_splits[1];
        Hatrix::Matrix& Doc = diagonal_splits[2];
        Hatrix::Matrix& Doo = diagonal_splits[3];

        Hatrix::lu(Dcc);
        solve_triangular(Dcc, Dco, Hatrix::Left, Hatrix::Lower, true, false, 1.0);
        solve_triangular(Dcc, Doc, Hatrix::Right, Hatrix::Upper, false, false, 1.0);
        matmul(Doc, Dco, Doo, false, false, -1.0, 1.0);
      }

      // Merge unfactorized portions.
      Hatrix::Matrix last(rank * nblocks, rank * nblocks);
      std::vector<Hatrix::Matrix> last_splits = last.split(nblocks, nblocks);

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          if (is_admissible(i, j)) {
            last_splits[i * nblocks + j] = S(i, j);
          }
        }
      }
      Hatrix::lu(last);
    }

    Hatrix::Matrix solve(Hatrix::Matrix& b) {
      Hatrix::Matrix x(b);

      return x;
    }

    double construction_error(const randvec_t& randpts) {
      double error = 0, fnorm = 0;
      int block_size = N / nblocks;

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          if (!is_admissible(i, j)) {
            double dense_error = Hatrix::norm(D(i, j) -
                                           Hatrix::generate_laplacend_matrix(randpts, block_size, block_size,
                                                                             block_size * i, block_size * j));
            error += pow(dense_error, 2);
          }
          else {
            Matrix& Ubig = U(i);
            Matrix& Vbig = V(j);
            Matrix expected = matmul(matmul(Ubig, S(i, j)), Vbig, false, true);
            Matrix actual = Hatrix::generate_laplacend_matrix(randpts, block_size, block_size,
                                                              i * block_size, j * block_size);
            error += pow(Hatrix::norm(expected - actual), 2);
          }
        }
      }
      return std::sqrt(error / N / N);
    }
  };

}

int main(int argc, char** argv) {
  int64_t N = atoi(argv[1]);
  int64_t nblocks = atoi(argv[2]);
  int64_t rank = atoi(argv[3]);
  int64_t admis = atoi(argv[4]);

  Hatrix::Context::init();
  randvec_t randpts;
  randpts.push_back(equally_spaced_vector(N, 0.0, 1.0 * N)); // 1D
  randpts.push_back(equally_spaced_vector(N, 0.0, 1.0 * N)); // 2D
  randpts.push_back(equally_spaced_vector(N, 0.0, 1.0 * N)); // 3D

  if (N % nblocks != 0) {
    std::cout << "N % nblocks != 0. Aborting.\n";
    abort();
  }

  Hatrix::Matrix b = Hatrix::generate_random_matrix(N, 1);

  Hatrix::BLR2 A(randpts, N, nblocks, rank, admis);
  double construct_error = A.construction_error(randpts);
  A.factorize();
  Hatrix::Matrix x = A.solve(b);

  // Verification with dense solver.
  Hatrix::Matrix Adense = Hatrix::generate_laplacend_matrix(randpts, N, N, 0, 0);
  Hatrix::Matrix x_solve(b);
  Hatrix::lu(Adense);
  Hatrix::solve_triangular(Adense, x_solve, Hatrix::Left, Hatrix::Lower, true);
  Hatrix::solve_triangular(Adense, x_solve, Hatrix::Left, Hatrix::Upper, false);

  double solve_error = std::sqrt(pow(Hatrix::norm(x - x_solve), 2) / N);

  Hatrix::Context::finalize();

  std::cout << "N: " << N << " rank: " << rank << " nblocks: " << nblocks << " admis: " <<  admis
            << " construct error: " << construct_error << "\n";
}
