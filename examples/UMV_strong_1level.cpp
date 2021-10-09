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
// UMV factorization using Miamiao Ma's method as shown in
// Accuracy Directly Controlled Fast Direct Solution of General H^2-matrices and Its
// Application to Solving Electrodynamics Volume Integral Equations

using randvec_t = std::vector<std::vector<double> >;

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

    void factorize_upper_strips(int block, int c_size, Hatrix::Matrix& Dcc) {
      for (int irow = 0; irow < block; ++irow) {
        if (!is_admissible(irow, block)) {
          std::vector<Hatrix::Matrix> upper_strip_splits = D(irow, block).split(std::vector<int64_t>(1, c_size),
                                                                                std::vector<int64_t>(1, c_size));
          Hatrix::Matrix& Roc = upper_strip_splits[2];
          solve_triangular(Dcc, Roc, Hatrix::Right, Hatrix::Upper, false, false, 1.0);
        }
      }
    }

    void factorize_left_strips(int block, int c_size, Hatrix::Matrix& Dcc) {
      for (int icol = 0; icol < block; ++icol) {
        if (!is_admissible(block, icol)) {
          std::vector<Hatrix::Matrix> lower_strip_splits = D(block, icol).split(std::vector<int64_t>(1, c_size),
                                                                                std::vector<int64_t>(1, c_size));
          Hatrix::Matrix& Lco = lower_strip_splits[1];
          solve_triangular(Dcc, Lco, Hatrix::Left, Hatrix::Lower, true, false, 1.0);
        }
      }
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
      int block_size = N / nblocks;
      RowColMap<Matrix> F; // store fill-in blocks

      for (int block = 0; block < nblocks; ++block) {


        // Diagonal block is always dense so obtain compliment matrices and perform partial LU
        // on it first.
        Hatrix::Matrix& diagonal = D(block, block);
        Hatrix::Matrix U_F = make_complement(U(block));
        Hatrix::Matrix V_F = make_complement(V(block));

        diagonal = matmul(matmul(U_F, diagonal, true, false), V_F);

        // in case of full rank, dont perform partial LU
        if (rank == diagonal.rows) { continue; }

        int c_size = diagonal.rows - rank;
        std::vector<Hatrix::Matrix> diagonal_splits = diagonal.split(std::vector<int64_t>(1, c_size),
                                                                     std::vector<int64_t>(1, c_size));
        Matrix& Dcc = diagonal_splits[0];
        Matrix& Dco = diagonal_splits[1];
        Matrix& Doc = diagonal_splits[2];
        Matrix& Doo = diagonal_splits[3];

        Hatrix::lu(Dcc);
        solve_triangular(Dcc, Dco, Hatrix::Left, Hatrix::Lower, true, false, 1.0);
        solve_triangular(Dcc, Doc, Hatrix::Right, Hatrix::Upper, false, false, 1.0);
        matmul(Doc, Dco, Doo, false, false, -1.0, 1.0);

        // TRSMs with the lower part of the diagonal block.
        // Multiply previously remaining unfactorized upper strips in the column
        // of the diagonal block. Fig. 4b in the paper.
        if (block > 0) {
          factorize_left_strips(block, c_size, Dcc);
        }

        // Multiply and TRSM right blocks.
        for (int icol = block + 1; icol < nblocks; ++icol) {
          if (!is_admissible(block, icol)) {
            Hatrix::Matrix& upper_right = D(block, icol);
            upper_right = matmul(U_F, upper_right, true, false);

            std::vector<Hatrix::Matrix> right_splits = upper_right.split(std::vector<int64_t>(1, c_size),
                                                                         std::vector<int64_t>(1, c_size));
            Matrix& Rcc = right_splits[0];
            Matrix& Rco = right_splits[1];
            Matrix& Roc = right_splits[2];
            Matrix& Roo = right_splits[3];

            solve_triangular(Dcc, Rcc, Hatrix::Left, Hatrix::Lower, true, false, 1.0);
            solve_triangular(Dcc, Rco, Hatrix::Left, Hatrix::Lower, true, false, 1.0);
            matmul(Doc, Rcc, Roc, false, false, -1.0, 1.0);
            matmul(Doc, Rco, Roo, false, false, -1.0, 1.0);
          }
        }

        // TRSMs with the upper part of the diagonal block.
        // Multiply previously remaining unfactorized lower strips in the row
        // of the diagonal block. Fig. 4b in the paper.
        if (block > 0) {
          factorize_upper_strips(block, c_size, Dcc);
        }

        // Multiply and TRSM lower blocks.
        for (int irow = block + 1; irow < nblocks; ++irow) {
          if (!is_admissible(irow, block)) {
            Hatrix::Matrix& lower_left = D(irow, block);
            lower_left = matmul(lower_left, V_F);

            std::vector<Hatrix::Matrix> left_splits = lower_left.split(std::vector<int64_t>(1, c_size),
                                                                       std::vector<int64_t>(1, c_size));
            Matrix& Lcc = left_splits[0];
            Matrix& Lco = left_splits[1];
            Matrix& Loc = left_splits[2];
            Matrix& Loo = left_splits[3];

            solve_triangular(Dcc, Lcc, Hatrix::Right, Hatrix::Upper, false, false, 1.0);
            solve_triangular(Dcc, Loc, Hatrix::Right, Hatrix::Upper, false, false, 1.0);
            matmul(Lcc, Dco, Lco, false, false, -1.0, 1.0);
            matmul(Loc, Dco, Loo, false, false, -1.0, 1.0);
          }
        }

        // Generate fill-in blocks and store temporarily.
        for (int irow = block + 1; irow < nblocks; ++irow) {
          for (int icol = block + 1; icol < nblocks; ++icol) {
            if (!is_admissible(irow, block) && !is_admissible(block, icol)) {
              Matrix& A_row_block = D(irow, block);
              Matrix& A_block_col = D(block, icol);

              auto A_row_block_splits = A_row_block.split(std::vector<int64_t>(1, c_size),
                                                          std::vector<int64_t>(1, c_size));
              auto A_block_col_splits = A_block_col.split(std::vector<int64_t>(1, c_size),
                                                          std::vector<int64_t>(1, c_size));

              Matrix fill(block_size, block_size);
              auto fill_splits = fill.split(std::vector<int64_t>(1, c_size),
                                            std::vector<int64_t>(1, c_size));

              matmul(A_row_block_splits[0], A_block_col_splits[0], fill_splits[0]);

              std::cout << "admis block: i-> " << irow << "," << block
                        << " j-> " << block << "," <<  icol << std::endl;
            }
          }
        }
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

    double construction_relative_error(const randvec_t& randpts) {
      double error = 0, dense_norm = 0;
      int block_size = N / nblocks;

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          Matrix actual = Hatrix::generate_laplacend_matrix(randpts, block_size, block_size,
                                                            i * block_size, j * block_size);
          dense_norm += pow(Hatrix::norm(actual), 2);

          if (!is_admissible(i, j)) {
            error += pow(Hatrix::norm(D(i, j) - actual), 2);
          }
          else {
            Matrix& Ubig = U(i);
            Matrix& Vbig = V(j);
            Matrix expected = matmul(matmul(Ubig, S(i, j)), Vbig, false, true);

            error += pow(Hatrix::norm(expected - actual), 2);
          }
        }
      }
      return std::sqrt(error / dense_norm);
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
  randpts.push_back(Hatrix::equally_spaced_vector(N, 0.0, 1.0 * N)); // 1D
  randpts.push_back(Hatrix::equally_spaced_vector(N, 0.0, 1.0 * N)); // 2D
  randpts.push_back(Hatrix::equally_spaced_vector(N, 0.0, 1.0 * N)); // 3D

  if (N % nblocks != 0) {
    std::cout << "N % nblocks != 0. Aborting.\n";
    abort();
  }

  Hatrix::Matrix b = Hatrix::generate_random_matrix(N, 1);

  Hatrix::BLR2 A(randpts, N, nblocks, rank, admis);
  double construct_error = A.construction_relative_error(randpts);
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
