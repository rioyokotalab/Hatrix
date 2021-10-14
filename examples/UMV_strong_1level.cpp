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

    void left_lower_trsm_solve(Matrix& x, std::vector<Matrix>& x_split, int row, int c_size, int block_size) {
      auto D_splits = D(row, row).split(std::vector<int64_t>(1, c_size),
                                        std::vector<int64_t>(1, c_size));
      Matrix temp(x_split[row]);

      auto temp_splits = temp.split(std::vector<int64_t>(1, c_size), {});
      solve_triangular(D_splits[0], temp_splits[0], Hatrix::Left, Hatrix::Lower, true);
      matmul(D_splits[2], temp_splits[0], temp_splits[1], false, false, -1.0, 1.0);
      x_split[row] = temp;
    }

    void left_upper_trsm_solve(Matrix& x, std::vector<Matrix>& x_split, int row, int c_size, int block_size) {
      auto D_splits = D(row, row).split(std::vector<int64_t>(1, c_size),
                                        std::vector<int64_t>(1, c_size));
      Matrix temp(x_split[row]);
      auto temp_splits = temp.split(std::vector<int64_t>(1, c_size), {});
      matmul(D_splits[1], temp_splits[1], temp_splits[0], false, false, -1.0, 1.0);
      solve_triangular(D_splits[0], temp_splits[0], Hatrix::Left, Hatrix::Upper, false);
      x_split[row] = temp;
    }

    void permute_forward(Matrix& x, int block_size, int c_size) {
      int offset = c_size * nblocks;
      Matrix temp(x);

      for (int block = 0; block < nblocks; ++block) {
        for (int i = 0; i < c_size; ++i) {
          temp(c_size * block + i, 0) = x(block_size * block + i, 0);
        }
        for (int i = 0; i < rank; ++i) {
          temp(block * rank + offset + i, 0) = x(block_size * block + c_size + i, 0);
        }
      }

      x = temp;
    }

    void permute_back(Matrix& x, int block_size, int c_size) {
      int offset = c_size * nblocks;
      Matrix temp(x);

      for (int block = 0; block < nblocks; ++block) {
        for (int i = 0; i < c_size; ++i) {
          temp(block_size * block + i, 0) = x(c_size * block + i, 0);
        }

        for (int i = 0; i < rank; ++i) {
          temp(block_size * block + c_size + i, 0) = x(offset + rank * block + i, 0);
        }
      }

      x = temp;
    }


    Matrix make_complement(const Hatrix::Matrix &Q) {
      Matrix Q_F(Q.rows, Q.rows);
      Matrix Q_full, R;
       std::tie(Q_full, R) = qr(Q, Hatrix::Lapack::QR_mode::Full, Hatrix::Lapack::QR_ret::OnlyQ);

      for (int i = 0; i < Q_F.rows; ++i) {
        for (int j = 0; j < Q_F.cols - Q.cols; ++j) {
          Q_F(i, j) = Q_full(i, j + Q.cols);
        }
      }

      for (int i = 0; i < Q_F.rows; ++i) {
        for (int j = 0; j < Q.cols; ++j) {
          Q_F(i, j + (Q_F.cols - Q.cols)) = Q(i, j);
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
          std::cout << "left factor: " << block << " icol:  " << icol << std::endl;
          D(block, icol).print();
          auto lower_strip_splits = D(block, icol).split(std::vector<int64_t>(1, c_size),
                                                         std::vector<int64_t>(1, c_size));
          Hatrix::Matrix& Lco = lower_strip_splits[1];
          solve_triangular(Dcc, Lco, Hatrix::Left, Hatrix::Lower, true, false, 1.0);

          // Dcc.print();
          // std::cout << "AFTER left factor: " << block << " icol:  " << icol << std::endl;
          // D(block, icol).print();
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
                     Hatrix::matmul(Hatrix::matmul(U(i), dense, true), V(j)));
            // S.insert(i, j, std::move(dense));
          }
        }
      }
    }

    // Perform factorization assuming the permuted form of the BLR2 matrix.
    Matrix factorize() {
      int block_size = N / nblocks;
      int c_size = block_size - rank;
      RowColMap<Matrix> F; // store fill-in blocks

      for (int block = 0; block < nblocks; ++block) {
        auto diagonal_splits = D(block, block).split(std::vector<int64_t>(1, c_size),
                                                     std::vector<int64_t>(1, c_size));
        Matrix& Dcc = diagonal_splits[0];

        lu(Dcc);

        // Reduce the large cc off-diagonals on the right.
        for (int icol = block+1; icol < nblocks; ++icol) {
          auto right_splits = D(block, icol).split(std::vector<int64_t>(1, c_size),
                                                  std::vector<int64_t>(1, c_size));
          Matrix& cc_right = right_splits[0];
          solve_triangular(Dcc, cc_right, Hatrix::Left, Hatrix::Lower, true, false, 1.0);
        }

        // Reduce the large cc off-diagonals on the bottom.
        for (int irow = block+1; irow < nblocks; ++irow) {
          auto bottom_splits = D(irow, block).split(std::vector<int64_t>(1, c_size),
                                                    std::vector<int64_t>(1, c_size));
          Matrix& cc_bottom = bottom_splits[0];
          solve_triangular(Dcc, cc_bottom, Hatrix::Right, Hatrix::Upper, false, false, 1.0);
        }

        // Reduce the small co vertical strips to the right.
        for (int icol = 0; icol < nblocks; ++icol) {
          auto right_splits = D(block, icol).split(std::vector<int64_t>(1, c_size),
                                                   std::vector<int64_t>(1, c_size));
          Matrix& co_right = right_splits[1];
          solve_triangular(Dcc, co_right, Hatrix::Left, Hatrix::Lower, true, false, 1.0);
        }

        // Reduce the small oc horizontal stips to the bottom.
        for (int irow = 0; irow < nblocks; ++irow) {
          auto bottom_splits = D(irow, block).split(std::vector<int64_t>(1, c_size),
                                                std::vector<int64_t>(1, c_size));
          Matrix& co_bottom = bottom_splits[2];
          solve_triangular(Dcc, co_bottom, Hatrix::Right, Hatrix::Upper, false, false, 1.0);
        }

        // Compute schur's compliments for cc blocks
        for (int irow = block+1; irow < nblocks; ++irow) {
          for (int icol = block+1; icol < nblocks; ++icol) {
            auto bottom_splits = D(irow, block).split(std::vector<int64_t>(1, c_size),
                                                     std::vector<int64_t>(1, c_size));
            auto right_splits = D(block, icol).split(std::vector<int64_t>(1, c_size),
                                                      std::vector<int64_t>(1, c_size));
            auto reduce_splits = D(irow, icol).split(std::vector<int64_t>(1, c_size),
                                                     std::vector<int64_t>(1, c_size));
            matmul(bottom_splits[0], right_splits[0], reduce_splits[0], false, false, -1.0, 1.0);
          }
        }

        // Compute schur's compliments for co blocks from cc and co blocks (right side of the matrix)
        for (int irow = block+1; irow < nblocks; ++irow) {
          for (int icol = 0; icol < nblocks; ++icol) {
            auto bottom_splits = D(irow, block).split(std::vector<int64_t>(1, c_size),
                                                      std::vector<int64_t>(1, c_size));
            auto right_splits = D(block, icol).split(std::vector<int64_t>(1, c_size),
                                                     std::vector<int64_t>(1, c_size));
            auto reduce_splits = D(irow, icol).split(std::vector<int64_t>(1, c_size),
                                                     std::vector<int64_t>(1, c_size));
            matmul(bottom_splits[0], right_splits[1], reduce_splits[1], false, false, -1.0, 1.0);
          }
        }

        // Compute Schur's compliments for oc blocks from cc and oc blocks (bottom side of the matrix)
        for (int icol = block+1; icol < nblocks; ++icol) {
          for (int irow = 0; irow < nblocks; ++irow) {
            auto bottom_splits = D(irow, block).split(std::vector<int64_t>(1, c_size),
                                                      std::vector<int64_t>(1, c_size));
            auto right_splits = D(block, icol).split(std::vector<int64_t>(1, c_size),
                                                     std::vector<int64_t>(1, c_size));
            auto reduce_splits = D(irow, icol).split(std::vector<int64_t>(1, c_size),
                                                     std::vector<int64_t>(1, c_size));
            matmul(bottom_splits[2], right_splits[0], reduce_splits[2], false, false, -1.0, 1.0);
          }
        }

        // Compute Schur's compliments for oo blocks from oc and co blocks
        for (int irow = 0; irow < nblocks; ++irow) {
          for (int icol = 0; icol < nblocks; ++icol) {
            auto bottom_splits = D(irow, block).split(std::vector<int64_t>(1, c_size),
                                                     std::vector<int64_t>(1, c_size));
            auto right_splits = D(block, icol).split(std::vector<int64_t>(1, c_size),
                                                      std::vector<int64_t>(1, c_size));
            auto reduce_splits = D(irow, icol).split(std::vector<int64_t>(1, c_size),
                                                     std::vector<int64_t>(1, c_size));
            matmul(bottom_splits[2], right_splits[1], reduce_splits[3], false, false, -1.0, 1.0);
          }
        }
      }

      // Merge unfactorized portions.
      Matrix last(rank * nblocks, rank * nblocks);
      auto last_splits = last.split(nblocks, nblocks);

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          if (is_admissible(i, j)) {
            last_splits[i * nblocks + j] = S(i, j);
          }
          else {
            auto dense_splits = D(i, j).split(std::vector<int64_t>(1, c_size),
                                              std::vector<int64_t>(1, c_size));
            last_splits[i * nblocks + j] = dense_splits[3];
          }
        }
      }

      lu(last);

      return last;
    }

    Hatrix::Matrix solve(Matrix& b, const Matrix& last) {
      int64_t block_size = N / nblocks;
      int64_t c_size = block_size - rank;
      Hatrix::Matrix x(b);
      std::vector<Matrix> x_split = x.split(nblocks, 1);

      for (int irow = 0; irow < nblocks; ++irow) {
        // auto U_F = make_complement(U(irow));
        // // TODO: Figure out how to make this work only with views. Too confusing now.
        // Matrix temp = matmul(U_F, x_split[irow], true);
        // for (int64_t i = 0; i < block_size; ++i) {
        //   x(irow * block_size + i, 0) = temp(i, 0);
        // }

        if (rank == block_size) { continue; }



        // As shown in Eq. 38 in the paper.
        // Perform TRSM between current diagonal block and corresponding part of RHS.
        left_lower_trsm_solve(x, x_split, irow, c_size, block_size);
        // Multiply lower left blocks with the current diagonal block.
        for (int icol = 0; icol < irow; ++icol) {
          if (!is_admissible(irow, icol)) {
            // matmul(D(irow, icol), x_split[icol], x_split[irow], false, false, -1.0, 1.0);
            D(irow, icol).print();
            auto D_splits = D(irow, icol).split({}, std::vector<int64_t>(1, c_size));

            Matrix x_icol(x_split[icol]);
            auto x_icol_splits = x_icol.split(std::vector<int64_t>(1, c_size), {});
            matmul(D_splits[0], x_icol_splits[0], x_split[irow], false, false, -1.0, 1.0);
          }
        }

        // Multiply the upper row strips from the partial factorization and reduce the vector
        for (int icol = irow + 1; icol < nblocks; ++icol) {
          if (!is_admissible(irow, icol)) {
            auto D_splits = D(irow, icol).split(std::vector<int64_t>(1, c_size),
                                                std::vector<int64_t>(1, c_size));
            Matrix x_icol(x_split[icol]);
            auto x_icol_splits = x_icol.split(std::vector<int64_t>(1, c_size), {});

            Matrix x_irow(x_split[irow]);
            auto x_irow_splits = x_irow.split(std::vector<int64_t>(1, c_size), {});

            matmul(D_splits[2], x_icol_splits[0], x_irow_splits[1], false, false, -1.0, 1.0);
            x_split[irow] = x_irow;
          }
        }
      }

      permute_forward(x, block_size, c_size);

      auto permute_splits = x.split(std::vector<int64_t>(1, c_size * nblocks), {});
      solve_triangular(last, permute_splits[1], Hatrix::Left, Hatrix::Lower, true);
      solve_triangular(last, permute_splits[1], Hatrix::Left, Hatrix::Upper, false);

      permute_back(x, block_size, c_size);

      for (int irow = nblocks-1; irow >= 0; --irow) {

        // Reduce the lower strips
        for (int icol = nblocks-1; icol > irow; --icol) {
          if (!is_admissible(icol, irow)) {
            std::cout << "BACK irow : " << irow << " icol: " << icol << std::endl;
            auto D_splits = D(icol, irow).split(std::vector<int64_t>(1, c_size), {});
            Matrix x_irow(x_split[icol]);
            auto x_irow_splits = x_irow.split(std::vector<int64_t>(1, c_size), {});

            Matrix x_icol(x_split[irow]);
            auto x_icol_splits = x_icol.split(std::vector<int64_t>(1, c_size), {});

            matmul(D_splits[1], x_irow_splits[1], x_icol_splits[0], false, false, -1.0, 1.0);
            x_split[irow] = x_icol;
          }
        }

        if (rank != block_size) {
          for (int icol = nblocks-1; icol > irow; --icol) {
            if (!is_admissible(irow, icol)) {
              // matmul(D(irow, icol), x_split[icol], x_split[irow], false, false, -1.0, 1.0);

              auto D_splits = D(irow, icol).split(std::vector<int64_t>(1, c_size), {});
              Matrix x_irow(x_split[irow]);
              auto x_irow_splits = x_irow.split(std::vector<int64_t>(1, c_size), {});

              matmul(D_splits[0], x_split[icol], x_irow_splits[0], false, false, -1.0, 1.0);

              x_split[irow] = x_irow;
            }
          }
        }



        left_upper_trsm_solve(x, x_split, irow, c_size, block_size);
        // auto V_F = make_complement(V(irow));
        // Matrix temp = matmul(V_F, x_split[irow]);
        // for (int i = 0; i < block_size; ++i) {
        //   x(irow * block_size + i, 0) = temp(i, 0);
        // }
      }

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
  // randpts.push_back(Hatrix::equally_spaced_vector(N, 0.0, 1.0 * N)); // 2D
  // randpts.push_back(Hatrix::equally_spaced_vector(N, 0.0, 1.0 * N)); // 3D

  if (N % nblocks != 0) {
    std::cout << "N % nblocks != 0. Aborting.\n";
    abort();
  }

  Hatrix::Matrix b = Hatrix::generate_random_matrix(N, 1);

  Hatrix::BLR2 A(randpts, N, nblocks, rank, admis);
  double construct_error = A.construction_relative_error(randpts);
  Hatrix::Matrix last = A.factorize();
  // last.print();
  Hatrix::Matrix x = A.solve(b, last);

  // Verification with dense solver.
  Hatrix::Matrix Adense = Hatrix::generate_laplacend_matrix(randpts, N, N, 0, 0);
  Hatrix::Matrix x_solve(b);
  Hatrix::lu(Adense);
  Adense.print();
  Hatrix::solve_triangular(Adense, x_solve, Hatrix::Left, Hatrix::Lower, true);
  Hatrix::solve_triangular(Adense, x_solve, Hatrix::Left, Hatrix::Upper, false);

  std::cout << "x real:\n";
  x.print();

  std::cout << "x solve:\n";
  x_solve.print();

  std::cout << "diff:\n";
  (x_solve - x).print();


  double solve_error = Hatrix::norm(x - x_solve) / Hatrix::norm(x_solve);

  Hatrix::Context::finalize();

  std::cout << "N: " << N << " rank: " << rank << " nblocks: " << nblocks << " admis: " <<  admis
            << " construct error: " << construct_error << " solve error: " << solve_error << "\n";
}
