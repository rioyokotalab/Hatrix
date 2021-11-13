#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cassert>

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
    RowColMap<Matrix> Loc, Uco;      // fill-ins of the small strips on the top and bottom.

    void permute_forward(Matrix& x, int64_t block_size) {
      int64_t c_size = block_size - rank;
      int64_t offset = c_size * nblocks;
      Matrix temp(x);

      int64_t c_size_offset = 0, rank_offset = 0;
      for (int block = 0; block < nblocks; ++block) {
        // Copy the compliment part of the RHS vector.
        for (int i = 0; i < c_size; ++i) {
          temp(c_size_offset + i, 0) = x(block_size * block + i, 0);
        }
        // Copy the rank part of the RHS vector.
        for (int i = 0; i < rank; ++i) {
          temp(rank_offset + offset + i, 0) = x(block_size * block + c_size + i, 0);
        }

        c_size_offset += c_size;
        rank_offset += rank;
      }

      x = temp;
    }

    void permute_back(Matrix& x, int64_t block_size) {
      int64_t offset = 0;
      for (int i = 0; i < nblocks; ++i) { offset += (block_size - V(i).cols); }
      Matrix temp(x);

      int64_t c_size_offset = 0, rank_offset = 0;
      for (int block = 0; block < nblocks; ++block) {
        int64_t c_size = block_size - V(block).cols;

        // Copy the compliment part of the vector.
        for (int i = 0; i < c_size; ++i) {
          temp(block_size * block + i, 0) = x(c_size_offset + i, 0);
        }

        // Copy the rank part of the vector.
        for (int i = 0; i < rank; ++i) {
          temp(block_size * block + c_size + i, 0) = x(offset + rank_offset + i, 0);
        }

        c_size_offset += c_size;
        rank_offset += V(block).cols;
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

  public:
    BLR2(const randvec_t& randpts, int64_t N, int64_t nblocks, int64_t rank, int64_t admis) :
      N(N), nblocks(nblocks), rank(rank), admis(admis) {
      int block_size = N / nblocks;

      for (int i = 0; i < nblocks; ++i) {

        // is_admissible.insert(i, i, std::abs(i - i) > admis);
        for (int j = 0; j < nblocks; ++j) {
          is_admissible.insert(i, j, std::abs(i - j) > admis);
        }
      }
      // for (int i = 0; i < nblocks-1; ++i) {
      //   is_admissible.insert(i+1, i, admis == 1 ? false : true);
      // }
      // is_admissible.insert(3, 2, false);
      // is_admissible.insert(2, 1, false);
      // is_admissible.insert(1, 0, false);
      //      is_admissible.insert(3, 1, false);
      // is_admissible.insert(2, 1, false);
      // is_admissible.insert(3, 2, true);

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          if (!is_admissible.exists(i, j)) {
            is_admissible.insert(i, j, true);
          }


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
        bool admissible_found = false;
        for (int64_t j = 0; j < nblocks; ++j) {
          if (is_admissible(i, j)) {
            admissible_found = true;
            Hatrix::Matrix dense = Hatrix::generate_laplacend_matrix(randpts,
                                                                     block_size, block_size,
                                                                     i*block_size, j*block_size);
            Hatrix::matmul(dense, Y[j], AY);
          }
        }
        std::tie(Utemp, Stemp, Vtemp, error) = Hatrix::truncated_svd(AY, rank);

        if (admissible_found) {
          U.insert(i, std::move(Utemp));
        }
      }

      for (int64_t j = 0; j < nblocks; ++j) {
        Hatrix::Matrix YtA(rank + oversampling, block_size);
        bool admissible_found = false;
        for (int64_t i = 0; i < nblocks; ++i) {
          if (is_admissible(i, j)) {
            admissible_found = true;
            Hatrix::Matrix dense = Hatrix::generate_laplacend_matrix(randpts,
                                                                     block_size, block_size,
                                                                     i*block_size, j*block_size);
            Hatrix::matmul(Y[i], dense, YtA, true);
          }
        }
        std::tie(Utemp, Stemp, Vtemp, error) = Hatrix::truncated_svd(YtA, rank);

        if (admissible_found) {
          V.insert(j, std::move(transpose(Vtemp)));
        }
      }

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          if (is_admissible(i, j)) {
            Hatrix::Matrix dense = Hatrix::generate_laplacend_matrix(randpts,
                                                                     block_size, block_size,
                                                                     i*block_size, j*block_size);
            S.insert(i, j,
                     Hatrix::matmul(Hatrix::matmul(U(i), dense, true), V(j)));
          }
        }
      }
    }

    // Perform factorization assuming the permuted form of the BLR2 matrix.
    Matrix factorize() {
      int block_size = N / nblocks;

      for (int block = 0; block < nblocks; ++block) {
        for (int icol = 0; icol < nblocks; ++icol) {
          if (!is_admissible(block, icol)) {
            Matrix U_F = make_complement(U(block));

            if (icol < block) {
              auto bottom_splits = D(block, icol).split({}, std::vector<int64_t>(1, block_size - rank));
              Matrix co = matmul(U_F, bottom_splits[1], true);
              bottom_splits[1] = co;
            }
            else {
              D(block, icol) = matmul(U_F, D(block, icol), true);
            }
          }
        }

        for (int irow = 0; irow < nblocks; ++irow) {
          if (!is_admissible(irow, block) && V.exists(block)) {
            Matrix V_F = make_complement(V(block));
            if (irow > block) {
              auto right_splits = D(irow, block).split(std::vector<int64_t>(1, block_size - rank), {});
              Matrix oc = matmul(right_splits[1], V_F);
              right_splits[1] = oc;
            }
            else {
              D(irow, block) = matmul(D(irow, block), V_F);
            }
          }
        }

        // Expanded rank as a result of fill-in compression might be different at this level.
        // The diagonal block is split along the row and column using the extended rank.
        auto diagonal_splits = D(block, block).split(std::vector<int64_t>(1, block_size - rank),
                                                     std::vector<int64_t>(1, block_size - rank));

        Matrix& Dcc = diagonal_splits[0];
        Matrix& Dco = diagonal_splits[1];
        Matrix& Doc = diagonal_splits[2];
        lu(Dcc);
        solve_triangular(Dcc, Dco, Hatrix::Left, Hatrix::Lower, true);
        solve_triangular(Dcc, Doc, Hatrix::Right, Hatrix::Upper, false);
        matmul(Doc, Dco, diagonal_splits[3], false, false, -1.0, 1.0);

        // Perform TRSM between A.UF blocks on the blocks behind the diagonal
        for (int icol = 0; icol < block; ++icol) {
          if (is_admissible(block, icol)) { continue; }
          auto left_splits = D(block, icol).split(std::vector<int64_t>(1, block_size - rank),
                                                  std::vector<int64_t>(1, block_size - rank));
          // TRSM between cc block on diagonal and co block here.
          solve_triangular(Dcc, left_splits[1], Hatrix::Left, Hatrix::Lower, true);
          // Use the lower part of the trapezoid.
          matmul(Doc, left_splits[1], left_splits[3], false, false, -1.0, 1.0);
        }

        // Perform lower TRSM between diagonal and the A.UF block on the right of the diagonal.
        for (int icol = block+1; icol < nblocks; ++icol) {
          if (is_admissible(block, icol)) { continue; }
          auto right_splits = D(block, icol).split(std::vector<int64_t>(1, block_size - rank), {});
          // TRSM between cc block on diagonal and c block on the right
          solve_triangular(Dcc, right_splits[0], Hatrix::Left, Hatrix::Lower, true);
          // Use the lower part of the trapezoid
          matmul(Doc, right_splits[0], right_splits[1], false, false, -1.0, 1.0);
        }

        // Perform upper TRSM between A.VF blocks above this diagonal blocks and cc of diagonal block.
        for (int irow = 0; irow < block; ++irow) {
          if (is_admissible(irow, block)) { continue; }
          auto top_splits = D(irow, block).split(std::vector<int64_t>(1, block_size - rank),
                                                 std::vector<int64_t>(1, block_size - rank));
          // TRSM between cc block on the diagonal and oc block above this diagonal block.
          solve_triangular(Dcc, top_splits[2], Hatrix::Right, Hatrix::Upper, false);
          // GEMM between oc block here and the
          matmul(top_splits[3], Dco, top_splits[3], false, false, -1.0, 1.0);
        }

        // Perform TRSM between A.VF blocks on the bottom of this diagonal block and the cc of the diagonal.
        for (int irow = block+1; irow < nblocks; ++irow) {
          if (is_admissible(irow, block)) { continue; }
          auto bottom_splits = D(irow, block).split({}, std::vector<int64_t>(1, block_size - rank));
          // TRSM between cc block on diagonal and c block here.
          solve_triangular(Dcc, bottom_splits[0], Hatrix::Right, Hatrix::Upper, false);
          // GEMM between c block here and co block from diagonal.
          matmul(bottom_splits[0], Dco, bottom_splits[1], false, false, -1.0, 1.0);
        }

        // Compute the schur's compliment between the reduced part of the A.UF block on the
        // upper right and reduced A.VF part of the bottom left.
        for (int irow = block+1; irow < nblocks; ++irow) {
          for (int icol = block+1; icol < nblocks; ++icol) {
            if (is_admissible(irow, block) || is_admissible(block, icol) || is_admissible(irow, icol)) { continue; }
            auto right_block = D(block, icol).split(std::vector<int64_t>(1, block_size - rank), {});
            auto bottom_block = D(irow, block).split({}, std::vector<int64_t>(1, block_size - rank));
            // std::cout << "reduce: " << irow << " " << icol << std::endl;
            matmul(bottom_block[0], right_block[0], D(irow, icol), false, false -1.0, 1.0);
          }
        }
      } // for (int block = 0; block < nblocks; ++block)

      // Merge unfactorized portions.
      Matrix last(nblocks * rank, nblocks * rank);
      auto last_splits = last.split(nblocks, nblocks);

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          if (is_admissible(i, j)) {
            last_splits[i * nblocks + j] = S(i, j);
          }
          else {
            auto D_splits = D(i, j).split(std::vector<int64_t>(1, block_size - rank),
                                          std::vector<int64_t>(1, block_size - rank));
            last_splits[i * nblocks + j] = D_splits[3];
          }
        }
      }

      return last;
    }

    Hatrix::Matrix solve(Matrix& b, Matrix& last) {
      int64_t block_size = N / nblocks;
      int64_t c_size = block_size - rank;
      Hatrix::Matrix x(b);
      std::vector<Matrix> x_split = x.split(nblocks, 1);

      // forward substitution with cc blocks
      for (int block = 0; block < nblocks; ++block) {
        if (U.exists(block)) {
          Matrix U_F = make_complement(U(block));
          Matrix prod = matmul(U_F, x_split[block], true);
          for (int64_t i = 0; i < block_size; ++i) {
            x(block * block_size + i, 0) = prod(i, 0);
          }
        }

        auto block_splits = D(block, block).split(std::vector<int64_t>(1, c_size),
                                            std::vector<int64_t>(1, c_size));
        Matrix temp(x_split[block]);
        auto temp_splits = temp.split(std::vector<int64_t>(1, c_size), {});
        solve_triangular(block_splits[0], temp_splits[0], Hatrix::Left, Hatrix::Lower, true);
        matmul(block_splits[2], temp_splits[0], temp_splits[1], false, false, -1.0, 1.0);
        for (int64_t i = 0; i < block_size; ++i) {
          x(block * block_size + i, 0) = temp(i, 0);
        }

        for (int irow = block+1; irow < nblocks; ++irow) {
          if (is_admissible(irow, block)) { continue; }
          auto D_splits = D(irow, block).split({}, std::vector<int64_t>(1, c_size));
          Matrix x_block(x_split[block]);
          auto x_block_splits = x_block.split(std::vector<int64_t>(1, c_size), {});
          matmul(D_splits[0], x_block_splits[0], x_split[irow], false, false, -1.0, 1.0);
        }
      }

      permute_forward(x, block_size);

      auto permute_splits = x.split(std::vector<int64_t>(1, (block_size - rank) * nblocks), {});
      Matrix copy(permute_splits[1]);
      auto s = lu_solve(last, copy);
      permute_splits[1] = s;

      permute_back(x, block_size);

      // backward substition using cc blocks
      for (int block = nblocks-1; block >= 0; --block) {
        // std::cout << "block: " << block << std::endl;

        for (int left_col = block-1; left_col >= 0; --left_col) {
          if (!is_admissible(block, left_col)) {
            // std::cout << "co update: b-> " << block << " u-> " << left_col << std::endl;
            auto D_splits = D(block, left_col).split(std::vector<int64_t>(1, block_size - rank),
                                                      std::vector<int64_t>(1, block_size - rank));
            Matrix x_block(x_split[block]);
            auto x_block_splits = x_block.split(std::vector<int64_t>(1, block_size - rank), {});
            Matrix x_left_col(x_split[left_col]);
            auto x_left_col_splits = x_left_col.split(std::vector<int64_t>(1, block_size - rank), {});

            matmul(D_splits[1], x_left_col_splits[1], x_block_splits[0], false, false, -1.0, 1.0);
            for (int64_t i = 0; i < block_size; ++i) {
              x(block * block_size + i, 0) = x_block(i, 0);
            }
          }
        }

        auto block_splits = D(block, block).split(std::vector<int64_t>(1, block_size - rank),
                                                std::vector<int64_t>(1, block_size - rank));
        Matrix x_block(x_split[block]);
        auto x_block_splits = x_block.split(std::vector<int64_t>(1, block_size - rank), {});
        matmul(block_splits[1], x_block_splits[1], x_block_splits[0], false, false, -1.0, 1.0);
        solve_triangular(block_splits[0], x_block_splits[0], Hatrix::Left, Hatrix::Upper, false);
        for (int64_t i = 0; i < block_size; ++i) {
          x(block * block_size + i, 0) = x_block(i, 0);
        }

        if (V.exists(block)) {
          auto V_F = make_complement(V(block));
          Matrix prod = matmul(V_F, x_split[block]);
          for (int i = 0; i < block_size; ++i) {
            x(block * block_size + i, 0) = prod(i, 0);
          }
        }
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

    void print_structure() {
      std::cout << "BLR " << nblocks << " x " << nblocks << std::endl;
      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          std::cout << " | " << is_admissible(i, j);
        }
        std::cout << " | \n";
      }
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
  randpts.push_back(Hatrix::equally_spaced_vector(N, 0.0, 1.0)); // 1D
  randpts.push_back(Hatrix::equally_spaced_vector(N, 0.0, 1.0)); // 2D
  randpts.push_back(Hatrix::equally_spaced_vector(N, 0.0, 1.0)); // 3D

  if (N % nblocks != 0) {
    std::cout << "N % nblocks != 0. Aborting.\n";
    abort();
  }

  Hatrix::Matrix b = Hatrix::generate_random_matrix(N, 1);

  Hatrix::BLR2 A(randpts, N, nblocks, rank, admis);
  A.print_structure();
  double construct_error = A.construction_relative_error(randpts);
  auto last = A.factorize();
  Hatrix::Matrix x = A.solve(b, last);

  // std::cout << "x:\n";
  // x.print();

  // Verification with dense solver.
  Hatrix::Matrix Adense = Hatrix::generate_laplacend_matrix(randpts, N, N, 0, 0);
  Hatrix::Matrix x_solve = lu_solve(Adense, b);
  // Hatrix::Matrix x_solve(b);
  // Hatrix::lu(Adense);
  // Hatrix::solve_triangular(Adense, x_solve, Hatrix::Left, Hatrix::Lower, true);
  // Hatrix::solve_triangular(Adense, x_solve, Hatrix::Left, Hatrix::Upper, false);



  // std::cout << "x solve:\n";
  // (x - x_solve).print();

  // last.print();

  double solve_error = Hatrix::norm(x - x_solve) / Hatrix::norm(x_solve);

  Hatrix::Context::finalize();

  std::cout << "N: " << N << " rank: " << rank << " nblocks: " << nblocks << " admis: " <<  admis
            << " construct error: " << construct_error << " solve error: " << solve_error << "\n";
}
