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

    void permute_forward(Matrix& x, int64_t block_size) {
      int64_t offset = 0;
      for (int i = 0; i < nblocks; ++i) {
        offset += (block_size - U(i).cols);
      }
      Matrix temp(x);

      int64_t c_size_offset = 0, rank_offset = 0;
      for (int block = 0; block < nblocks; ++block) {
        int64_t level_rank = U(block).cols;
        int64_t c_size = block_size - level_rank;

        // Copy the compliment part of the RHS vector.
        for (int i = 0; i < c_size; ++i) {
          temp(c_size_offset + i, 0) = x(block_size * block + i, 0);
        }
        // Copy the rank part of the RHS vector.
        for (int i = 0; i < level_rank; ++i) {
          temp(rank_offset + offset + i, 0) = x(block_size * block + c_size + i, 0);
        }

        c_size_offset += c_size;
        rank_offset += level_rank;
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
            // S.insert(i, j, std::move(dense));
          }
        }
      }
    }

    // Perform factorization assuming the permuted form of the BLR2 matrix.
    Matrix factorize() {
      int block_size = N / nblocks;
      RowColMap<Matrix> Loc, Uco;      // fill-ins of the small strips on the top and bottom.

      for (int block = 0; block < nblocks; ++block) {
        for (int icol = 0; icol < nblocks; ++icol) {
          if (!is_admissible(block, icol)) {
            Matrix U_F = make_complement(U(block));
            D(block, icol) = matmul(U_F, D(block, icol), true);
          }
        }

        for (int irow = 0; irow < nblocks; ++irow) {
          if (!is_admissible(irow, block)) {
            Matrix V_F = make_complement(V(irow));
            D(irow, block) = matmul(D(irow, block), V_F);
          }
        }

        // Expanded rank as a result of fill-in compression might be different at this level.
        assert(U(block).cols == V(block).cols);
        int64_t row_rank = U(block).cols, col_rank = V(block).cols;

        // The diagonal block is split along the row and column using the extended rank.
        auto diagonal_splits = D(block, block).split(std::vector<int64_t>(1, block_size - row_rank),
                                                     std::vector<int64_t>(1, block_size - col_rank));
        Matrix& Dcc = diagonal_splits[0];
        lu(Dcc);
        solve_triangular(Dcc, diagonal_splits[1], Hatrix::Left, Hatrix::Lower, true, false, 1.0);
        solve_triangular(Dcc, diagonal_splits[2], Hatrix::Right, Hatrix::Upper, false, false, 1.0);

        // Compute Schur's compliments for oo blocks from oc and co blocks.
        for (int irow = 0; irow < nblocks; ++irow) {
          for (int icol = 0; icol < nblocks; ++icol) {
            if (is_admissible(block, icol) || is_admissible(irow, block) || is_admissible(irow, icol)) { continue; }
            int64_t row_rank = U(irow).cols;
            int64_t col_rank = V(block).cols;
            auto bottom_splits = D(irow, block).split(std::vector<int64_t>(1, block_size - row_rank),
                                                     std::vector<int64_t>(1, block_size - col_rank));
            row_rank = U(block).cols;
            col_rank = V(icol).cols;
            auto right_splits = D(block, icol).split(std::vector<int64_t>(1, block_size - row_rank),
                                                     std::vector<int64_t>(1, block_size - col_rank));

            row_rank = U(irow).cols;
            col_rank = V(icol).cols;
            auto reduce_splits = D(irow, icol).split(std::vector<int64_t>(1, block_size - row_rank),
                                                     std::vector<int64_t>(1, block_size - col_rank));
            matmul(bottom_splits[2], right_splits[1], reduce_splits[3], false, false, -1.0, 1.0);
          }
        }

        // Reduce the large cc off-diagonals on the same row.
        for (int icol = block+1; icol < nblocks; ++icol) {
          if (is_admissible(block, icol)) { continue; }
          // The splitting for the inadmissible blocks on the right of the diagonal block
          // happens with differnt row and col rank since the U and V over here can have
          // different ranks due to fill-in compression.s
          int64_t row_rank = U(block).cols;
          int64_t col_rank = V(icol).cols;
          auto right_splits = D(block, icol).split(std::vector<int64_t>(1, block_size - row_rank),
                                                   std::vector<int64_t>(1, block_size - col_rank));
          solve_triangular(Dcc, right_splits[0], Hatrix::Left, Hatrix::Lower, true);
          solve_triangular(Dcc, right_splits[1], Hatrix::Left, Hatrix::Lower, true);
        }

        // Reduce the large cc off-diagonals on the same column.
        for (int irow = block+1; irow < nblocks; ++irow) {
          if (is_admissible(irow, block)) { continue; }
          int64_t row_rank = U(irow).cols;
          int64_t col_rank = V(block).cols;
          auto bottom_splits = D(irow, block).split(std::vector<int64_t>(1, block_size - row_rank),
                                                    std::vector<int64_t>(1, block_size - col_rank));
          solve_triangular(Dcc, bottom_splits[0], Hatrix::Right, Hatrix::Upper, false);
          solve_triangular(Dcc, bottom_splits[2], Hatrix::Right, Hatrix::Upper, false);
        }

        // Compute schur's compliments for co blocks from cc and co blocks (right side of the matrix)
        for (int irow = block+1; irow < nblocks; ++irow) {
          for (int icol = 0; icol < nblocks; ++icol) {
            if (is_admissible(block, icol) || is_admissible(irow, block) || is_admissible(irow, icol)) { continue; }
            int64_t row_rank = U(irow).cols;
            int64_t col_rank = V(block).cols;

            auto bottom_splits = D(irow, block).split(std::vector<int64_t>(1, block_size - row_rank),
                                                      std::vector<int64_t>(1, block_size - col_rank));

            row_rank = U(block).cols;
            col_rank = V(icol).cols;
            auto right_splits = D(block, icol).split(std::vector<int64_t>(1, block_size - row_rank),
                                                     std::vector<int64_t>(1, block_size - col_rank));

            row_rank = U(irow).cols;
            col_rank = V(icol).cols;
            auto reduce_splits = D(irow, icol).split(std::vector<int64_t>(1, block_size - row_rank),
                                                     std::vector<int64_t>(1, block_size - col_rank));
            matmul(bottom_splits[0], right_splits[1], reduce_splits[1], false, false, -1.0, 1.0);
          }
        }

        // Compute Schur's compliments for oc blocks from cc and oc blocks (bottom side of the matrix)
        for (int icol = block+1; icol < nblocks; ++icol) {
          for (int irow = 0; irow < nblocks; ++irow) {
            if (is_admissible(block, icol) || is_admissible(irow, block) || is_admissible(irow, icol)) { continue; }
            int64_t row_rank = U(irow).cols;
            int64_t col_rank = V(block).cols;
            auto bottom_splits = D(irow, block).split(std::vector<int64_t>(1, block_size - row_rank),
                                                      std::vector<int64_t>(1, block_size - col_rank));

            row_rank = U(block).cols;
            col_rank = V(icol).cols;
            auto right_splits = D(block, icol).split(std::vector<int64_t>(1, block_size - row_rank),
                                                     std::vector<int64_t>(1, block_size - col_rank));

            row_rank = U(irow).cols;
            col_rank = V(icol).cols;
            auto reduce_splits = D(irow, icol).split(std::vector<int64_t>(1, block_size - row_rank),
                                                     std::vector<int64_t>(1, block_size - col_rank));
            matmul(bottom_splits[2], right_splits[0], reduce_splits[2], false, false, -1.0, 1.0);
          }
        }

        // Compute fill-in blocks between co stips on the right side and cc blocks in the middle.
        // These fill-in blocks are of size <(nb - rank) x rank> and do not contribute to worsening
        // the scalability of the algorithm. I don't know why they are not compressed. Refer to pg. 320
        // in Sameer's H-matrix notes for detailed studies of the fill-in procedure in UMV and IFMM.
        for (int irow = block+1; irow < nblocks; ++irow) {
          for (int icol = 0; icol < nblocks; ++icol) {
            Matrix matrix(block_size - rank, rank);

            if (!is_admissible(irow, block)) {
              // This block exists on lower side of the matrix. It is multiplied by VF(block).
              int64_t row_rank = U(irow).cols;
              int64_t col_rank = V(block).cols;
              auto cc_splits = D(irow, block).split(std::vector<int64_t>(1, block_size - row_rank),
                                                    std::vector<int64_t>(1, block_size - col_rank));

              // Fill-in exists between blocks that were there before.
              if (!is_admissible(block, icol) && is_admissible(irow, icol)) {
                // This is the block on the right. It is multiplied by UF(block), so it split along
                // the column by U(block).cols.
                row_rank = U(block).cols;
                col_rank = V(icol).cols;
                auto co_splits = D(block, icol).split(std::vector<int64_t>(1, block_size - row_rank),
                                                      std::vector<int64_t>(1, block_size - col_rank));

                std::cout << "filling in co\n";
                matmul(cc_splits[0], co_splits[1], matrix, false, false, -1.0, 0.0);
                Uco.insert(irow, icol, std::move(matrix));
              }
              // Fill-in exists between previously created fill-in and a cc block on the inside.
              else if (Uco.exists(block, icol) && is_admissible(irow, icol)) {
                std::cout << "filling in co\n";
                matmul(cc_splits[0], Uco(block, icol), matrix, false, false, -1.0, 0.0);
                Uco.insert(irow, icol, std::move(matrix));
              }
            }
          }
        }

        // Compute fill-in blocks between oc strips on the bottom side and cc blocks in the middle.
        for (int irow = 0; irow < nblocks; ++irow) {
          for (int icol = block+1; icol < nblocks; ++icol) {
            if (!is_admissible(block, icol)) {
              Matrix matrix(rank, block_size - rank);

              int64_t row_rank = U(block).cols;
              int64_t col_rank = V(icol).cols;
              auto cc_splits = D(block, icol).split(std::vector<int64_t>(1, block_size - row_rank),
                                                    std::vector<int64_t>(1, block_size - col_rank));
              if (!is_admissible(irow, block) && is_admissible(irow, icol)) {
                std::cout << "filling oc\n";
                row_rank = U(irow).cols;
                col_rank = V(block).cols;
                auto oc_splits = D(irow, block).split(std::vector<int64_t>(1, block_size - row_rank),
                                                      std::vector<int64_t>(1, block_size - col_rank));
                matmul(oc_splits[2], cc_splits[0], matrix, false, false, -1.0, 0.0);
                Loc.insert(irow, icol, std::move(matrix));
              }
              else if (Loc.exists(irow, block) && is_admissible(irow, icol)) {
                std::cout << "filling oc\n";
                matmul(Loc(irow, block), cc_splits[0], matrix, false, false, -1.0, 0.0);
                Loc.insert(irow, icol, std::move(matrix));
              }
            }
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
            int64_t row_rank = U(i).cols;
            int64_t col_rank = V(j).cols;

            auto D_splits = D(i, j).split(std::vector<int64_t>(1, block_size - row_rank),
                                          std::vector<int64_t>(1, block_size - col_rank));
            last_splits[i * nblocks + j] = D_splits[3];
          }
        }
      }

      lu(last);

      return last;
    }

    Hatrix::Matrix solve(Matrix& b, Matrix& last) {
      int64_t block_size = N / nblocks;
      int64_t c_size = block_size - rank;
      Hatrix::Matrix x(b);
      std::vector<Matrix> x_split = x.split(nblocks, 1);

      // forward substitution with cc blocks
      for (int irow = 0; irow < nblocks; ++irow) {
        if (U.exists(irow)) {
          Matrix U_F = make_complement(U(irow));
          Matrix prod = matmul(U_F, x_split[irow], true);
          for (int64_t i = 0; i < block_size; ++i) {
            x(irow * block_size + i, 0) = prod(i, 0);
          }
        }

        assert(U(irow).cols == V(irow).cols);
        int64_t row_rank = U(irow).cols;

        auto D_splits = D(irow, irow).split(std::vector<int64_t>(1, block_size - row_rank),
                                            std::vector<int64_t>(1, block_size - row_rank));
        Matrix temp(x_split[irow]);
        auto temp_splits = temp.split(std::vector<int64_t>(1, block_size - row_rank), {});
        solve_triangular(D_splits[0], temp_splits[0], Hatrix::Left, Hatrix::Lower, true);
        matmul(D_splits[2], temp_splits[0], temp_splits[1], false, false, -1.0, 1.0);
        for (int64_t i = 0; i < block_size; ++i) {
          x(irow * block_size + i, 0) = temp(i, 0);
        }

        for (int icol = 0; icol < irow; ++icol) {
          if (is_admissible(irow, icol)) { continue; }
          int64_t col_rank = V(icol).cols;
          auto D_splits = D(irow, icol).split({}, std::vector<int64_t>(1, block_size - col_rank));

          Matrix x_irow(x_split[irow]);
          Matrix x_icol(x_split[icol]);
          auto x_irow_splits = x_irow.split(std::vector<int64_t>(1, block_size - row_rank), {});
          auto x_icol_splits = x_icol.split(std::vector<int64_t>(1, block_size - col_rank), {});

          matmul(D_splits[0], x_irow_splits[0], x_icol, false, false, -1.0, 1.0);
          for (int64_t i = 0; i < block_size; ++i) {
            x(icol * block_size + i, 0) = x_icol(i, 0);
          }
        }
      }

      permute_forward(x, block_size);

      int64_t c_size_offset = 0;
      for (int i = 0; i < nblocks; ++i) { c_size_offset += (block_size - U(i).cols); }

      auto permute_splits = x.split(std::vector<int64_t>(1, c_size_offset), {});
      solve_triangular(last, permute_splits[1], Hatrix::Left, Hatrix::Lower, true);
      solve_triangular(last, permute_splits[1], Hatrix::Left, Hatrix::Upper, false);

      permute_back(x, block_size);

      // backward substition using cc blocks
      for (int irow = nblocks-1; irow >= 0; --irow) {
        assert(U(irow).cols == V(irow).cols);

        int64_t row_rank = V(irow).cols;

        // Reduce the RHS with the cc blocks of the upper right part of the matrix.
        for (int icol = nblocks-1; icol > irow; --icol) {
          if (is_admissible(irow, icol)) { continue; }
          int64_t col_rank = V(icol).cols;

          auto D_splits = D(irow, icol).split(std::vector<int64_t>(1, block_size - row_rank), {});
          Matrix x_irow(x_split[irow]);
          Matrix x_icol(x_split[icol]);
          auto x_irow_splits = x_irow.split(std::vector<int64_t>(1, block_size - row_rank), {});
          auto x_icol_splits = x_icol.split(std::vector<int64_t>(1, block_size - col_rank), {});

          matmul(D_splits[0], x_icol, x_irow_splits[0], false, false, -1.0, 1.0);

          for (int64_t i = 0; i < block_size; ++i) {
            x(irow * block_size + i, 0) = x_irow(i, 0);
          }
        }

        auto block_splits = D(irow, irow).split(std::vector<int64_t>(1, block_size - row_rank),
                                                std::vector<int64_t>(1, block_size - row_rank));
        Matrix x_irow(x_split[irow]);
        auto x_irow_splits = x_irow.split(std::vector<int64_t>(1, block_size - row_rank), {});
        matmul(block_splits[1], x_irow_splits[1], x_irow_splits[0], false, false, -1.0, 1.0);
        solve_triangular(block_splits[0], x_irow_splits[0], Hatrix::Left, Hatrix::Upper, false);
        for (int64_t i = 0; i < block_size; ++i) {
          x(irow * block_size + i, 0) = x_irow(i, 0);
        }

        if (V.exists(irow)) {
          auto V_F = make_complement(V(irow));
          Matrix prod = matmul(V_F, x_split[irow]);
          for (int i = 0; i < block_size; ++i) {
            x(irow * block_size + i, 0) = prod(i, 0);
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
  auto last = A.factorize();
  Hatrix::Matrix x = A.solve(b, last);

  // std::cout << "x:\n";
  // x.print();

  // Verification with dense solver.
  Hatrix::Matrix Adense = Hatrix::generate_laplacend_matrix(randpts, N, N, 0, 0);
  Hatrix::Matrix x_solve(b);
  Hatrix::lu(Adense);
  Hatrix::solve_triangular(Adense, x_solve, Hatrix::Left, Hatrix::Lower, true);
  Hatrix::solve_triangular(Adense, x_solve, Hatrix::Left, Hatrix::Upper, false);

  // std::cout << "x solve:\n";
  // x_solve.print();

  double solve_error = Hatrix::norm(x - x_solve) / Hatrix::norm(x_solve);

  Hatrix::Context::finalize();

  std::cout << "N: " << N << " rank: " << rank << " nblocks: " << nblocks << " admis: " <<  admis
            << " construct error: " << construct_error << " solve error: " << solve_error << "\n";
}
