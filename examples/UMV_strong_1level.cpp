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

constexpr double PV = 1e-3;
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
        // for (int j = 0; j < nblocks; ++j) {
          is_admissible.insert(i, i, std::abs(i - i) > admis);
        // }
      }

      is_admissible.insert(2, 3, admis == 1 ? false : true);
      is_admissible.insert(3, 2, admis == 1 ? false : true);

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          if (!is_admissible.exists(i, j)) {
            is_admissible.insert(i, j, true);
          }
          if (!is_admissible(i, j)) {
            D.insert(i, j,
                     Hatrix::generate_laplacend_matrix(randpts,
                                                       block_size, block_size,
                                                       i*block_size, j*block_size, PV));
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
                                                                     i*block_size, j*block_size, PV);
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
                                                                     i*block_size, j*block_size, PV);
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
                                                                     i*block_size, j*block_size, PV);
            S.insert(i, j,
                     Hatrix::matmul(Hatrix::matmul(U(i), dense, true), V(j)));
          }
        }
      }
    }

    // Perform factorization assuming the permuted form of the BLR2 matrix.
    Matrix factorize(const randvec_t &randpts) {
      int block_size = N / nblocks;
      int c_size = block_size - rank;
      RowColMap<Matrix> F;      // fill-in blocks.

      for (int block = 0; block < nblocks; ++block) {
        if (block > 0) {
          Matrix Utemp, Stemp, Vtemp; double error;
          // Recompress fill-ins on the row and update row bases.
          Matrix Urow_bases_concat(U(block).rows, 0), Vcol_bases_concat(0, V(block).rows);
          bool update_bases = false;
          for (int icol = 0; icol < block; ++icol) {
            if (F.exists(block, icol)) {
              update_bases = true;
              std::tie(Utemp, Stemp, Vtemp, error) =
                Hatrix::truncated_svd(F(block, icol), rank);
              Matrix Ubases = matmul(U(block), S(block, icol));
              Matrix bases_concat = concat(Ubases, matmul(Utemp, Stemp), 1);
              Urow_bases_concat = concat(Urow_bases_concat, bases_concat, 1);

              // std::cout << "bl: " << block << " ic: " << icol << std::endl;
              // Matrix Vbases = matmul(S(block, icol), transpose(V(icol)));
              // Matrix Vbases_concat = concat(Vbases, matmul(Stemp, Vtemp), 0);
              // std::tie(Utemp, Stemp, Vtemp, error) =
              //   Hatrix::truncated_svd(Vbases_concat, rank);
              // V.erase(icol);
              // V.insert(icol, std::move(transpose(Vtemp)));

              F.erase(block, icol);
            }
          }
          if (update_bases) {
            std::tie(Utemp, Stemp, Vtemp, error) =
              Hatrix::truncated_svd(Urow_bases_concat, rank);
            // U.erase(block);
            // U.insert(block, std::move(Utemp));
          }
        }

        // Recompress fill-ins on the col and update col bases.
        if (block > 0) {
          Matrix Utemp, Stemp, Vtemp; double error;
          Matrix Urow_bases_concat(U(block).rows, 0), Vcol_bases_concat(0, V(block).rows);
          bool update_bases = false;
          for (int irow = 0; irow < block; ++irow) {
            if (F.exists(irow, block)) {
              update_bases = true;
              Matrix Vbases = matmul(S(irow, block), transpose(V(block)));
              std::tie(Utemp, Stemp, Vtemp, error) =
                Hatrix::truncated_svd(F(irow, block), rank);
              Matrix Vbases_concat = concat(Vbases, matmul(Stemp, Vtemp), 0);
              Vcol_bases_concat = concat(Vcol_bases_concat, Vbases_concat, 0);

              // Matrix Ubases = matmul(U(irow), S(irow, block));
              // Matrix Ubases_concat = concat(Ubases, matmul(Utemp, Stemp), 1);
              // std::tie(Utemp, Stemp, Vtemp, error) =
              //   Hatrix::truncated_svd(Ubases_concat, rank);
              // U.erase(irow);
              // U.insert(irow, std::move(Utemp));

              F.erase(irow, block);
            }
          }

          if (update_bases) {
            std::tie(Utemp, Stemp, Vtemp, error) = Hatrix::truncated_svd(Vcol_bases_concat,
                                                                         rank);
            // V.erase(block);
            // V.insert(block, std::move(transpose(Vtemp)));
          }
        }

        Matrix U_F = make_complement(U(block));
        Matrix V_F = make_complement(V(block));
        D(block, block) = matmul(matmul(U_F, D(block, block), true), V_F);

        // Multiplication of UF with block ahead of the diagonal block on the same row
        for (int icol = block+1; icol < nblocks; ++icol) {
          if (!is_admissible(block, icol)) {
            D(block, icol) = matmul(U_F, D(block, icol), true);
          }
        }

        // Multiplication of VF with block below this diagonal on the same column.
        for (int irow = block+1; irow < nblocks; ++irow) {
          if (!is_admissible(irow, block)) {
            D(irow, block) = matmul(D(irow, block), V_F);
          }
        }

        // Multiplication of UF with blocks before the diagonal block on the same row
        for (int icol = 0; icol < block; ++icol) {
          if (!is_admissible(block, icol)) {
            auto bottom_splits = D(block, icol).split({}, std::vector<int64_t>(1, c_size));
            Matrix o = matmul(U_F, bottom_splits[1], true);
            bottom_splits[1] = o;
          }
        }

        // Multiplication of VF with blocks before the diagoanl block on the same column.
        for (int irow = 0; irow < block; ++irow) {
          if (!is_admissible(irow, block)) {
            auto right_splits = D(irow, block).split(std::vector<int64_t>(1, c_size), {});
            Matrix o = matmul(right_splits[1], V_F);
            right_splits[1] = o;
          }
        }

        // The diagonal block is split along the row and column.
        auto diagonal_splits = D(block, block).split(std::vector<int64_t>(1, c_size),
                                                     std::vector<int64_t>(1, c_size));
        Matrix& Dcc = diagonal_splits[0];
        Matrix& Dco = diagonal_splits[1];
        Matrix& Doc = diagonal_splits[2];
        lu(Dcc);
        solve_triangular(Dcc, Dco, Hatrix::Left, Hatrix::Lower, true);
        solve_triangular(Dcc, Doc, Hatrix::Right, Hatrix::Upper, false);

        // Perform lower TRSM between diagonal and the A.UF block on the right of the diagonal.
        for (int icol = block+1; icol < nblocks; ++icol) {
          if (is_admissible(block, icol)) { continue; }
          auto right_splits = D(block, icol).split(std::vector<int64_t>(1, c_size),
                                                   {});
          // TRSM between cc block on diagonal and c block on the right
          solve_triangular(Dcc, right_splits[0], Hatrix::Left, Hatrix::Lower, true);
          // Use the lower part of the trapezoid
          matmul(Doc, right_splits[0], right_splits[1], false, false, -1.0, 1.0);
        }

        // Perform TRSM between A.VF blocks on the bottom of this diagonal block and the cc of the diagonal.
        for (int irow = block+1; irow < nblocks; ++irow) {
          if (is_admissible(irow, block)) { continue; }
          auto bottom_splits = D(irow, block).split({}, std::vector<int64_t>(1, c_size));
          // TRSM between cc block on diagonal and c block here.
          solve_triangular(Dcc, bottom_splits[0], Hatrix::Right, Hatrix::Upper, false);
          // GEMM between c block here and co block from diagonal.
          matmul(bottom_splits[0], Dco, bottom_splits[1], false, false, -1.0, 1.0);
        }

        // Perform TRSM between A.UF blocks on the blocks behind the diagonal
        for (int icol = 0; icol < block; ++icol) {
          if (is_admissible(block, icol)) { continue; }
          auto left_splits = D(block, icol).split(std::vector<int64_t>(1, c_size),
                                                  std::vector<int64_t>(1, c_size));
          // TRSM between cc block on diagonal and co block here.
          solve_triangular(Dcc, left_splits[1], Hatrix::Left, Hatrix::Lower, true);
        }


        // Perform upper TRSM between A.VF blocks (oc part) above this diagonal blocks and cc of diagonal block.
        for (int irow = 0; irow < block; ++irow) {
          if (is_admissible(irow, block)) { continue; }
          auto top_splits = D(irow, block).split(std::vector<int64_t>(1, c_size),
                                                 std::vector<int64_t>(1, c_size));
          // TRSM between cc block on the diagonal and oc block above this diagonal block.
          solve_triangular(Dcc, top_splits[2], Hatrix::Right, Hatrix::Upper, false);
        }

        // Schur's compliment between the co and oc blocks on the left and upper parts of
        // the diagonal, respectively that have been newly factorized and then reduced from
        // the oo parts of the blocks on the upper left of the matrix.
        for (int irow = 0; irow <= block; ++irow) {
          for (int icol = 0; icol <= block; ++icol) {
            if (is_admissible(irow, block) || is_admissible(block, icol) ||
                is_admissible(irow, icol)) { continue; }
            auto top_splits = D(irow, block).split(std::vector<int64_t>(1, c_size),
                                                   std::vector<int64_t>(1, c_size));
            auto left_splits = D(block, icol).split(std::vector<int64_t>(1, block_size - rank),
                                                    std::vector<int64_t>(1, block_size - rank));
            auto reduce_splits = D(irow, icol).split(std::vector<int64_t>(1, block_size - rank),
                                                     std::vector<int64_t>(1, block_size - rank));

            std::cout << "REDUCE: row -> " << irow << " col -> " << icol << " block -> " << block << std::endl;
            matmul(top_splits[2], left_splits[1], reduce_splits[3], false, false, -1.0, 1.0);
          }
        }

        // Schur's compliment leading to fill-in between oc blocks on the upper part
        // of the matrix and the (b-r) * r upper slice of the row that is being TRSM'd.
        // Results in a fill-in of size r * (b-r) at the bottom slice of the block into
        // which the fill-in takes place.
        for (int irow = 0; irow < block; ++irow) {
          for (int icol = block+1; icol < nblocks; ++icol) {
            if (is_admissible(block, icol) || is_admissible(irow, block)) { continue; }
            auto right_splits = D(block, icol).split(std::vector<int64_t>(1, c_size),
                                                     {});
            auto top_splits = D(irow, block).split(std::vector<int64_t>(1, block_size - rank),
                                                   std::vector<int64_t>(1, block_size - rank));
            Matrix fill_in(block_size, block_size);
            auto fill_in_splits = fill_in.split(std::vector<int64_t>(1, block_size - rank), {});
            Matrix t = matmul(top_splits[2], right_splits[0]);
            fill_in_splits[1] = t;
            // F.insert(irow, icol, std::move(fill_in));
          }
        }

        // Schur's compliment leading to fill-in between co blocks on lower part of the matrix
        // and the r * (b-r) left slice of the column that is being TRSM'd.
        for (int irow = block+1; irow < nblocks; ++irow) {
          for (int icol = 0; icol < block; ++icol) {
            if (is_admissible(block, icol) || is_admissible(irow, block)) { continue; }

            auto left_splits = D(block, icol).split(std::vector<int64_t>(1, block_size - rank),
                                                    std::vector<int64_t>(1, block_size - rank));
            auto bottom_splits = D(irow, block).split({},
                                                      std::vector<int64_t>(1, block_size - rank));
            Matrix fill_in(block_size, block_size);
            auto fill_in_splits = fill_in.split({}, std::vector<int64_t>(1, block_size - rank));
            Matrix t = matmul(bottom_splits[0], left_splits[1]);
            fill_in_splits[1] = t;
            // F.insert(irow, icol, std::move(fill_in));
          }
        }

        // Compute the schur's compliment between the reduced part of the A.UF block on the
        // upper right and reduced A.VF part of the bottom left.
        for (int irow = block+1; irow < nblocks; ++irow) {
          for (int icol = block+1; icol < nblocks; ++icol) {
            if (is_admissible(irow, block) || is_admissible(block, icol) ||
                is_admissible(irow, icol)) { continue; }
            auto right_block = D(block, icol).split(std::vector<int64_t>(1, c_size), {});
            auto bottom_block = D(irow, block).split({}, std::vector<int64_t>(1, c_size));
            matmul(bottom_block[0], right_block[0], D(irow, icol), false, false -1.0, 1.0);
          }
        }
      } // for (int block = 0; block < nblocks; ++block)

      for (int irow = 0; irow < nblocks; ++irow) {
        for (int icol = 0; icol < nblocks; ++icol) {
          if (is_admissible(irow, icol)) {
            Hatrix::Matrix dense = Hatrix::generate_laplacend_matrix(randpts,
                                                                     block_size,
                                                                     block_size,
                                                                     irow*block_size,
                                                                     icol*block_size,
                                                                     PV);
            // S.erase(irow, icol);
            // S.insert(irow, icol,
            //          Hatrix::matmul(Hatrix::matmul(U(irow), dense, true), V(icol)));
          }
        }
      }

      // Merge unfactorized portions.
      Matrix last(nblocks * rank, nblocks * rank);
      auto last_splits = last.split(nblocks, nblocks);

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          if (is_admissible(i, j)) {
            last_splits[i * nblocks + j] = S(i, j);
          }
          else {
            auto D_splits = D(i, j).split(std::vector<int64_t>(1, c_size),
                                          std::vector<int64_t>(1, c_size));
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
      for (int block = 0; block < nblocks; ++block) {
        Matrix U_F = make_complement(U(block));
        Matrix prod = matmul(U_F, x_split[block], true);
        for (int64_t i = 0; i < block_size; ++i) {
          x(block * block_size + i, 0) = prod(i, 0);
        }

        auto block_splits = D(block, block).split(std::vector<int64_t>(1, c_size),
                                                  std::vector<int64_t>(1, c_size));
        Matrix x_block(x_split[block]);
        auto x_block_splits = x_block.split(std::vector<int64_t>(1, c_size), {});
        solve_triangular(block_splits[0], x_block_splits[0], Hatrix::Left, Hatrix::Lower, true);
        matmul(block_splits[2], x_block_splits[0], x_block_splits[1], false, false, -1.0, 1.0);
        for (int64_t i = 0; i < block_size; ++i) {
          x(block * block_size + i, 0) = x_block(i, 0);
        }

        //      Forward with the big c blocks on the lower part.
        for (int irow = block+1; irow < nblocks; ++irow) {
          if (is_admissible(irow, block)) { continue; }
          auto lower_splits = D(irow, block).split({}, std::vector<int64_t>(1, c_size));

          Matrix x_block(x_split[block]);
          auto x_block_splits = x_block.split(std::vector<int64_t>(1, c_size), {});

          Matrix x_irow(x_split[irow]);
          matmul(lower_splits[0], x_block_splits[0], x_irow, false, false, -1.0, 1.0);
          for (int64_t i = 0; i < block_size; ++i) {
            x(irow * block_size + i, 0) = x_irow(i, 0);
          }
        }

        // Forward with the oc parts of the block that are actually in the upper part of the matrix.
        for (int irow = 0; irow < block; ++irow) {
          if (is_admissible(irow, block)) { continue; }
          auto top_splits = D(irow, block).split(std::vector<int64_t>(1, c_size),
                                               std::vector<int64_t>(1, c_size));
          Matrix x_irow(x_split[irow]), x_block(x_split[block]);
          auto x_irow_splits = x_irow.split(std::vector<int64_t>(1, c_size), {});
          auto x_block_splits = x_block.split(std::vector<int64_t>(1, c_size), {});

          matmul(top_splits[2], x_block_splits[0], x_irow_splits[1], false, false, -1.0, 1.0);

          for (int64_t i = 0; i < block_size; ++i) {
            x(irow * block_size + i, 0) = x_irow(i, 0);
          }
        }
      }

      permute_forward(x, block_size);
      auto permute_splits = x.split(std::vector<int64_t>(1, c_size * nblocks), {});

      solve_triangular(last, permute_splits[1], Hatrix::Left, Hatrix::Lower, true);
      solve_triangular(last, permute_splits[1], Hatrix::Left, Hatrix::Upper, false);
      permute_back(x, block_size);

      // backward substition using cc blocks
      for (int block = nblocks-1; block >= 0; --block) {
        auto block_splits = D(block, block).split(std::vector<int64_t>(1, c_size),
                                                  std::vector<int64_t>(1, c_size));
        // Apply co block.
        for (int left_col = block-1; left_col >= 0; --left_col) {
          if (!is_admissible(block, left_col)) {
            auto left_splits = D(block, left_col).split(std::vector<int64_t>(1, c_size),
                                                        std::vector<int64_t>(1, c_size));
            Matrix x_block(x_split[block]), x_left_col(x_split[left_col]);
            auto x_block_splits = x_block.split(std::vector<int64_t>(1, block_size - rank), {});
            auto x_left_col_splits = x_left_col.split(std::vector<int64_t>(1, block_size - rank), {});

            matmul(left_splits[1], x_left_col_splits[1], x_block_splits[0], false, false, -1.0, 1.0);
            for (int64_t i = 0; i < block_size; ++i) {
              x(block * block_size + i, 0) = x_block(i, 0);
            }
          }
        }

        // Apply c block present on the right of this diagonal block.
        for (int right_col = nblocks-1; right_col > block; --right_col) {
          if (!is_admissible(block, right_col)) {
            auto right_splits = D(block, right_col).split(std::vector<int64_t>(1, block_size - rank), {});

            Matrix x_block(x_split[block]);
            auto x_block_splits = x_block.split(std::vector<int64_t>(1, block_size - rank), {});

            matmul(right_splits[0], x_split[right_col], x_block_splits[0], false, false, -1.0, 1.0);
            for (int64_t i = 0; i < block_size; ++i) {
              x(block * block_size + i, 0) = x_block(i, 0);
            }
          }
        }

        Matrix x_block(x_split[block]);
        auto x_block_splits = x_block.split(std::vector<int64_t>(1, c_size), {});
        matmul(block_splits[1], x_block_splits[1], x_block_splits[0], false, false, -1.0, 1.0);
        solve_triangular(block_splits[0], x_block_splits[0], Hatrix::Left, Hatrix::Upper, false);
        for (int64_t i = 0; i < block_size; ++i) {
          x(block * block_size + i, 0) = x_block(i, 0);
        }

        auto V_F = make_complement(V(block));
        Matrix prod = matmul(V_F, x_split[block]);
        for (int i = 0; i < block_size; ++i) {
          x(block * block_size + i, 0) = prod(i, 0);
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
                                                            i * block_size, j * block_size, PV);
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
  A.print_structure();
  double construct_error = A.construction_relative_error(randpts);
  auto last = A.factorize(randpts);
  Hatrix::Matrix x = A.solve(b, last);

  // std::cout << "x:\n";
  // x.print();

  // Verification with dense solver.
  Hatrix::Matrix Adense = Hatrix::generate_laplacend_matrix(randpts, N, N, 0, 0, PV);
  Hatrix::Matrix x_solve = lu_solve(Adense, b);
  // Hatrix::Matrix x_solve(b);
  // Hatrix::lu(Adense);
  // Hatrix::solve_triangular(Adense, x_solve, Hatrix::Left, Hatrix::Lower, true);
  // Hatrix::solve_triangular(Adense, x_solve, Hatrix::Left, Hatrix::Upper, false);



  std::cout << "x solve:\n";
  auto res = x - x_solve;
  std::cout << "---- 0 ----\n";
  for (int i = 0; i < N/4; ++i) {
    std::cout << res(i, 0) << std::endl;
  }
  std::cout << "---- 1 ----\n";
  for (int i = N/4; i < N/2; ++i) {
    std::cout << res(i, 0) << std::endl;
  }
  std::cout << "---- 2 ----\n";
  for (int i = N/2; i < 3 * N / 4; ++i) {
    std::cout << res(i, 0) << std::endl;
  }
  std::cout << "---- 3 ----\n";
  for (int i = 3*N/4; i < N; ++i) {
    std::cout << res(i, 0) << std::endl;
  }

  double solve_error = Hatrix::norm(x - x_solve) / Hatrix::norm(x_solve);

  Hatrix::Context::finalize();

  std::cout << "N: " << N << " rank: " << rank << " nblocks: " << nblocks << " admis: " <<  admis
            << " construct error: " << construct_error << " solve error: " << solve_error << "\n";
}
