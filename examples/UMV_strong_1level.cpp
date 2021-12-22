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

constexpr double PV = 1;
using randvec_t = std::vector<std::vector<double> >;

double rel_error(const Hatrix::Matrix& A, const Hatrix::Matrix& B) {
  return Hatrix::norm(A - B) / Hatrix::norm(B);
}

// TODO: Make a better copy constructor for Matrix and replace this macro with a function.
#define SPLIT_DENSE(dense, row_split, col_split)        \
  dense.split(std::vector<int64_t>(1, row_split),       \
              std::vector<int64_t>(1, col_split));

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
      int64_t c_size_offset = 0, rank_offset = 0;
      Matrix temp(x);
      int64_t offset = 0;
      for (int i = 0; i < nblocks; ++i) {
        offset += block_size - U(i).cols;
      }

      for (int block = 0; block < nblocks; ++block) {
        int64_t row_rank = U(block).cols;
        int64_t c_size = block_size - U(block).cols;
        // Copy the compliment part of the RHS vector.
        for (int i = 0; i < c_size; ++i) {
          temp(c_size_offset + i, 0) = x(block_size * block + i, 0);
        }
        // Copy the rank part of the RHS vector.
        for (int i = 0; i < row_rank; ++i) {
          temp(rank_offset + offset + i, 0) = x(block_size * block + c_size + i, 0);
        }

        c_size_offset += c_size;
        rank_offset += row_rank;
      }

      x = temp;
    }

    void permute_back(Matrix& x, int64_t block_size) {
      int64_t offset = 0;
      for (int i = 0; i < nblocks; ++i) { offset += (block_size - V(i).cols); }
      Matrix temp(x);

      int64_t c_size_offset = 0, rank_offset = 0;
      for (int block = 0; block < nblocks; ++block) {
        int64_t col_rank = V(block).cols;
        int64_t c_size = block_size - col_rank;

        // Copy the compliment part of the vector.
        for (int i = 0; i < c_size; ++i) {
          temp(block_size * block + i, 0) = x(c_size_offset + i, 0);
        }

        // Copy the rank part of the vector.
        for (int i = 0; i < col_rank; ++i) {
          temp(block_size * block + c_size + i, 0) = x(offset + rank_offset + i, 0);
        }

        c_size_offset += c_size;
        rank_offset += col_rank;
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
        }
      }

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
                                                                     block_size,
                                                                     block_size,
                                                                     i*block_size,
                                                                     j*block_size, PV);
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
                                                                     block_size,
                                                                     block_size,
                                                                     i*block_size,
                                                                     j*block_size, PV);
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
                                                                     i*block_size,
                                                                     j*block_size, PV);
            S.insert(i, j,
                     Hatrix::matmul(Hatrix::matmul(U(i), dense, true), V(j)));
          }
        }
      }
    }

    // Perform factorization assuming the permuted form of the BLR2 matrix.
    Matrix factorize(const randvec_t &randpts) {
      int block_size = N / nblocks;
      RowColMap<Matrix> F;      // fill-in blocks.

      for (int block = 0; block < nblocks; ++block) {
        if (block > 0) {
          {
            // Scan for fill-ins in the same row as this diagonal block.
            Matrix row_concat(block_size, 0);
            std::vector<int64_t> VN1_col_splits;
            bool found_row_fill_in = false;
            for (int j = 0; j < nblocks; ++j) {
              if (F.exists(block, j)) {
                found_row_fill_in = true;
                break;
              }
            }

            if (found_row_fill_in) {
              for (int j = 0; j < nblocks; ++j) {
                if (is_admissible(block, j)) {
                  row_concat = concat(row_concat, matmul(U(block), S(block, j)), 1);
                  if (F.exists(block, j)) {
                    Matrix Fp = matmul(F(block, j), V(j), false, true);
                    row_concat = concat(row_concat, Fp, 1);
                  }
                }
              }

              Matrix UN1, _SN1, _VN1T; double error;
              std::tie(UN1, _SN1, _VN1T, error) = truncated_svd(row_concat, rank);

              for (int j = 0; j < nblocks; ++j) {
                if (is_admissible(block, j)) {
                  Matrix r_block_j = matmul(UN1, U(block), true, false);
                  Matrix Sbar_block_j = matmul(r_block_j, S(block, j));

                  Matrix SpF(rank, rank);
                  if (F.exists(block, j)) {
                    Matrix Fp = matmul(F(block, j), V(j), false, true);

                    SpF = matmul(matmul(UN1, Fp, true, false), V(j));
                    Sbar_block_j = Sbar_block_j + SpF;
                  }

                  S.erase(block, j);
                  S.insert(block, j, std::move(Sbar_block_j));

                  if (F.exists(block, j)) {
                    F.erase(block, j);
                  }
                }
              }

              U.erase(block);
              U.insert(block, std::move(UN1));
            }
          }

          {
            // Scan for fill-ins in the same col as this diagonal block.
            Matrix col_concat(0, block_size);
            std::vector<int64_t> UN2_row_splits;
            bool found_col_fill_in = false;
            for (int i = 0; i < nblocks; ++i) {
              if (F.exists(i, block)) {
                found_col_fill_in = true;
                break;
              }
            }

            if (found_col_fill_in) {
              for (int i = 0; i < nblocks; ++i) {
                if (is_admissible(i, block)) {
                  col_concat = concat(col_concat, matmul(S(i, block), transpose(V(block))), 0);
                  if (F.exists(i, block)) {
                    Matrix Fp = matmul(U(i), F(i, block));
                    col_concat = concat(col_concat, Fp, 0);
                  }
                }
              }

              Matrix _UN2, _SN2, VN2T; double error;
              std::tie(_UN2, _SN2, VN2T, error) = truncated_svd(col_concat, rank);

              for (int i = 0; i < nblocks; ++i) {
                if (is_admissible(i, block)) {
                  Matrix t_i_block = matmul(V(block), VN2T, true, true);
                  Matrix Sbar_i_block = matmul(S(i, block), t_i_block);
                  if (F.exists(i, block)) {
                    Matrix Fp = matmul(U(i), F(i, block));
                    Matrix SpF = matmul(matmul(U(i), Fp, true, false), VN2T, false, true);
                    Sbar_i_block = Sbar_i_block + SpF;

                    F.erase(i, block);
                  }

                  // std::cout << "UPDATE S COL: " << i << ", " << block << std::endl;

                  S.erase(i, block);
                  S.insert(i, block, std::move(Sbar_i_block));
                }
              }
              V.erase(block);
              V.insert(block, transpose(VN2T));
            }
          }
        }

        for (int j = 0; j < nblocks; ++j) {
          if (!is_admissible(block, j)) {
            Matrix U_F = make_complement(U(block));
            D(block, j) = matmul(U_F, D(block, j), true);
          }
        }

        for (int i = 0; i < nblocks; ++i) {
          if (!is_admissible(i, block)) {
            Matrix V_F = make_complement(V(block));
            D(i, block) = matmul(D(i, block), V_F);
          }
        }

        int64_t row_rank = U(block).cols, col_rank = V(block).cols;
        int64_t row_split = block_size - row_rank, col_split = block_size - col_rank;

        // The diagonal block is split along the row and column.
        auto diagonal_splits = SPLIT_DENSE(D(block, block), row_split, col_split);
        Matrix& Dcc = diagonal_splits[0];
        lu(Dcc);

        // TRSM with CC blocks on the row
        for (int j = block + 1; j < nblocks; ++j) {
          if (!is_admissible(block, j)) {
            int64_t col_split = block_size - V(j).cols;
            auto D_splits = SPLIT_DENSE(D(block, j), row_split, col_split);
            solve_triangular(Dcc, D_splits[0], Hatrix::Left, Hatrix::Lower, true);
          }
        }

        // TRSM with co blocks on this row
        for (int j = 0; j < nblocks; ++j) {
          if (!is_admissible(block, j)) {
            int64_t col_split = block_size - V(j).cols;
            auto D_splits = SPLIT_DENSE(D(block, j), row_split, col_split);
            solve_triangular(Dcc, D_splits[1], Hatrix::Left, Hatrix::Lower, true);
          }
        }

        // TRSM with cc blocks on the column
        for (int i = block + 1; i < nblocks; ++i) {
          if (!is_admissible(i, block)) {
            int64_t row_split = block_size - U(i).cols;
            auto D_splits = SPLIT_DENSE(D(i, block), row_split, col_split);
            solve_triangular(Dcc, D_splits[0], Hatrix::Right, Hatrix::Upper, false);
          }
        }

        // TRSM with oc blocks on the column
        for (int i = 0; i < nblocks; ++i) {
          if (!is_admissible(i, block)) {
            auto D_splits = SPLIT_DENSE(D(i, block), block_size - U(i).cols, col_split);
            solve_triangular(Dcc, D_splits[2], Hatrix::Right, Hatrix::Upper, false);
          }
        }

        // Schur's compliment between cc blocks
        for (int i = block+1; i < nblocks; ++i) {
          for (int j = block+1; j < nblocks; ++j) {
            if (!is_admissible(block, j) && !is_admissible(i, block) && !is_admissible(i, j)) {
              auto lower_splits = SPLIT_DENSE(D(i, block),
                                              block_size - U(i).cols,
                                              col_split);
              auto right_splits = SPLIT_DENSE(D(block, j),
                                              row_split,
                                              block_size - V(j).cols);
              auto reduce_splits = SPLIT_DENSE(D(i, j),
                                               row_split,
                                               col_split);

              matmul(lower_splits[0], right_splits[0], reduce_splits[0], false, false, -1.0, 1.0);
            }
          }
        }

        // Schur's compliment between oc and co blocks
        for (int i = 0; i < nblocks; ++i) {
          for (int j = 0; j < nblocks; ++j) {
            if (!is_admissible(block, j) && !is_admissible(i, block)) {
              auto lower_splits = SPLIT_DENSE(D(i, block), block_size - U(i).cols, col_split);
              auto right_splits = SPLIT_DENSE(D(block, j), row_split, block_size - V(j).cols);

              if (!is_admissible(i, j)) {
                auto reduce_splits = SPLIT_DENSE(D(i, j),
                                                 block_size - U(i).cols,
                                                 block_size - V(j).cols);
                matmul(lower_splits[2], right_splits[1], reduce_splits[3], false, false, -1.0, 1.0);
              }
            }
          }
        }

        for (int i = block+1; i < nblocks; ++i) {
          for (int j = 0; j < nblocks; ++j) {
            if (!is_admissible(block, j) && !is_admissible(i, block)) {
              auto lower_splits = SPLIT_DENSE(D(i, block), block_size - U(i).cols, col_split);
              auto right_splits = SPLIT_DENSE(D(block, j), row_split, block_size - V(j).cols);
              // Schur's compliement between co and cc blocks where product exists as dense.
              if (!is_admissible(i, j)) {
                auto reduce_splits = SPLIT_DENSE(D(i, j),
                                                 block_size - U(i).cols,
                                                 block_size - V(j).cols);
                matmul(lower_splits[0], right_splits[1], reduce_splits[1], false, false, -1.0, 1.0);
              }
              // Schur's compliement between co and cc blocks where a new fill-in is created.
              // The product is a (co; oo)-sized matrix.
              else {
                if (!F.exists(i, j)) {
                  Matrix fill_in(block_size, rank);
                  auto fill_splits = fill_in.split(std::vector<int64_t>(1, block_size - rank), {});
                  // Update the co block within the fill-in.
                  matmul(lower_splits[0], right_splits[1], fill_splits[0], false, false, -1.0, 1.0);

                  // Update the oo block within the fill-in.
                  matmul(lower_splits[2], right_splits[1], fill_splits[1], false, false, -1.0, 1.0);

                  F.insert(i, j, std::move(fill_in));
                }
                else {
                  Matrix &fill_in = F(i, j);
                  auto fill_splits = fill_in.split(std::vector<int64_t>(1, block_size - rank), {});
                  // Update the co block within the fill-in.
                  matmul(lower_splits[0], right_splits[1], fill_splits[0], false, false, -1.0, 1.0);
                  // Update the oo block within the fill-in.
                  matmul(lower_splits[2], right_splits[1], fill_splits[1], false, false, -1.0, 1.0);
                }
              }
            }
          }
        }

        // Schur's compliment between oc and cc blocks
        for (int i = 0; i < nblocks; ++i) {
          for (int j = block+1; j < nblocks; ++j) {
            if (!is_admissible(block, j) && !is_admissible(i, block)) {
              auto lower_splits = SPLIT_DENSE(D(i, block), block_size - U(i).cols, col_split);
              auto right_splits = SPLIT_DENSE(D(block, j), row_split, block_size - V(j).cols);
              // Schur's compliement between oc and cc blocks where product exists as dense.
              if (!is_admissible(i, j)) {
                auto reduce_splits = SPLIT_DENSE(D(i, j),
                                                 block_size - U(i).cols,
                                                 block_size - V(j).cols);
                matmul(lower_splits[2], right_splits[0], reduce_splits[2],
                       false, false, -1.0, 1.0);
              }
              // Schur's compliement between co and cc blocks where a new fill-in is created.
              // The product is a (oc, oo)-sized block.
              else {
                if (!F.exists(i, j)) {
                  Matrix fill_in(rank, block_size);
                  auto fill_splits = fill_in.split({}, std::vector<int64_t>(1, block_size - rank));
                  // Update the oc block within the fill-ins.
                  matmul(lower_splits[2], right_splits[0], fill_splits[0], false, false, -1.0, 1.0);
                  // Update the oo block within the fill-ins.
                  matmul(lower_splits[2], right_splits[1], fill_splits[1], false, false, -1.0, 1.0);
                  F.insert(i, j, std::move(fill_in));
                }
                else {
                  Matrix& fill_in = F(i, j);
                  auto fill_splits = fill_in.split({}, std::vector<int64_t>(1, block_size - rank));
                  // Update the oc block within the fill-ins.
                  matmul(lower_splits[2], right_splits[0], fill_splits[0], false, false, -1.0, 1.0);
                  // Update the oo block within the fill-ins.
                  matmul(lower_splits[2], right_splits[1], fill_splits[1], false, false, -1.0, 1.0);
                }
              }
            }
          }
        }
      } // for (int block = 0; block < nblocks; ++block)

      // Merge unfactorized portions.
      std::vector<int64_t> row_splits, col_splits;
      int64_t nrows = 0, ncols = 0;
      for (int i = 0; i < nblocks; ++i) {
        int64_t row_rank = U(i).cols, col_rank = V(i).cols;
        row_splits.push_back(nrows + row_rank);
        col_splits.push_back(ncols + col_rank);
        nrows += row_rank;
        ncols += col_rank;
      }

      Matrix last(nrows, ncols);
      auto last_splits = last.split(row_splits, col_splits);

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          if (is_admissible(i, j)) {
            last_splits[i * nblocks + j] = S(i, j);
          }
          else {
            auto D_splits = SPLIT_DENSE(D(i, j),
                                        block_size - U(i).cols,
                                        block_size - V(j).cols);
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

      for (int block = 0; block < nblocks; ++block) {
        Matrix U_F = make_complement(U(block));
        Matrix prod = matmul(U_F, x_split[block], true);
        for (int64_t i = 0; i < block_size; ++i) {
          x(block * block_size + i, 0) = prod(i, 0);
        }
      }

      // forward substitution with cc blocks
      for (int block = 0; block < nblocks; ++block) {
        int64_t row_split = block_size - U(block).cols, col_split = block_size - V(block).cols;
        auto block_splits = SPLIT_DENSE(D(block, block), row_split, col_split);
        Matrix x_block(x_split[block]);
        auto x_block_splits = x_block.split(std::vector<int64_t>(1, row_split), {});
        solve_triangular(block_splits[0], x_block_splits[0], Hatrix::Left, Hatrix::Lower, true);
        matmul(block_splits[2], x_block_splits[0], x_block_splits[1], false, false, -1.0, 1.0);
        for (int64_t i = 0; i < block_size; ++i) {
          x(block * block_size + i, 0) = x_block(i, 0);
        }

        //      Forward with the big c blocks on the lower part.
        for (int irow = block+1; irow < nblocks; ++irow) {
          if (is_admissible(irow, block)) { continue; }
          int64_t row_split = block_size - U(irow).cols;
          int64_t col_split = block_size - V(block).cols;
          auto lower_splits = D(irow, block).split({}, std::vector<int64_t>(1, row_split));

          Matrix x_block(x_split[block]);
          auto x_block_splits = x_block.split(std::vector<int64_t>(1, col_split), {});

          Matrix x_irow(x_split[irow]);
          matmul(lower_splits[0], x_block_splits[0], x_irow, false, false, -1.0, 1.0);
          for (int64_t i = 0; i < block_size; ++i) {
            x(irow * block_size + i, 0) = x_irow(i, 0);
          }
        }

        // Forward with the oc parts of the block that are actually in the upper part of the matrix.
        for (int irow = 0; irow < block; ++irow) {
          if (is_admissible(irow, block)) { continue; }
          int64_t row_split = block_size - U(irow).cols;
          int64_t col_split = block_size - V(block).cols;
          auto top_splits = SPLIT_DENSE(D(irow, block), row_split, col_split);
          Matrix x_irow(x_split[irow]), x_block(x_split[block]);
          auto x_irow_splits = x_irow.split(std::vector<int64_t>(1, row_split), {});
          auto x_block_splits = x_block.split(std::vector<int64_t>(1, col_split), {});

          matmul(top_splits[2], x_block_splits[0], x_irow_splits[1], false, false, -1.0, 1.0);

          for (int64_t i = 0; i < block_size; ++i) {
            x(irow * block_size + i, 0) = x_irow(i, 0);
          }
        }
      }

      permute_forward(x, block_size);
      int64_t offset = 0;
      for (int i = 0; i < nblocks; ++i) {
        offset += block_size - U(i).cols;
      }
      auto permute_splits = x.split(std::vector<int64_t>(1, offset), {});
      solve_triangular(last, permute_splits[1], Hatrix::Left, Hatrix::Lower, true);
      solve_triangular(last, permute_splits[1], Hatrix::Left, Hatrix::Upper, false);
      permute_back(x, block_size);

      // backward substition using cc blocks
      for (int block = nblocks-1; block >= 0; --block) {
        int64_t row_split = block_size - U(block).cols, col_split = block_size - V(block).cols;
        auto block_splits = SPLIT_DENSE(D(block, block), row_split, col_split);
        // Apply co block.
        for (int left_col = block-1; left_col >= 0; --left_col) {
          if (!is_admissible(block, left_col)) {
            int64_t row_split = block_size - U(block).cols;
            int64_t col_split = block_size - V(left_col).cols;
            auto left_splits = SPLIT_DENSE(D(block, left_col), row_split, col_split);

            Matrix x_block(x_split[block]), x_left_col(x_split[left_col]);
            auto x_block_splits = x_block.split(std::vector<int64_t>(1, row_split), {});
            auto x_left_col_splits = x_left_col.split(std::vector<int64_t>(1, col_split), {});

            matmul(left_splits[1], x_left_col_splits[1], x_block_splits[0], false, false, -1.0, 1.0);
            for (int64_t i = 0; i < block_size; ++i) {
              x(block * block_size + i, 0) = x_block(i, 0);
            }
          }
        }

        // Apply c block present on the right of this diagonal block.
        for (int right_col = nblocks-1; right_col > block; --right_col) {
          if (!is_admissible(block, right_col)) {
            int64_t row_split = block_size - U(block).cols;
            auto right_splits = D(block, right_col).split(std::vector<int64_t>(1, row_split), {});

            Matrix x_block(x_split[block]);
            auto x_block_splits = x_block.split(std::vector<int64_t>(1, row_split), {});

            matmul(right_splits[0], x_split[right_col], x_block_splits[0], false, false, -1.0, 1.0);
            for (int64_t i = 0; i < block_size; ++i) {
              x(block * block_size + i, 0) = x_block(i, 0);
            }
          }
        }

        Matrix x_block(x_split[block]);
        auto x_block_splits = x_block.split(std::vector<int64_t>(1, row_split), {});
        matmul(block_splits[1], x_block_splits[1], x_block_splits[0], false, false, -1.0, 1.0);
        solve_triangular(block_splits[0], x_block_splits[0], Hatrix::Left, Hatrix::Upper, false);
        for (int64_t i = 0; i < block_size; ++i) {
          x(block * block_size + i, 0) = x_block(i, 0);
        }
      }

      for (int block = nblocks-1; block >= 0; --block) {
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
  randpts.push_back(Hatrix::equally_spaced_vector(N, 0.0, 1.0)); // 1D
  randpts.push_back(Hatrix::equally_spaced_vector(N, 0.0, 1.0)); // 2D
  randpts.push_back(Hatrix::equally_spaced_vector(N, 0.0, 1.0)); // 3D

  if (N % nblocks != 0) {
    std::cout << "N % nblocks != 0. Aborting.\n";
    abort();
  }

  Hatrix::Matrix b = Hatrix::generate_random_matrix(N, 1);

  Hatrix::BLR2 A(randpts, N, nblocks, rank, admis);
  // A.print_structure();
  double construct_error = A.construction_relative_error(randpts);
  auto last = A.factorize(randpts);
  Hatrix::Matrix x = A.solve(b, last);

  // std::cout << "x:\n";
  // x.print();

  // Verification with dense solver.
  Hatrix::Matrix Adense = Hatrix::generate_laplacend_matrix(randpts, N, N, 0, 0, PV);
  // Adense.print();
  Hatrix::Matrix x_solve = lu_solve(Adense, b);


  double solve_error = Hatrix::norm(x - x_solve) / Hatrix::norm(x_solve);

  Hatrix::Context::finalize();

  std::cout << "N: " << N << " rank: " << rank << " nblocks: " << nblocks << " admis: " <<  admis
            << " construct error: " << construct_error << " solve error: " << solve_error << "\n";
}
