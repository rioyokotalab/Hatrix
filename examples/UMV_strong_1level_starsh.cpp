#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cassert>
#include <functional>

#include <starsh.h>
#include <starsh-randtlr.h>
#include <starsh-electrodynamics.h>
#include <starsh-spatial.h>

#include "Hatrix/Hatrix.h"

// BLR2 compression scheme using randomization and multiplying each row/col block
// into another to generate the shared bases.
// UMV factorization using Miamiao Ma's method as shown in
// Accuracy Directly Controlled Fast Direct Solution of General H^2-matrices and Its
// Application to Solving Electrodynamics Volume Integral Equations

double PV = 1;
using randvec_t = std::vector<std::vector<double> >;
randvec_t randpts;

int ndim;
STARSH_kernel *s_kernel;
void *starsh_data;
STARSH_int * starsh_index;

double rel_error(const Hatrix::Matrix& A, const Hatrix::Matrix& B) {
  return Hatrix::norm(A - B) / Hatrix::norm(B);
}

// TODO: Make a better copy constructor for Matrix and replace this macro with a function.
#define SPLIT_DENSE(dense, row_split, col_split)        \
  dense.split(std::vector<int64_t>(1, row_split),       \
              std::vector<int64_t>(1, col_split));

namespace Hatrix {


  Matrix
  generate_starsh_kernel(int64_t rows, int64_t cols,
                         int64_t row_start, int64_t col_start) {
    Matrix out(rows, cols);

    s_kernel(
             rows, cols,
             starsh_index + row_start, starsh_index + col_start,
             starsh_data, starsh_data,
             &out, out.stride);

    return out;
  }

  Matrix
  generate_laplacend_kernel(int64_t rows, int64_t cols,
                            int64_t row_start, int64_t col_start) {
    return generate_laplacend_matrix(randpts, rows, cols, row_start, col_start, PV);
  }

  std::function<Matrix(int64_t, int64_t, int64_t, int64_t)> kernel_function;

  class BLR2 {
  public:
    RowLevelMap U;
    ColLevelMap V;
    RowColLevelMap<bool> is_admissible;
    RowColLevelMap<Matrix> D, S;
    int64_t N, nblocks, rank, admis;
    const int64_t level = 1;
  private:
    void permute_forward(Matrix& x, int64_t block_size) {
      int64_t c_size_offset = 0, rank_offset = 0;
      Matrix temp(x);
      int64_t offset = 0;
      for (int i = 0; i < nblocks; ++i) {
        offset += block_size - U(i, level).cols;
      }

      for (int block = 0; block < nblocks; ++block) {
        int64_t row_rank = U(block, level).cols;
        int64_t c_size = block_size - U(block, level).cols;
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
      for (int i = 0; i < nblocks; ++i) { offset += (block_size - V(i, level).cols); }
      Matrix temp(x);

      int64_t c_size_offset = 0, rank_offset = 0;
      for (int block = 0; block < nblocks; ++block) {
        int64_t col_rank = V(block, level).cols;
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
    BLR2(int64_t N, int64_t nblocks, int64_t rank, int64_t admis) :
      N(N), nblocks(nblocks), rank(rank), admis(admis) {
      int block_size = N / nblocks;


      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          is_admissible.insert(i, j, level, std::abs(i - j) > admis);
        }
      }

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          if (!is_admissible.exists(i, j, level)) {
            is_admissible.insert(i, j, level, true);
          }
          if (!is_admissible(i, j, level)) {
            D.insert(i, j, level,
                     Hatrix::kernel_function(block_size, block_size,
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
          if (is_admissible(i, j, level)) {
            admissible_found = true;
            Hatrix::Matrix dense = Hatrix::kernel_function(block_size,
                                                                     block_size,
                                                                     i*block_size,
                                                                     j*block_size);
            Hatrix::matmul(dense, Y[j], AY);
          }
        }
        std::tie(Utemp, Stemp, Vtemp, error) = Hatrix::truncated_svd(AY, rank);

        if (admissible_found) {
          U.insert(i, level, std::move(Utemp));
        }
      }

      for (int64_t j = 0; j < nblocks; ++j) {
        Hatrix::Matrix YtA(rank + oversampling, block_size);
        bool admissible_found = false;
        for (int64_t i = 0; i < nblocks; ++i) {
          if (is_admissible(i, j, level)) {
            admissible_found = true;
            Hatrix::Matrix dense = Hatrix::kernel_function(block_size,
                                                                     block_size,
                                                                     i*block_size,
                                                                     j*block_size);
            Hatrix::matmul(Y[i], dense, YtA, true);
          }
        }
        std::tie(Utemp, Stemp, Vtemp, error) = Hatrix::truncated_svd(YtA, rank);

        if (admissible_found) {
          V.insert(j, level, std::move(transpose(Vtemp)));
        }
      }

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          if (is_admissible(i, j, level)) {
            Hatrix::Matrix dense = Hatrix::kernel_function(block_size, block_size,
                                                                     i*block_size,
                                                                     j*block_size);
            S.insert(i, j, level,
                     Hatrix::matmul(Hatrix::matmul(U(i, level), dense, true), V(j, level)));
          }
        }
      }
    }

    void factorize_level(int level, int nblocks) {
      RowColMap<Matrix> F;      // fill-in blocks.

      for (int block = 0; block < nblocks; ++block) {
        int64_t block_size = U(block, level).rows;
        if (block > 0 && admis != 0) {
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
                if (is_admissible.exists(block, j, level) && is_admissible(block, j, level)) {
                  row_concat = concat(row_concat, matmul(U(block, level),
                                                         S(block, j, level)), 1);
                  if (F.exists(block, j)) {
                    Matrix Fp = matmul(F(block, j), V(j, level), false, true);
                    row_concat = concat(row_concat, Fp, 1);
                  }
                }
              }

              Matrix UN1, _SN1, _VN1T; double error;
              std::tie(UN1, _SN1, _VN1T, error) = truncated_svd(row_concat, rank);

              Matrix r_block = matmul(UN1, U(block, level), true, false);

              for (int j = 0; j < nblocks; ++j) {
                if (is_admissible.exists(block, j, level) && is_admissible(block, j, level)) {
                  Matrix Sbar_block_j = matmul(r_block, S(block, j, level));

                  Matrix SpF(rank, rank);
                  if (F.exists(block, j)) {
                    Matrix Fp = matmul(F(block, j), V(j, level), false, true);
                    SpF = matmul(matmul(UN1, Fp, true, false), V(j, level));
                    Sbar_block_j = Sbar_block_j + SpF;
                    F.erase(block, j);
                  }

                  S.erase(block, j, level);
                  S.insert(block, j, level, std::move(Sbar_block_j));
                }
              }

              U.erase(block, level);
              U.insert(block, level, std::move(UN1));
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
                if (is_admissible.exists(i, block, level) && is_admissible(i, block, level)) {
                  col_concat = concat(col_concat, matmul(S(i, block, level),
                                                         transpose(V(block, level))), 0);
                  if (F.exists(i, block)) {
                    Matrix Fp = matmul(U(i, level), F(i, block));
                    col_concat = concat(col_concat, Fp, 0);
                  }
                }
              }

              Matrix _UN2, _SN2, VN2T; double error;
              std::tie(_UN2, _SN2, VN2T, error) = truncated_svd(col_concat, rank);

              Matrix t_block = matmul(V(block, level), VN2T, true, true);

              for (int i = 0; i < nblocks; ++i) {
                if (is_admissible.exists(i, block, level) && is_admissible(i, block, level)) {
                  Matrix Sbar_i_block = matmul(S(i, block,level), t_block);
                  if (F.exists(i, block)) {
                    Matrix Fp = matmul(U(i,level), F(i, block));
                    Matrix SpF = matmul(matmul(U(i,level), Fp, true, false), VN2T, false, true);
                    Sbar_i_block = Sbar_i_block + SpF;

                    F.erase(i, block);
                  }

                  S.erase(i, block, level);
                  S.insert(i, block, level, std::move(Sbar_i_block));
                }
              }

              V.erase(block, level);
              V.insert(block, level, transpose(VN2T));
            }
          }
        }

        Matrix U_F = make_complement(U(block, level));
        Matrix V_F = make_complement(V(block, level));

        for (int j = 0; j < nblocks; ++j) {
          if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) {
            D(block, j, level) = matmul(U_F, D(block, j, level), true);
          }
        }

        for (int i = 0; i < nblocks; ++i) {
          if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) {
            D(i, block, level) = matmul(D(i, block, level), V_F);
          }
        }

        int64_t row_rank = U(block, level).cols, col_rank = V(block, level).cols;
        int64_t row_split = block_size - row_rank, col_split = block_size - col_rank;

        // The diagonal block is split along the row and column.
        auto diagonal_splits = SPLIT_DENSE(D(block, block, level), row_split, col_split);
        Matrix& Dcc = diagonal_splits[0];
        lu(Dcc);

        // TRSM with CC blocks on the row
        for (int j = block + 1; j < nblocks; ++j) {
          if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) {
            int64_t col_split = block_size - V(j, level).cols;
            auto D_splits = SPLIT_DENSE(D(block, j, level), row_split, col_split);
            solve_triangular(Dcc, D_splits[0], Hatrix::Left, Hatrix::Lower, true);
          }
        }

        // TRSM with co blocks on this row
        for (int j = 0; j < nblocks; ++j) {
          if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) {
            int64_t col_split = block_size - V(j, level).cols;
            auto D_splits = SPLIT_DENSE(D(block, j, level), row_split, col_split);
            solve_triangular(Dcc, D_splits[1], Hatrix::Left, Hatrix::Lower, true);
          }
        }

        // TRSM with cc blocks on the column
        for (int i = block + 1; i < nblocks; ++i) {
          if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) {
            int64_t row_split = block_size - U(i, level).cols;
            auto D_splits = SPLIT_DENSE(D(i, block, level), row_split, col_split);
            solve_triangular(Dcc, D_splits[0], Hatrix::Right, Hatrix::Upper, false);
          }
        }

        // TRSM with oc blocks on the column
        for (int i = 0; i < nblocks; ++i) {
          if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) {
            auto D_splits = SPLIT_DENSE(D(i, block, level), block_size - U(i, level).cols, col_split);
            solve_triangular(Dcc, D_splits[2], Hatrix::Right, Hatrix::Upper, false);
          }
        }

        // Schur's compliment between cc blocks
        for (int i = block+1; i < nblocks; ++i) {
          for (int j = block+1; j < nblocks; ++j) {
            if ((is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) &&
                (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) &&
                (is_admissible.exists(i, j, level) && !is_admissible(i, j, level))) {
              auto lower_splits = SPLIT_DENSE(D(i, block, level),
                                              block_size - U(i, level).cols,
                                              col_split);
              auto right_splits = SPLIT_DENSE(D(block, j, level),
                                              row_split,
                                              block_size - V(j, level).cols);
              auto reduce_splits = SPLIT_DENSE(D(i, j, level),
                                               row_split,
                                               col_split);

              matmul(lower_splits[0], right_splits[0], reduce_splits[0], false, false, -1.0, 1.0);
            }
          }
        }

        // Schur's compliment between oc and co blocks
        for (int i = 0; i < nblocks; ++i) {
          for (int j = 0; j < nblocks; ++j) {
            if ((is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) &&
                (is_admissible.exists(i, block, level) && !is_admissible(i, block, level))) {
              auto lower_splits = SPLIT_DENSE(D(i, block, level), block_size -
                                              U(i, level).cols, col_split);
              auto right_splits = SPLIT_DENSE(D(block, j, level), row_split,
                                              block_size - V(j, level).cols);

              if (!is_admissible(i, j, level)) {
                auto reduce_splits = SPLIT_DENSE(D(i, j, level),
                                                 block_size - U(i, level).cols,
                                                 block_size - V(j, level).cols);
                matmul(lower_splits[2], right_splits[1], reduce_splits[3], false, false, -1.0, 1.0);
              }
            }
          }
        }


        for (int i = block+1; i < nblocks; ++i) {
          for (int j = 0; j < nblocks; ++j) {
            if ((is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) &&
                (is_admissible.exists(i, block, level) && !is_admissible(i, block, level))) {
              auto lower_splits = SPLIT_DENSE(D(i, block, level), block_size -
                                              U(i, level).cols, col_split);
              auto right_splits = SPLIT_DENSE(D(block, j, level), row_split, block_size -
                                              V(j, level).cols);
              // Schur's compliement between co and cc blocks where product exists as dense.
              if (is_admissible.exists(i, j, level) && !is_admissible(i, j, level)) {
                auto reduce_splits = SPLIT_DENSE(D(i, j, level),
                                                 block_size - U(i, level).cols,
                                                 block_size - V(j, level).cols);
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
            if ((is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) &&
                (is_admissible.exists(i, block, level) && !is_admissible(i, block, level))) {
              auto lower_splits = SPLIT_DENSE(D(i, block, level), block_size -
                                              U(i, level).cols, col_split);
              auto right_splits = SPLIT_DENSE(D(block, j, level),
                                              row_split, block_size - V(j, level).cols);
              // Schur's compliement between oc and cc blocks where product exists as dense.
              if (is_admissible.exists(i, j, level) && !is_admissible(i, j, level)) {
                auto reduce_splits = SPLIT_DENSE(D(i, j, level),
                                                 block_size - U(i, level).cols,
                                                 block_size - V(j, level).cols);
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
    }

    // Perform factorization assuming the permuted form of the BLR2 matrix.
    Matrix factorize() {
      int block_size = N / nblocks;

      factorize_level(level, nblocks);

      // Merge unfactorized portions.
      std::vector<int64_t> row_splits, col_splits;
      int64_t nrows = 0, ncols = 0;
      for (int i = 0; i < nblocks; ++i) {
        int64_t row_rank = U(i, level).cols, col_rank = V(i, level).cols;
        row_splits.push_back(nrows + row_rank);
        col_splits.push_back(ncols + col_rank);
        nrows += row_rank;
        ncols += col_rank;
      }

      Matrix last(nrows, ncols);
      auto last_splits = last.split(row_splits, col_splits);

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          if (is_admissible(i, j, level)) {
            last_splits[i * nblocks + j] = S(i, j, level);
          }
          else {
            auto D_splits = SPLIT_DENSE(D(i, j, level),
                                        block_size - U(i, level).cols,
                                        block_size - V(j, level).cols);
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
        Matrix U_F = make_complement(U(block, level));
        Matrix prod = matmul(U_F, x_split[block], true);
        for (int64_t i = 0; i < block_size; ++i) {
          x(block * block_size + i, 0) = prod(i, 0);
        }
      }

      // forward substitution with cc blocks
      for (int block = 0; block < nblocks; ++block) {
        int64_t row_split = block_size - U(block, level).cols,
          col_split = block_size - V(block, level).cols;

        auto block_splits = SPLIT_DENSE(D(block, block, level), row_split, col_split);
        Matrix x_block(x_split[block]);
        auto x_block_splits = x_block.split(std::vector<int64_t>(1, row_split), {});
        solve_triangular(block_splits[0], x_block_splits[0], Hatrix::Left, Hatrix::Lower, true);
        matmul(block_splits[2], x_block_splits[0], x_block_splits[1], false, false, -1.0, 1.0);
        for (int64_t i = 0; i < block_size; ++i) {
          x(block * block_size + i, 0) = x_block(i, 0);
        }

        //      Forward with the big c blocks on the lower part.
        for (int irow = block+1; irow < nblocks; ++irow) {
          if (is_admissible(irow, block, level)) { continue; }
          int64_t row_split = block_size - U(irow, level).cols;
          int64_t col_split = block_size - V(block, level).cols;
          auto lower_splits = D(irow, block, level).split({}, std::vector<int64_t>(1, row_split));

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
          if (is_admissible(irow, block, level)) { continue; }
          int64_t row_split = block_size - U(irow, level).cols;
          int64_t col_split = block_size - V(block, level).cols;
          auto top_splits = SPLIT_DENSE(D(irow, block, level), row_split, col_split);
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
        offset += block_size - U(i, level).cols;
      }
      auto permute_splits = x.split(std::vector<int64_t>(1, offset), {});
      solve_triangular(last, permute_splits[1], Hatrix::Left, Hatrix::Lower, true);
      solve_triangular(last, permute_splits[1], Hatrix::Left, Hatrix::Upper, false);
      permute_back(x, block_size);

      // backward substition using cc blocks
      for (int block = nblocks-1; block >= 0; --block) {
        int64_t row_split = block_size - U(block, level).cols,
          col_split = block_size - V(block, level).cols;
        auto block_splits = SPLIT_DENSE(D(block, block, level), row_split, col_split);
        // Apply co block.
        for (int left_col = block-1; left_col >= 0; --left_col) {
          if (!is_admissible(block, left_col, level)) {
            int64_t row_split = block_size - U(block, level).cols;
            int64_t col_split = block_size - V(left_col, level).cols;
            auto left_splits = SPLIT_DENSE(D(block, left_col, level), row_split, col_split);

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
          if (!is_admissible(block, right_col, level)) {
            int64_t row_split = block_size - U(block, level).cols;
            auto right_splits = D(block, right_col, level).
              split(std::vector<int64_t>(1, row_split), {});

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
        auto V_F = make_complement(V(block, level));
        Matrix prod = matmul(V_F, x_split[block]);
        for (int i = 0; i < block_size; ++i) {
          x(block * block_size + i, 0) = prod(i, 0);
        }
      }

      return x;
    }


    double construction_relative_error() {
      double error = 0, dense_norm = 0;
      int block_size = N / nblocks;

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          Matrix actual = Hatrix::kernel_function(block_size, block_size,
                                                            i * block_size, j * block_size);
          dense_norm += pow(Hatrix::norm(actual), 2);

          if (!is_admissible(i, j, level)) {
            error += pow(Hatrix::norm(D(i, j, level) - actual), 2);
          }
          else {
            Matrix& Ubig = U(i, level);
            Matrix& Vbig = V(j, level);
            Matrix expected = matmul(matmul(Ubig, S(i, j, level)), Vbig, false, true);

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
          std::cout << " | " << is_admissible(i, j, level);
        }
        std::cout << " | \n";
      }
    }
  };
}

// kernel func:
// 0 - Default 3D laplace. straight line geometry.
// 1 - starsh sin_kernel_2d. Grid geometry.
// 2 - sqrexp_kernel_2d.
// 3 - sqrexp_kernel_3d
// 4 - exp_kernel_3d
int main(int argc, char** argv) {
  int64_t N = atoi(argv[1]);
  int64_t nblocks = atoi(argv[2]);
  int64_t rank = atoi(argv[3]);
  int64_t admis = atoi(argv[4]);
  int64_t kernel_func = atoi(argv[5]);

  double beta = 0.1;
  double nu = 0.5;     //in matern, nu=0.5 exp (half smooth), nu=inf sqexp (inifinetly smooth)
  double noise = 1.e-1;
  double sigma = 1.0;

  Hatrix::Context::init();

  enum STARSH_PARTICLES_PLACEMENT place = STARSH_PARTICLES_UNIFORM;
  Hatrix::kernel_function = Hatrix::generate_starsh_kernel;
  int ndim;
  switch(kernel_func) {
  case 0:
    randpts.push_back(Hatrix::equally_spaced_vector(N, 0.0, 1.0 * N)); // 1D
    randpts.push_back(Hatrix::equally_spaced_vector(N, 0.0, 1.0 * N)); // 2D
    randpts.push_back(Hatrix::equally_spaced_vector(N, 0.0, 1.0 * N)); // 3D
    PV = 1e-3 * (1 / pow(10, 1));
    Hatrix::kernel_function = Hatrix::generate_laplacend_kernel;
    break;
  case 1: {
    double wave_k = 50;           // default value from hicma_parsec.c
    double add_diag = 0.0;        // default value from hicma_parsec.c
    ndim = 2;
    s_kernel = starsh_eddata_block_sin_kernel_2d;
    starsh_eddata_generate((STARSH_eddata **)&starsh_data, N, ndim, wave_k,
                           add_diag, place);
    break;
  }
  case 2:
    ndim = 2;
    s_kernel = starsh_ssdata_block_sqrexp_kernel_2d;
    // This function will generate a 2D spatial geometry. Grid.
    starsh_ssdata_generate((STARSH_ssdata**)&starsh_data, N, ndim, beta,
                           nu, noise, place, sigma);
    break;
  case 3:
    ndim = 3;
    // This function will generate a 3D spatial geometry. Grid.
    s_kernel = starsh_ssdata_block_sqrexp_kernel_3d;
    starsh_ssdata_generate((STARSH_ssdata **)&starsh_data, N, ndim,
                           beta, nu, noise, place, sigma);
    break;
  case 4:
    ndim = 3;
    s_kernel = starsh_ssdata_block_exp_kernel_3d;
    starsh_ssdata_generate((STARSH_ssdata **)&starsh_data, N, ndim,
                           beta, nu, noise, place, sigma);

    break;
  };

  if (N % nblocks != 0) {
    std::cout << "N % nblocks != 0. Aborting.\n";
    abort();
  }

  Hatrix::Matrix b = Hatrix::generate_range_matrix(N, 1, 0);

  Hatrix::BLR2 A(N, nblocks, rank, admis);
  // A.print_structure();
  double construct_error = A.construction_relative_error();
  auto last = A.factorize();
  Hatrix::Matrix x = A.solve(b, last);

  // Verification with dense solver.
  Hatrix::Matrix Adense = Hatrix::generate_laplacend_kernel(N, N, 0, 0);
  Hatrix::Matrix x_solve = lu_solve(Adense, b);

  double solve_error = Hatrix::norm(x - x_solve) / Hatrix::norm(x_solve);

  Hatrix::Context::finalize();

  std::cout << "N: " << N << " rank: " << rank << " nblocks: " << nblocks << " admis: " <<  admis
            << " construct error: " << construct_error << " solve error: " << solve_error << "\n";
}
