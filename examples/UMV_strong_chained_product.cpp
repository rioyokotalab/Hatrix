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

// BLR2 strong admis UMV factorization with chained product verification using eq 33 in Ma2018.

using namespace Hatrix;
constexpr double PV = 1e-3;
using randvec_t = std::vector<std::vector<double> >;

double rel_error(const Hatrix::Matrix& A, const Hatrix::Matrix& B) {
  return Hatrix::norm(A - B) / Hatrix::norm(B);
}


// TODO: Make a better copy constructor for Matrix and replace this macro with a function.
#define SPLIT_DENSE(dense, row_split, col_split)        \
  dense.split(std::vector<int64_t>(1, row_split),       \
              std::vector<int64_t>(1, col_split));

Hatrix::Matrix make_complement(const Hatrix::Matrix &Q) {
  Hatrix::Matrix Q_F(Q.rows, Q.rows);
  Hatrix::Matrix Q_full, R;
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


// Copy the input matrix into a lower triangular matrix. Uses N^2 storage.
// The diagonal is always identity.
Hatrix::Matrix lower(Hatrix::Matrix A) {
  Hatrix::Matrix mat(A.rows, A.cols);

  for (int i = 0; i < A.rows; ++i) {
    mat(i, i) = 1.0;
    for (int j = 0; j < i; ++j) {
      mat(i, j) = A(i, j);
    }
  }

  return mat;
}

Hatrix::Matrix upper(Hatrix::Matrix A) {
  Hatrix::Matrix mat(A.rows, A.cols);

  for (int i = 0; i < A.rows; ++i) {
    for (int j = i; j < A.cols; ++j) {
      mat(i, j) = A(i, j);
    }
  }

  return mat;
}

namespace Hatrix {
  class BLR2 {
  public:
    RowMap U;
    ColMap V;
    RowColMap<bool> is_admissible;
    RowColMap<Matrix> D, S;
    int64_t N, nblocks, rank, admis;
    RowColMap<Matrix> Loc, Uco;      // fill-ins of the small strips on the top and bottom.

    int significant_bases(const Matrix& Sa, double eta=1e-13) {
      return 2;
    }

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

    BLR2(Hatrix::BLR2& A) :
      U(A.U), V(A.V), is_admissible(A.is_admissible), D(A.D), S(A.S),
      N(A.N), nblocks(A.nblocks), rank(A.rank), admis(A.admis) {}

    BLR2(const randvec_t& randpts, int64_t N, int64_t nblocks, int64_t rank, int64_t admis) :
      N(N), nblocks(nblocks), rank(rank), admis(admis) {
      int block_size = N / nblocks;

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          is_admissible.insert(i, j, std::abs(i - j) > admis);
        }
      }

      // is_admissible.insert(0, 1, admis == 1 ? false : true);
      // is_admissible.insert(1, 0, admis == 1 ? false : true);
      // is_admissible.insert(1, 2, admis == 1 ? false : true);
      // is_admissible.insert(2, 1, admis == 1 ? false : true);
      // is_admissible.insert(3, 2, admis == 1 ? false : true);
      // is_admissible.insert(2, 3, admis == 1 ? false : true);
      // is_admissible.insert(4, 3, admis == 1 ? false : true);
      // is_admissible.insert(3, 4, admis == 1 ? false : true);



      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          // if (!is_admissible.exists(i, j)) {
          //   is_admissible.insert(i, j, true);
          // }
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
                                                                     block_size, block_size,
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
    std::tuple<Matrix, RowColMap<Matrix>> factorize(const randvec_t &randpts) {
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

      return {last, F};
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


// Take a BLR2 matrix A as argument and generate a permuted dense matrix.
Matrix generate_full_permuted(BLR2& A) {
  Matrix M(A.N, A.N);
  int64_t block_size = A.N / A.nblocks;
  int64_t c_size = block_size - A.rank;

  int64_t c_size_offset_rows = 0, c_size_offset_cols = 0;
  std::vector<int64_t> row_offsets, col_offsets;
  for (int i = 0; i < A.nblocks; ++i) {
    int64_t row_split = block_size - A.U(i).cols;
    int64_t col_split = block_size - A.V(i).cols;

    row_offsets.push_back(c_size_offset_rows + row_split);
    col_offsets.push_back(c_size_offset_cols + col_split);

    c_size_offset_rows += row_split;
    c_size_offset_cols += col_split;
  }

  int64_t row_rank_offset = 0, col_rank_offset = 0;
  for (int i = 0; i < A.nblocks-1; ++i) {
    row_offsets.push_back(c_size_offset_rows + row_rank_offset + A.U(i).cols);
    col_offsets.push_back(c_size_offset_cols + col_rank_offset + A.V(i).cols);

    row_rank_offset += A.U(i).cols;
    col_rank_offset += A.V(i).cols;
  }

  auto M_splits = M.split(row_offsets, col_offsets);
  int64_t permuted_nblocks = A.nblocks * 2;

  for (int i = 0; i < A.nblocks; ++i) {
    for (int j = 0; j < A.nblocks; ++j) {
      int64_t row_split = block_size - A.U(i).cols;
      int64_t col_split = block_size - A.V(i).cols;

      if (!A.is_admissible(i, j)) {
        auto D_splits = SPLIT_DENSE(A.D(i, j), row_split, col_split);
        // Copy cc blocks
        M_splits[i * permuted_nblocks + j] = D_splits[0];

        // Copy oc blocks
        M_splits[(i + A.nblocks) * permuted_nblocks + j] = D_splits[2];

        // Copy co blocks
        M_splits[i * permuted_nblocks + (j + A.nblocks)] = D_splits[1];

        // Copy oo blocks
        M_splits[(i + A.nblocks) * permuted_nblocks + (j + A.nblocks)] = D_splits[3];
      }
      else {
        M_splits[(i + A.nblocks) * permuted_nblocks + (j + A.nblocks)] = A.S(i, j);
      }
    }
  }

  return M;
}

std::tuple<std::vector<int64_t>, std::vector<int64_t> >
generate_offsets(BLR2& A) {
  std::vector<int64_t> row_offsets, col_offsets;
  int64_t c_size_offset_rows = 0, c_size_offset_cols = 0;
  int64_t block_size = A.N / A.nblocks;
  for (int i = 0; i < A.nblocks; ++i) {
    int64_t row_split = block_size - A.U(i).cols;
    int64_t col_split = block_size - A.V(i).cols;

    row_offsets.push_back(c_size_offset_rows + row_split);
    col_offsets.push_back(c_size_offset_cols + col_split);

    c_size_offset_rows += row_split;
    c_size_offset_cols += col_split;
  }

  int64_t row_rank_offset = 0, col_rank_offset = 0;
  for (int i = 0; i < A.nblocks-1; ++i) {
    row_offsets.push_back(c_size_offset_rows + row_rank_offset + A.U(i).cols);
    col_offsets.push_back(c_size_offset_cols + col_rank_offset + A.V(i).cols);

    row_rank_offset += A.U(i).cols;
    col_rank_offset += A.V(i).cols;
  }

  return {row_offsets, col_offsets};
}

std::tuple<std::vector<int64_t>, std::vector<int64_t> >
generate_rank_offsets(BLR2& A) {
  std::vector<int64_t> row_rank_splits, col_rank_splits;
  int64_t nrows = 0, ncols = 0;
  for (int i = 0; i < A.nblocks; ++i) {
    int64_t row_rank = A.U(i).cols, col_rank = A.V(i).cols;
    row_rank_splits.push_back(nrows + row_rank);
    col_rank_splits.push_back(ncols + col_rank);
    nrows += row_rank;
    ncols += col_rank;
  }

  return {row_rank_splits, col_rank_splits};
}

//
std::vector<Matrix> generate_UF_chain(Hatrix::BLR2& A) {
  std::vector<int64_t> row_offsets, col_offsets;
  std::tie(row_offsets, col_offsets) = generate_offsets(A);
  std::vector<Matrix> U_F;

  int64_t block_size = A.N / A.nblocks;

  for (int block = 0; block < A.nblocks; ++block) {
    Matrix UF_full = generate_identity_matrix(A.N, A.N);
    Matrix UF_block = make_complement(A.U(block));

    auto UF_full_splits = UF_full.split(row_offsets, col_offsets);
    auto UF_block_splits = SPLIT_DENSE(UF_block,
                                       block_size - A.U(block).cols,
                                       block_size - A.U(block).cols);
    int64_t permuted_nblocks = A.nblocks * 2;

    UF_full_splits[block * permuted_nblocks + block] = UF_block_splits[0];
    UF_full_splits[(block + A.nblocks) * permuted_nblocks + block] = UF_block_splits[2];
    UF_full_splits[block * permuted_nblocks + block + A.nblocks] = UF_block_splits[1];
    UF_full_splits[(block + A.nblocks) * permuted_nblocks + block + A.nblocks] = UF_block_splits[3];

    U_F.push_back(UF_full);
  }

  return U_F;
}

std::vector<Matrix> generate_VF_chain(Hatrix::BLR2& A) {
  int64_t block_size = A.N / A.nblocks;
  std::vector<int64_t> row_offsets, col_offsets;
  std::tie(row_offsets, col_offsets) = generate_offsets(A);
  std::vector<Matrix> V_F;

  for (int block = 0; block < A.nblocks; ++block) {
    Matrix VF_full = generate_identity_matrix(A.N, A.N);
    Matrix VF_block = make_complement(A.V(block));

    auto VF_full_splits = VF_full.split(row_offsets, col_offsets);
    auto VF_block_splits = SPLIT_DENSE(VF_block,
                                       block_size - A.U(block).cols,
                                       block_size - A.U(block).cols);
    int64_t permuted_nblocks = A.nblocks * 2;

    VF_full_splits[block * permuted_nblocks + block] = VF_block_splits[0];
    VF_full_splits[(block + A.nblocks) * permuted_nblocks + block] = VF_block_splits[2];
    VF_full_splits[block * permuted_nblocks + block + A.nblocks] = VF_block_splits[1];
    VF_full_splits[(block + A.nblocks) * permuted_nblocks + block + A.nblocks] = VF_block_splits[3];

    V_F.push_back(VF_full);
  }

  return V_F;
}

std::vector<Matrix> generate_L_chain(Hatrix::BLR2& A) {
  int64_t block_size = A.N / A.nblocks;
  int64_t permuted_nblocks = A.nblocks * 2;
  std::vector<int64_t> row_offsets, col_offsets;
  std::tie(row_offsets, col_offsets) = generate_offsets(A);
  std::vector<Matrix> L;

  for (int block = 0; block < A.nblocks; ++block) {
    Matrix L_block = generate_identity_matrix(A.N, A.N);
    auto L_splits = L_block.split(row_offsets, col_offsets);

    for (int j = 0; j <= block; ++j) {
      if (!A.is_admissible(block, j)) {
        int64_t row_split = block_size - A.U(block).cols;
        int64_t col_split = block_size - A.V(j).cols;
        auto D_splits = SPLIT_DENSE(A.D(block, j), row_split, col_split);

        // Copy the cc parts
        if (block == j) {
          L_splits[block * permuted_nblocks + j] = lower(D_splits[0]);
        }
        else {
          L_splits[block * permuted_nblocks + j] = D_splits[0];
        }

        L_splits[(block + A.nblocks) * permuted_nblocks + j] = D_splits[2];
      }
    }

    // Copy oc parts belonging to the 'upper' parts of the matrix
    for (int i = 0; i < block; ++i) {
      if (!A.is_admissible(i, block)) {
        int64_t row_split = block_size - A.U(i).cols;
        int64_t col_split = block_size - A.V(block).cols;
        auto D_splits = A.D(i, block).split(std::vector<int64_t>(1, row_split),
                                            std::vector<int64_t>(1, col_split));
        L_splits[(i + A.nblocks) * permuted_nblocks + block] = D_splits[2];
      }
    }

    L.push_back(L_block);
  }

  return L;
}

std::vector<Matrix> generate_U_chain(Hatrix::BLR2& A) {
  int64_t block_size = A.N / A.nblocks;
  int64_t permuted_nblocks = A.nblocks * 2;
  std::vector<int64_t> row_offsets, col_offsets;
  std::tie(row_offsets, col_offsets) = generate_offsets(A);
  std::vector<Matrix> U;

  for (int block = 0; block < A.nblocks; ++block) {
    Matrix U_block = generate_identity_matrix(A.N, A.N);
    auto U_splits = U_block.split(row_offsets, col_offsets);

    for (int i = 0; i <= block; ++i) {
      if (!A.is_admissible(i, block)) {
        int64_t row_split = block_size - A.U(i).cols;
        int64_t col_split = block_size - A.V(block).cols;
        auto D_splits = SPLIT_DENSE(A.D(i, block), row_split, col_split);

        // Copy the cc blocks
        if (block == i) {
          U_splits[i * permuted_nblocks + block] = upper(D_splits[0]);
        }
        else {
          U_splits[i * permuted_nblocks + block] = D_splits[0];
        }

        // Copy the co parts
        U_splits[i * permuted_nblocks + block + A.nblocks] = D_splits[1];
      }
    }

    for (int j = 0; j < block; ++j) {
      if (!A.is_admissible(block, j)) {
        int64_t row_split = block_size - A.U(block).cols;
        int64_t col_split = block_size - A.V(j).cols;
        auto D_splits = SPLIT_DENSE(A.D(block, j), row_split, col_split);
        U_splits[block * permuted_nblocks + (j + A.nblocks)] = D_splits[1];
      }
    }

    U.push_back(U_block);
  }

  return U;
}

Matrix generate_L0_permuted(Hatrix::BLR2& A, Hatrix::Matrix& last) {
  int64_t block_size = A.N / A.nblocks;
  int64_t permuted_nblocks = A.nblocks * 2;
  std::vector<int64_t> row_offsets, col_offsets;
  std::tie(row_offsets, col_offsets) = generate_offsets(A);

  Matrix L0 = generate_identity_matrix(A.N, A.N);
  auto L0_splits = L0.split(row_offsets, col_offsets);

  std::vector<int64_t> row_rank_splits, col_rank_splits;
  std::tie(row_rank_splits, col_rank_splits)= generate_rank_offsets(A);

  auto last_splits = last.split(row_rank_splits, col_rank_splits);

  for (int i = 0; i < A.nblocks; ++i) {
    for (int j = 0; j <= i; ++j) {
      if (!A.is_admissible(i, j)) {
        int64_t row_split = block_size - A.U(i).cols;
        int64_t col_split = block_size - A.V(j).cols;
        auto D_splits = A.D(i, j).split(std::vector<int64_t>(1, row_split),
                                        std::vector<int64_t>(1, col_split));

        // Copy the oo parts
        if (i == j) {
          L0_splits[(i + A.nblocks) * permuted_nblocks + (j + A.nblocks)] =
            lower(last_splits[i * A.nblocks + j]);
        }
        else {
          L0_splits[(i + A.nblocks) * permuted_nblocks + (j + A.nblocks)] =
            last_splits[i * A.nblocks + j];
        }
      }
      else {
        // Copy S blocks into the lower right corner
        L0_splits[(i + A.nblocks) * permuted_nblocks + (j + A.nblocks)] =
          last_splits[i * A.nblocks + j];
      }
    }
  }

  return L0;
}

Matrix generate_U0_permuted(Hatrix::BLR2& A, Hatrix::Matrix& last) {
  int64_t block_size = A.N / A.nblocks;
  int64_t permuted_nblocks = A.nblocks * 2;
  std::vector<int64_t> row_offsets, col_offsets;
  std::tie(row_offsets, col_offsets) = generate_offsets(A);

  Matrix U0 = generate_identity_matrix(A.N, A.N);
  auto U0_splits = U0.split(row_offsets, col_offsets);

  std::vector<int64_t> row_rank_splits, col_rank_splits;
  std::tie(row_rank_splits, col_rank_splits)= generate_rank_offsets(A);
  auto last_splits = last.split(row_rank_splits, col_rank_splits);

  for (int i = 0; i < A.nblocks; ++i) {
    for (int j = i; j < A.nblocks; ++j) {
      if (!A.is_admissible(i, j)) {
        int64_t row_split = block_size - A.U(i).cols;
        int64_t col_split = block_size - A.V(j).cols;
        auto D_splits = A.D(i, j).split(std::vector<int64_t>(1, row_split),
                                        std::vector<int64_t>(1, col_split));

        // Copy the oo parts
        if (i == j) {
          U0_splits[(i + A.nblocks) * permuted_nblocks + (j + A.nblocks)] =
            upper(last_splits[i * A.nblocks + j]);
        }
        else {
          U0_splits[(i + A.nblocks) * permuted_nblocks + (j + A.nblocks)] =
            last_splits[i * A.nblocks + j];
        }
      }
      else {
        // Copy S blocks
        U0_splits[(i + A.nblocks) * permuted_nblocks + (j + A.nblocks)] =
          last_splits[i * A.nblocks + j];
      }
    }
  }

  return U0;
}

Matrix chain_product(BLR2& A,
                     std::vector<Matrix>& U_F,
                     std::vector<Matrix>& L,
                     Matrix& L0, Matrix& U0,
                     std::vector<Matrix>& U,
                     std::vector<Matrix>& V_F) {
  Matrix product = generate_identity_matrix(A.N, A.N);

  // Multiply UF and L blocks.
  for (int i = 0; i < A.nblocks; ++i) {
    product = matmul(product, U_F[i]);
    product = matmul(product, L[i]);
  }

  product = matmul(product, L0);
  product = matmul(product, U0);

  for (int i = A.nblocks-1; i >= 0; --i) {
    product = matmul(product, U[i]);
    product = matmul(product, V_F[i], false, true);
  }

  return product;
}

Matrix unpermute_matrix(Matrix PA, BLR2& A) {
  Matrix M(A.N, A.N);
  int64_t block_size = A.N / A.nblocks;
  int64_t permuted_nblocks = A.nblocks * 2;
  std::vector<int64_t> row_offsets, col_offsets;
  std::tie(row_offsets, col_offsets) = generate_offsets(A);

  auto PA_splits = PA.split(row_offsets, col_offsets);
  auto M_splits = M.split(A.nblocks, A.nblocks);

  for (int i = 0; i < A.nblocks; ++i) {
    for (int j = 0; j < A.nblocks; ++j) {
      Matrix block(block_size, block_size);
      auto block_splits = SPLIT_DENSE(block,
                                      block_size - A.U(i).cols,
                                      block_size - A.V(j).cols);

      block_splits[0] = PA_splits[(i) * permuted_nblocks + j];
      block_splits[1] = PA_splits[i * permuted_nblocks + j + A.nblocks];
      block_splits[2] = PA_splits[(i + A.nblocks) * permuted_nblocks + j];
      block_splits[3] = PA_splits[(i + A.nblocks) * permuted_nblocks + j + A.nblocks];

      M_splits[i * A.nblocks + j] = block;
    }
  }

  return M;
}

Matrix generate_A0_matrix(BLR2& A) {
  Matrix A0 = generate_identity_matrix(A.N, A.N);
  int64_t block_size = A.N / A.nblocks;
  int64_t permuted_nblocks = A.nblocks * 2;
  std::vector<int64_t> row_offsets, col_offsets;
  std::tie(row_offsets, col_offsets) = generate_offsets(A);

  auto A0_splits = A0.split(row_offsets, col_offsets);

  for (int i = 0; i < A.nblocks; ++i) {
    for (int j = 0; j < A.nblocks; ++j) {
      if (A.is_admissible(i, j)) {
        A0_splits[(i + A.nblocks) * permuted_nblocks + (j + A.nblocks)] = A.S(i, j);
      }
      else {
        int64_t row_split = block_size - A.U(i).cols;
        int64_t col_split = block_size - A.V(j).cols;
        auto D_splits = SPLIT_DENSE(A.D(i, j), row_split, col_split);

        A0_splits[(i + A.nblocks) * permuted_nblocks + (j + A.nblocks)] =
          D_splits[3];
      }
    }
  }

  return A0;
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
  Hatrix::BLR2 A_expected_blr(A);
  double construct_error = A.construction_relative_error(randpts);
  Matrix last; RowColMap<Matrix> F;
  std::tie(last, F) = A.factorize(randpts);

  // Multiply by UF and VF.
  Matrix A_expected = generate_laplacend_matrix(randpts, A.N, A.N, 0, 0);
  //generate_full_permuted(A_expected_blr);

  // Generate permuted L and U matrices.
  std::vector<Matrix> U_F = generate_UF_chain(A);
  std::vector<Matrix> V_F = generate_VF_chain(A);
  std::vector<Matrix> L = generate_L_chain(A);
  std::vector<Matrix> U = generate_U_chain(A);
  Matrix L0 = generate_L0_permuted(A, last);
  Matrix U0 = generate_U0_permuted(A, last);
  Matrix A0 = generate_A0_matrix(A);

  double A0_acc = norm(matmul(L0, U0) - A0) / norm(A0);

  Matrix A_actual = unpermute_matrix(chain_product(A, U_F, L, L0, U0, U, V_F), A);

  Matrix diff = (A_actual - A_expected);
  auto d_splits = diff.split(A.nblocks, A.nblocks);
  auto m_splits = A_expected.split(A.nblocks, A.nblocks);

  // int idx;

  std::cout << "-- BLR2 verification --\n";
  for (int i = 0; i < nblocks; ++i) {
    for (int j = 0; j < nblocks; ++j) {
      int idx = i * A.nblocks + j;
      std::cout << "(" << i << "," << j << ") block rel error: " << (norm(d_splits[idx]) / norm(m_splits[idx])) << std::endl;
    }
  }

  // (A_actual - A_expected).print();

  double acc = norm(A_actual - A_expected) / norm(A_expected);

  Hatrix::Context::finalize();

  std::cout << "A0 accuracy: " << A0_acc << " factorize accuracy: "
            << acc << " contruct error: " << construct_error  << std::endl;
}
