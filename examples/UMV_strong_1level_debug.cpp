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
        // for (int j = 0; j < nblocks; ++j) {
        is_admissible.insert(i, i, std::abs(i - i) > admis);
        // }
      }

      is_admissible.insert(1, 2, admis == 1 ? false : true);
      is_admissible.insert(2, 1, admis == 1 ? false : true);
      is_admissible.insert(3, 2, admis == 1 ? false : true);
      is_admissible.insert(2, 3, admis == 1 ? false : true);


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
    Matrix factorize(const randvec_t &randpts) {
      int block_size = N / nblocks;
      RowColMap<Matrix> F;      // fill-in blocks.

      for (int block = 0; block < nblocks; ++block) {
        if (block == 3 && admis == 1) {
          Matrix I = generate_identity_matrix(block_size, block_size);

          // Compute row bases from fill-in.
          Matrix gramian = matmul(F(3, 1), F(3, 1), false, true);
          Matrix range = matmul(U(3), U(3), false, true);
          Matrix diff = I - range;
          Matrix G31 = matmul(matmul(diff, gramian), diff, false, true);
          Matrix U3_add, _S, _V; double error;
          std::tie(U3_add, _S, _V, error) = truncated_svd(G31, rank);
          int nbases = significant_bases(_S);
          U3_add.shrink(block_size, nbases);
          Matrix U3_copy(U(3));
          U.erase(3);
          U.insert(3, concat(U3_copy, U3_add, 1));

          // Compute col bases from fill-in.
          gramian = matmul(F(1, 3), F(1, 3), false, true);
          range = matmul(V(3), V(3), false, true);
          diff = I - range;
          Matrix G13 = matmul(matmul(diff, gramian), diff, false, true);
          Matrix V3_add;
          std::tie(V3_add, _S, _V, error) = truncated_svd(G13, rank);
          nbases = significant_bases(_S);
          V3_add.shrink(block_size, nbases);
          Matrix V3_copy(V(3));
          V.erase(3);
          V.insert(3, concat(V3_copy, V3_add, 1));
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
              else {
                // Generate the fill-in block that holds the product here and that from
                // Schur's compliment present below this block.
                Matrix fill_in(block_size, block_size);
                auto fill_splits = SPLIT_DENSE(fill_in,
                                               block_size - U(i).cols,
                                               block_size - V(j).cols);
                matmul(lower_splits[2], right_splits[1], fill_splits[3], false, false, 1.0, 1.0);
                F.insert(i, j, std::move(fill_in));
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
              else {
                Matrix& fill_in = F(i, j);
                auto fill_splits = SPLIT_DENSE(fill_in,
                                               block_size - U(i).cols,
                                               block_size - V(j).cols);
                matmul(lower_splits[0], right_splits[1], fill_splits[1], false, false, 1.0, 1.0);
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
              // Schur's compliement between co and cc blocks where product exists as dense.
              if (!is_admissible(i, j)) {
                auto reduce_splits = SPLIT_DENSE(D(i, j),
                                                 block_size - U(i).cols,
                                                 block_size - V(j).cols);
                matmul(lower_splits[2], right_splits[0], reduce_splits[2],
                       false, false, -1.0, 1.0);
              }
              // Schur's compliement between co and cc blocks where a new fill-in is created.
              else {
                Matrix& fill_in = F(i, j);
                auto fill_splits = SPLIT_DENSE(fill_in,
                                               block_size - U(i).cols,
                                               block_size - V(j).cols);
                matmul(lower_splits[2], right_splits[0], fill_splits[2], false, false, 1.0, 1.0);
              }
            }
          }
        }
      } // for (int block = 0; block < nblocks; ++block)

      // Append zeros to admissible blocks in the same row as a fill-in. This is detected
      // when there is a dimension mismatch between the number of bases and the nrows of
      // S block.
      for (int ib = 0; ib < nblocks; ++ib) {
        int64_t row_rank = U(ib).cols;
        for (int jb = 0; jb < nblocks; ++jb) {
          int64_t col_rank = V(jb).cols;
          if (is_admissible(ib, jb)) {
            Matrix& oldS = S(ib, jb);
            Matrix newS(row_rank, col_rank);
            if (S(ib, jb).rows != row_rank || S(ib, jb).cols != col_rank) {
              // Zero-pad the rows of this S block.
              for (int i = 0; i < oldS.rows; ++i) {
                for (int j = 0; j < oldS.cols; ++j) {
                  newS(i, j) = oldS(i, j);
                }
              }

              S.erase(ib, jb);
              S.insert(ib, jb, std::move(newS));
            }
          }
        }
      }

      // Update S blocks for admissible blocks that have fill-ins.
      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          if (F.exists(i, j)) {
            Matrix& newS = S(i, j);
            newS = S(i, j) + matmul(matmul(U(i), F(i, j), true), V(j));
          }
        }
      }

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

void multiply_compliments(Hatrix::BLR2& A) {
  int64_t nblocks = A.nblocks;

  for (int i = 0; i < nblocks; ++i) {
    for (int j = 0; j < nblocks; ++j) {
      if (!A.is_admissible(i, j)) {
        Matrix U_F = make_complement(A.U(i));
        Matrix V_F = make_complement(A.V(j));
        A.D(i, j) = matmul(matmul(U_F, A.D(i, j), true), V_F);
      }
    }
  }
}

Matrix generate_UFbar(Hatrix::BLR2& A) {
  Matrix UFbar(A.N, A.N);
  auto UFbar_splits = UFbar.split(A.nblocks, A.nblocks);

  for (int i = 0; i < A.nblocks; ++i) {
    UFbar_splits[i * A.nblocks + i] = make_complement(A.U(i));
  }

  return UFbar;
}

Matrix generate_VFbar(Hatrix::BLR2& A) {
  Matrix VFbar(A.N, A.N);
  auto VFbar_splits = VFbar.split(A.nblocks, A.nblocks);
  for (int i = 0; i < A.nblocks; ++i) {
    VFbar_splits[i * A.nblocks + i] = make_complement(A.V(i));
  }
  return VFbar;
}

Matrix generate_unpermuted(Hatrix::Matrix Aperm, Hatrix::BLR2& Ablr) {
  Matrix A(Ablr.N, Ablr.N);
  std::vector<int64_t> row_offsets, col_offsets;

  int64_t c_size_offset_rows = 0, c_size_offset_cols = 0;
  for (int i = 0; i < A.nblocks; ++i) {
    int64_t c_size_rows = block_size - A.U(i).cols;
    int64_t c_size_cols = block_size - A.V(i).cols;
    row_offsets.push_back(c_size_offset_rows + c_size_rows);
    col_offsets.push_back(c_size_offset_cols + c_size_cols);

    c_size_offset_rows += c_size_rows;
    c_size_offset_cols += c_size_cols;
  }

  for (int i = 0; i < A.nblocks-1; ++i) {
    row_offsets.push_back(c_size_offset_rows + (i+1) * A.U(i).cols);
    col_offsets.push_back(c_size_offset_cols + (i+1) * A.V(i).cols );
  }

  auto A_splits = A.split(row_offfsets, col_offsets);
  auto Aperm_splits = Aperm.split(row_offfsets, col_offsets);

  for (int i = 0; i < Ablr.nblocks; ++i) {
    for (int j = 0; j < Ablr.nblocks; ++j) {
      if (!Ablr.is_admissible(i, j)) {

      }
    }
  }


  return A;
}

Matrix generate_UFbar_permuted(Hatrix::BLR2& A) {
  int64_t block_size = A.N / A.nblocks;
  Matrix UFbar(A.N, A.N);
  std::vector<int64_t> row_offsets, col_offsets;

  int64_t c_size_offset_rows = 0, c_size_offset_cols = 0;
  for (int i = 0; i < A.nblocks; ++i) {
    int64_t c_size_rows = block_size - A.U(i).cols;
    int64_t c_size_cols = block_size - A.V(i).cols;
    row_offsets.push_back(c_size_offset_rows + c_size_rows);
    col_offsets.push_back(c_size_offset_cols + c_size_cols);

    c_size_offset_rows += c_size_rows;
    c_size_offset_cols += c_size_cols;
  }

  for (int i = 0; i < A.nblocks-1; ++i) {
    row_offsets.push_back(c_size_offset_rows + (i+1) * A.U(i).cols);
    col_offsets.push_back(c_size_offset_cols + (i+1) * A.V(i).cols );
  }

  auto UFbar_splits = UFbar.split(row_offsets, col_offsets);
  int64_t permuted_nblocks = 2 * A.nblocks;

  for (int i = 0; i < A.nblocks; ++i) {
    Matrix U_F = make_complement(A.U(i));
    int64_t row_split = block_size - A.U(i).cols, col_split = block_size - A.V(i).cols;

    auto UF_splits = U_F.split(std::vector<int64_t>(1, row_split),
                               std::vector<int64_t>(1, col_split));

    // Copy cc block.
    UFbar_splits[i * permuted_nblocks + i] = UF_splits[0];

    // Copy oc block.
    UFbar_splits[(i + A.nblocks) * permuted_nblocks + i].print_meta();
    UF_splits[2].print_meta();

    UFbar_splits[(i + A.nblocks) * permuted_nblocks + i] = UF_splits[2];

    // Copy co block.
    UFbar_splits[i * permuted_nblocks + i + A.nblocks] = UF_splits[1];

    // Copy oo block.
    UFbar_splits[(i + A.nblocks) * permuted_nblocks + i + A.nblocks] = UF_splits[3];
  }

  return UFbar;
}

Matrix generate_VFbar_permuted(Hatrix::BLR2& A) {
  int64_t block_size = A.N / A.nblocks;
  Matrix VFbar(A.N, A.N);
  std::vector<int64_t> row_offsets, col_offsets;

  int64_t c_size_offset_rows = 0, c_size_offset_cols = 0;
  for (int i = 0; i < A.nblocks; ++i) {
    int64_t c_size_rows = block_size - A.U(i).cols;
    int64_t c_size_cols = block_size - A.V(i).cols;
    row_offsets.push_back(c_size_offset_rows + c_size_rows);
    col_offsets.push_back(c_size_offset_cols + c_size_cols);

    c_size_offset_rows += c_size_rows;
    c_size_offset_cols += c_size_cols;
  }

  for (int i = 0; i < A.nblocks-1; ++i) {
    row_offsets.push_back(c_size_offset_rows + (i+1) * A.U(i).cols);
    col_offsets.push_back(c_size_offset_cols + (i+1) * A.V(i).cols );
  }

  auto VFbar_splits = VFbar.split(row_offsets, col_offsets);
  int64_t permuted_nblocks = 2 * A.nblocks;

  for (int i = 0; i < A.nblocks; ++i) {
    Matrix V_F = make_complement(A.V(i));
    int64_t row_split = block_size - A.U(i).cols, col_split = block_size - A.V(i).cols;

    auto VF_splits = V_F.split(std::vector<int64_t>(1, row_split),
                               std::vector<int64_t>(1, col_split));

    // Copy cc block.
    VFbar_splits[i * permuted_nblocks + i] = VF_splits[0];

    // Copy oc block.
    VFbar_splits[(i + A.nblocks) * permuted_nblocks + i].print_meta();
    VF_splits[2].print_meta();

    VFbar_splits[(i + A.nblocks) * permuted_nblocks + i] = VF_splits[2];

    // Copy co block.
    VFbar_splits[i * permuted_nblocks + i + A.nblocks] = VF_splits[1];

    // Copy oo block.
    VFbar_splits[(i + A.nblocks) * permuted_nblocks + i + A.nblocks] = VF_splits[3];
  }

  return VFbar;
}

double factorization_accuracy(Matrix& full_matrix, BLR2& A_expected) {
  double s_diff = 0, d_diff = 0, actual = 0;
  int64_t nblocks = A_expected.nblocks;
  int64_t block_size = A_expected.N / nblocks;
  int64_t c_size = block_size - A_expected.rank;
  auto full_splits = full_matrix.split(nblocks, nblocks);

  for (int i = 0; i < nblocks; ++i) {
    for (int j = 0; j < nblocks; ++j) {
      Matrix block(full_splits[i * nblocks + j]);
      if (A_expected.is_admissible(i, j)) {
        auto block_splits = block.split(std::vector<int64_t>(1, c_size),
                                        std::vector<int64_t>(1, c_size));
        actual += pow(norm(A_expected.S(i, j)), 2);
        s_diff += pow(norm(block_splits[3] - A_expected.S(i, j)), 2);
      }
      else {
        std::cout << " i-> " << i+1 << " j-> " << j+1 << std::endl;
        (block - A_expected.D(i, j)).print();
        actual += pow(norm(block), 2);
        d_diff += pow(norm(block - A_expected.D(i, j)), 2);
      }
    }
  }

  std::cout << "S diff: " << s_diff << " D diff: " << d_diff << std::endl;

  return std::sqrt((s_diff + d_diff) / actual);
}

Matrix generate_L_permuted(BLR2& A, Matrix& last) {
  int64_t block_size = A.N / A.nblocks;
  int64_t c_size = block_size - A.rank;
  Matrix L(A.N, A.N);

  std::vector<int64_t> row_offsets, col_offsets;

  int64_t c_size_offset_rows = 0, c_size_offset_cols = 0;
  for (int i = 0; i < A.nblocks; ++i) {
    int64_t c_size_rows = block_size - A.U(i).cols;
    int64_t c_size_cols = block_size - A.V(i).cols;
    row_offsets.push_back(c_size_offset_rows + c_size_rows);
    col_offsets.push_back(c_size_offset_cols + c_size_cols);

    c_size_offset_rows += c_size_rows;
    c_size_offset_cols += c_size_cols;
  }

  for (int i = 0; i < A.nblocks-1; ++i) {
    row_offsets.push_back(c_size_offset_rows + (i+1) * A.U(i).cols);
    col_offsets.push_back(c_size_offset_cols + (i+1) * A.V(i).cols );
  }

  // Merge unfactorized portions.
  std::vector<int64_t> row_rank_splits, col_rank_splits;
  int64_t nrows = 0, ncols = 0;
  for (int i = 0; i < A.nblocks; ++i) {
    int64_t row_rank = A.U(i).cols, col_rank = A.V(i).cols;
    row_rank_splits.push_back(nrows + row_rank);
    col_rank_splits.push_back(ncols + col_rank);
    nrows += row_rank;
    ncols += col_rank;
  }

  auto L_splits = L.split(row_offsets, col_offsets);
  auto last_splits = last.split(row_rank_splits, col_rank_splits);

  int64_t permuted_nblocks = A.nblocks * 2;

  for (int i = 0; i < A.nblocks; ++i) {
    for (int j = 0; j <= i; ++j) {
      if (!A.is_admissible(i, j)) {
        int64_t row_split = block_size - A.U(i).cols;
        int64_t col_split = block_size - A.V(j).cols;
        auto D_splits = A.D(i, j).split(std::vector<int64_t>(1, row_split),
                                        std::vector<int64_t>(1, col_split));

        // Copy the cc parts
        if (i == j) {
          L_splits[i * permuted_nblocks + j] = lower(D_splits[0]);
        }
        else {
          L_splits[i * permuted_nblocks + j] = D_splits[0];
        }

        // Copy the oc parts
        L_splits[(i + A.nblocks) * permuted_nblocks + j] = D_splits[2];

        // Copy the oo parts
        if (i == j) {
          L_splits[(i + A.nblocks) * permuted_nblocks + (j + A.nblocks)] =
            lower(last_splits[i * A.nblocks + j]);
        }
        else {
          L_splits[(i + A.nblocks) * permuted_nblocks + (j + A.nblocks)] =
            last_splits[i * A.nblocks + j];
        }
      }
      else {
        // Copy S blocks into the lower right corner
        L_splits[(i + A.nblocks) * permuted_nblocks + (j + A.nblocks)] =
          last_splits[i * A.nblocks + j];
      }
    }
  }

  // Copy oc parts belonging to the 'upper' parts of the matrix
  for (int i = 0; i < A.nblocks; ++i) {
    for (int j = i+1; j < A.nblocks; ++j) {
      if (!A.is_admissible(i, j)) {
        int64_t row_split = block_size - A.U(i).cols;
        int64_t col_split = block_size - A.V(j).cols;
        auto D_splits = A.D(i, j).split(std::vector<int64_t>(1, row_split),
                                        std::vector<int64_t>(1, col_split));
        L_splits[(i + A.nblocks) * permuted_nblocks + j] = D_splits[2];
      }
    }
  }

  return L;
}

Matrix generate_U_permuted(BLR2& A, Matrix& last) {
  int64_t block_size = A.N / A.nblocks;
  int64_t c_size = block_size - A.rank;
  Matrix U(A.N, A.N);

  std::vector<int64_t> row_offsets, col_offsets;

  int64_t c_size_offset_rows = 0, c_size_offset_cols = 0;
  for (int i = 0; i < A.nblocks; ++i) {
    int64_t c_size_rows = block_size - A.U(i).cols;
    int64_t c_size_cols = block_size - A.V(i).cols;
    row_offsets.push_back(c_size_offset_rows + c_size_rows);
    col_offsets.push_back(c_size_offset_cols + c_size_cols);

    c_size_offset_rows += c_size_rows;
    c_size_offset_cols += c_size_cols;
  }

  for (int i = 0; i < A.nblocks-1; ++i) {
    row_offsets.push_back(c_size_offset_rows + (i+1) * A.U(i).cols);
    col_offsets.push_back(c_size_offset_cols + (i+1) * A.V(i).cols );
  }

  // Merge unfactorized portions.
  std::vector<int64_t> row_rank_splits, col_rank_splits;
  int64_t nrows = 0, ncols = 0;
  for (int i = 0; i < A.nblocks; ++i) {
    int64_t row_rank = A.U(i).cols, col_rank = A.V(i).cols;
    row_rank_splits.push_back(nrows + row_rank);
    col_rank_splits.push_back(ncols + col_rank);
    nrows += row_rank;
    ncols += col_rank;
  }

  auto U_splits = U.split(row_offsets, col_offsets);
  auto last_splits = last.split(A.nblocks, A.nblocks);
  int64_t permuted_nblocks = A.nblocks * 2;

  for (int i = 0; i < A.nblocks; ++i) {
    for (int j = i; j < A.nblocks; ++j) {
      if (!A.is_admissible(i, j)) {
        int64_t row_split = block_size - A.U(i).cols;
        int64_t col_split = block_size - A.V(j).cols;
        auto D_splits = A.D(i, j).split(std::vector<int64_t>(1, row_split),
                                        std::vector<int64_t>(1, col_split));
        // Copy the cc blocks
        if (i == j) {
          U_splits[i * permuted_nblocks + j] = upper(D_splits[0]);
        }
        else {
          U_splits[i * permuted_nblocks + j] = D_splits[0];
        }

        // Copy the co parts
        U_splits[i * permuted_nblocks + j + A.nblocks] = D_splits[1];

        // Copy the oo parts
        if (i == j) {
          U_splits[(i + A.nblocks) * permuted_nblocks + (j + A.nblocks)] =
            upper(last_splits[i * A.nblocks + j]);
        }
        else {
          U_splits[(i + A.nblocks) * permuted_nblocks + (j + A.nblocks)] =
            last_splits[i * A.nblocks + j];
        }
      }
      else {
        // Copy S blocks
        U_splits[(i + A.nblocks) * permuted_nblocks + (j + A.nblocks)] =
          last_splits[i * A.nblocks + j];
      }
    }
  }

  // Cupy co blocks that actually exist in the lower part of the matrix.
  for (int i = 0; i < A.nblocks; ++i) {
    for (int j = 0; j < i; ++j) {
      if (!A.is_admissible(i, j)) {
        int64_t row_split = block_size - A.U(i).cols;
        int64_t col_split = block_size - A.V(j).cols;
        auto D_splits = A.D(i, j).split(std::vector<int64_t>(1, row_split),
                                        std::vector<int64_t>(1, col_split));

        U_splits[(i) * permuted_nblocks + (j + A.nblocks)] = D_splits[1];
      }
    }
  }

  return U;
}

Matrix generate_full_permuted(BLR2& A) {
  Matrix M(A.N, A.N);
  int64_t block_size = A.N / A.nblocks;
  int64_t c_size = block_size - A.rank;
  std::vector<int64_t> row_offsets, col_offsets;
  for (int i = 0; i < A.nblocks; ++i) {
    row_offsets.push_back((i+1) * c_size);
    col_offsets.push_back((i+1) * c_size);
  }
  for (int i = 0; i < A.nblocks-1; ++i) {
    row_offsets.push_back(c_size*(A.nblocks) + (i+1) * A.rank);
    col_offsets.push_back(c_size*(A.nblocks) + (i+1) * A.rank);
  }

  auto M_splits = M.split(row_offsets, col_offsets);
  int64_t permuted_nblocks = A.nblocks * 2;

  for (int i = 0; i < A.nblocks; ++i) {
    for (int j = 0; j < A.nblocks; ++j) {

      if (!A.is_admissible(i, j)) {
        auto D_splits = A.D(i, j).split(std::vector<int64_t>(1, c_size),
                                        std::vector<int64_t>(1, c_size));
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
  Hatrix::BLR2 A_expected(A);
  double construct_error = A.construction_relative_error(randpts);
  auto last = A.factorize(randpts);

  // multiply_compliments(A_expected);

  Matrix UFbar_permuted = generate_UFbar_permuted(A);
  Matrix VFbar_permuted = generate_VFbar_permuted(A);

  Matrix UFbar = generate_UFbar(A);
  Matrix VFbar = generate_VFbar(A);

  Matrix L_permuted = generate_L_permuted(A, last);
  Matrix U_permuted = generate_U_permuted(A, last);
  Matrix A1 = generate_unpermuted(matmul(L_permuted, U_permuted), A);

  matmul(matmul(UFbar, A1), VFbar, false, true).print();
  // Matrix tt = matmul(matmul(UFbar_permuted, matmul(L_permuted, U_permuted)), VFbar_permuted, false, true);
  Matrix tt = matmul(L_permuted, U_permuted);
  Matrix ff = generate_full_permuted(A_expected);
  // (tt - Hatrix::generate_laplacend_matrix(randpts, N, N, 0, 0, PV)).print();
  // (tt - ff).print();
  double acc = pow(norm(tt - ff), 2);

  Hatrix::Context::finalize();

  std::cout << "accuracy: " << acc << std::endl;
}
