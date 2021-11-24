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

              F.erase(block, icol);
            }
          }
          if (update_bases) {
            std::tie(Utemp, Stemp, Vtemp, error) =
              Hatrix::truncated_svd(Urow_bases_concat, rank);
            U.erase(block);
            U.insert(block, std::move(Utemp));
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

              F.erase(irow, block);
            }
          }

          if (update_bases) {
            std::tie(Utemp, Stemp, Vtemp, error) = Hatrix::truncated_svd(Vcol_bases_concat,
                                                                         rank);
            V.erase(block);
            V.insert(block, std::move(transpose(Vtemp)));
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

        // // Multiplication of UF with block ahead of the diagonal block on the same row
        // for (int icol = block+1; icol < nblocks; ++icol) {
        //   if (!is_admissible(block, icol)) {
        //     D(block, icol) = matmul(U_F, D(block, icol), true);
        //   }
        // }

        // // Multiplication of VF with block below this diagonal on the same column.
        // for (int irow = block+1; irow < nblocks; ++irow) {
        //   if (!is_admissible(irow, block)) {
        //     D(irow, block) = matmul(D(irow, block), V_F);
        //   }
        // }

        // // Multiplication of UF with blocks before the diagonal block on the same row
        // for (int icol = 0; icol < block; ++icol) {
        //   if (!is_admissible(block, icol)) {
        //     auto bottom_splits = D(block, icol).split({}, std::vector<int64_t>(1, c_size));
        //     Matrix o = matmul(U_F, bottom_splits[1], true);
        //     bottom_splits[1] = o;
        //   }
        // }

        // // Multiplication of VF with blocks before the diagoanl block on the same column.
        // for (int irow = 0; irow < block; ++irow) {
        //   if (!is_admissible(irow, block)) {
        //     auto right_splits = D(irow, block).split(std::vector<int64_t>(1, c_size), {});
        //     Matrix o = matmul(right_splits[1], V_F);
        //     right_splits[1] = o;
        //   }
        // }

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
          auto right_splits = D(block, icol).split(std::vector<int64_t>(1, c_size), {});
          // TRSM between cc block on diagonal and c block on the right
          solve_triangular(Dcc, right_splits[0], Hatrix::Left, Hatrix::Lower, true);
          // Use the lower part of the trapezoid
          matmul(Doc, right_splits[0], right_splits[1], false, false, -1.0, 1.0);
        }

        // Perform TRSM between A.VF blocks on the bottom of this diagonal block and
        // the cc of the diagonal.
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


        // Perform upper TRSM between A.VF blocks (oc part) above this diagonal
        // blocks and cc of diagonal block.
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
            auto left_splits = D(block, icol).split(std::vector<int64_t>(1, c_size),
                                                    std::vector<int64_t>(1, c_size));
            auto reduce_splits = D(irow, icol).split(std::vector<int64_t>(1, c_size),
                                                     std::vector<int64_t>(1, c_size));
            matmul(top_splits[2], left_splits[1], reduce_splits[3], false, false, -1.0, 1.0);
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
            std::cout << "left: " << irow << "," << block << " up: "<< block << "," << icol
                      << " trail: " << irow << "," << icol << std::endl;
            //  matmul(bottom_block[0], right_block[0], D(irow, icol), false, false -1.0, 1.0);
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
                                                      std::vector<int64_t>(1, c_size));
            Matrix fill_in(block_size, block_size);
            auto fill_in_splits = fill_in.split({}, std::vector<int64_t>(1, block_size - rank));
            Matrix t = matmul(bottom_splits[0], left_splits[1]);
            fill_in_splits[1] = t;
            // F.insert(irow, icol, std::move(fill_in));
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
  int64_t block_size = A.N / nblocks;
  int64_t c_size = block_size - A.rank;

  for (int i = 0; i < nblocks; ++i) {
    for (int j = 0; j < nblocks; ++j) {
      if (!A.is_admissible(i, j)) {
        Matrix U_F = make_complement(A.U(i));
        Matrix V_F = make_complement(A.V(j));
        A.D(i, j) = matmul(matmul(U_F, A.D(i, j), true), V_F);
      }
    }
  }

  // for (int block = 0; block < A.nblocks; ++block) {
  //   if (!A.is_admissible(i, j))
  //   Matrix U_F = make_complement(A.U(block));
  //   Matrix V_F = make_complement(A.V(block));
  //   A.D(block, block) = matmul(matmul(U_F, A.D(block, block), true), V_F);

  //   // Multiplication of UF with block ahead of the diagonal block on the same row
  //   for (int icol = block+1; icol < A.nblocks; ++icol) {
  //     if (!A.is_admissible(block, icol)) {
  //       A.D(block, icol) = matmul(U_F, A.D(block, icol), true);
  //     }
  //   }

  //   // Multiplication of VF with block below this diagonal on the same column.
  //   for (int irow = block+1; irow < A.nblocks; ++irow) {
  //     if (!A.is_admissible(irow, block)) {
  //       A.D(irow, block) = matmul(A.D(irow, block), V_F);
  //     }
  //   }

  //   // Multiplication of UF with blocks before the diagonal block on the same row
  //   for (int icol = 0; icol < block; ++icol) {
  //     if (!A.is_admissible(block, icol)) {
  //       auto bottom_splits = A.D(block, icol).split({}, std::vector<int64_t>(1, c_size));
  //       Matrix o = matmul(U_F, bottom_splits[1], true);
  //       bottom_splits[1] = o;
  //     }
  //   }

  //   // Multiplication of VF with blocks before the diagoanl block on the same column.
  //   for (int irow = 0; irow < block; ++irow) {
  //     if (!A.is_admissible(irow, block)) {
  //       auto right_splits = A.D(irow, block).split(std::vector<int64_t>(1, c_size), {});
  //       Matrix o = matmul(right_splits[1], V_F);
  //       right_splits[1] = o;
  //     }
  //   }
  // }
}

Matrix generate_A0(Hatrix::BLR2& A, Hatrix::Matrix& last) {
  int64_t block_size = A.N / A.nblocks;
  int64_t c_size = block_size - A.rank;
  double expected = 0, diff = 0;

  Hatrix::Matrix A0(A.N, A.N);
  auto A0_splits = A0.split(A.nblocks, A.nblocks);

  for (int i = 0; i < A.nblocks; ++i) {
    A0_splits[i * A.nblocks + i] = Hatrix::generate_identity_matrix(block_size, block_size);
  }
  Hatrix::Matrix last_lower = lower(last);
  Hatrix::Matrix last_upper = upper(last);
  Hatrix::Matrix result = matmul(last_lower, last_upper);

  // Assemble the permuted matrix of Aoo and S blocks.
  auto result_splits = result.split(A.nblocks, A.nblocks);
  for (int i = 0; i < A.nblocks; ++i) {
    for (int j = 0; j < A.nblocks; ++j) {
      Matrix block(block_size, block_size);
      if (i == j) {
        block = generate_identity_matrix(block_size, block_size);
      }
      auto block_splits = block.split(std::vector<int64_t>(1, c_size),
                                      std::vector<int64_t>(1, c_size));
      block_splits[3] = result_splits[i * A.nblocks + j];
      A0_splits[i * A.nblocks + j] = block;
    }
  }

  return A0;
}

Matrix generate_L0(BLR2& A) {
  int64_t nblocks = A.nblocks;
  int64_t block_size = A.N / nblocks;
  int64_t c_size = block_size - A.rank;
  Hatrix::Matrix L0(A.N, A.N);

  auto L0_splits = L0.split(nblocks, nblocks);
  for (int i = 0; i < nblocks; ++i) {
    Matrix block = generate_identity_matrix(block_size, block_size);
    auto block_splits = block.split(std::vector<int64_t>(1, c_size),
                                    std::vector<int64_t>(1, c_size));
    auto D_splits = A.D(i, i).split(std::vector<int64_t>(1, c_size),
                                    std::vector<int64_t>(1, c_size));

    block_splits[0] = lower(D_splits[0]);
    block_splits[2] = D_splits[2];
    L0_splits[i * nblocks + i] = block;

    // Populate the blocks c blocks below the diagonal.
    for (int irow = i + 1; irow < nblocks; ++irow) {
      if (!A.is_admissible(irow, i)) {
        Matrix block(block_size, block_size);
        auto D_splits = A.D(irow, i).split({}, std::vector<int64_t>(1, c_size));
        auto block_splits = block.split({}, std::vector<int64_t>(1, c_size));

        block_splits[0] = D_splits[0];
        L0_splits[irow * nblocks + i] = block;
      }
    }

    // Populate the rows above the diagonal block with oc strips.
    for (int irow = 0; irow < i; ++irow) {
      if (!A.is_admissible(irow, i)) {
        Matrix block(block_size, block_size);
        auto D_splits = A.D(irow, i).split(std::vector<int64_t>(1, c_size),
                                           std::vector<int64_t>(1, c_size));
        auto block_splits = block.split(std::vector<int64_t>(1, c_size),
                                        std::vector<int64_t>(1, c_size));
        block_splits[2] = D_splits[2];
        L0_splits[irow * nblocks + i] = block;
      }
    }
  }

  return L0;
}

Matrix generate_U0(BLR2& A) {
  Matrix U0(A.N, A.N);
  int64_t nblocks = A.nblocks;
  int64_t block_size = A.N / A.nblocks;
  int64_t c_size = block_size - A.rank;

  auto U0_splits = U0.split(A.nblocks, A.nblocks);
  for (int i = 0; i < A.nblocks; ++i) {
    Matrix block = generate_identity_matrix(block_size, block_size);
    auto block_splits = block.split(std::vector<int64_t>(1, c_size),
                                    std::vector<int64_t>(1, c_size));
    auto D_splits = A.D(i, i).split(std::vector<int64_t>(1, c_size),
                                    std::vector<int64_t>(1, c_size));
    block_splits[0] = upper(D_splits[0]);
    block_splits[1] = D_splits[1];
    U0_splits[i * nblocks + i] = block;

    // Populate the columns to the right of the block diagonal
    for (int icol = i + 1; icol < nblocks; ++icol) {
      if (!A.is_admissible(i, icol))  {
        Matrix block(block_size, block_size);
        auto D_splits = A.D(i, icol).split(std::vector<int64_t>(1, c_size), {});
        auto block_splits = block.split(std::vector<int64_t>(1, c_size), {});
        block_splits[0] = D_splits[0];

        U0_splits[i * nblocks + icol] = block;
      }
    }

    for (int icol = 0; icol < i; ++icol) {
      if (!A.is_admissible(i, icol)) {
        Matrix block(block_size, block_size);
        auto D_splits = A.D(i, icol).split(std::vector<int64_t>(1, c_size),
                                           std::vector<int64_t>(1, c_size));
        auto block_splits = block.split(std::vector<int64_t>(1, c_size),
                                        std::vector<int64_t>(1, c_size));
        block_splits[1] = D_splits[1];

        U0_splits[i * nblocks + icol] = block;
      }
    }
  }


  return U0;
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

double check_A0_accuracy(Matrix& A0, BLR2& A) {
  int64_t nblocks = A.nblocks;
  int64_t block_size = A.N / nblocks;
  int64_t c_size = block_size - A.rank;
  double s_diff = 0, d_diff = 0, actual = 0;
  auto A0_splits = A0.split(nblocks, nblocks);

  for (int i = 0; i < nblocks; ++i) {
    for (int j = 0; j < nblocks; ++j) {
      Matrix block(A0_splits[i * nblocks + j]);
      auto block_splits = block.split(std::vector<int64_t>(1, c_size),
                                      std::vector<int64_t>(1, c_size));
      Matrix oo(block_splits[3]);
      actual += pow(norm(oo), 2);
      if (A.is_admissible(i, j)) {
        s_diff += pow(norm(A.S(i, j) - oo), 2);
      }
      else {
        auto D_splits = A.D(i, j).split(std::vector<int64_t>(1, c_size),
                                        std::vector<int64_t>(1, c_size));
        // (D_splits[3] - oo).print();
        d_diff += pow(norm(D_splits[3] - oo), 2);
      }
    }
  }

  std::cout << "A0 acc d: " << d_diff << " s: " << s_diff << std::endl;

  return std::sqrt((d_diff + s_diff) / actual);
}

Matrix generate_L_permuted(BLR2& A, Matrix& last) {
  int64_t block_size = A.N / A.nblocks;
  int64_t c_size = block_size - A.rank;
  Matrix L(A.N, A.N);

  std::vector<int64_t> row_offsets, col_offsets;
  for (int i = 0; i < A.nblocks; ++i) {
    row_offsets.push_back((i+1) * c_size);
    col_offsets.push_back((i+1) * c_size);
  }
  for (int i = 0; i < A.nblocks-1; ++i) {
    row_offsets.push_back(c_size*(A.nblocks) + (i+1) * A.rank);
    col_offsets.push_back(c_size*(A.nblocks) + (i+1) * A.rank);
  }

  auto L_splits = L.split(row_offsets, col_offsets);
  auto last_splits = last.split(A.nblocks, A.nblocks);

  int64_t permuted_nblocks = A.nblocks * 2;

  for (int i = 0; i < A.nblocks; ++i) {
    for (int j = 0; j <= i; ++j) {
      if (!A.is_admissible(i, j)) {
        auto D_splits = A.D(i, j).split(std::vector<int64_t>(1, c_size),
                                        std::vector<int64_t>(1, c_size));

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
          L_splits[(i + A.nblocks) * permuted_nblocks + (j + A.nblocks)] = lower(last_splits[i * A.nblocks + j]);
        }
        else {
          L_splits[(i + A.nblocks) * permuted_nblocks + (j + A.nblocks)] = last_splits[i * A.nblocks + j];
        }
      }
      else {
        // Copy S blocks into the lower right corner
        L_splits[(i + A.nblocks) * permuted_nblocks + (j + A.nblocks)] = last_splits[i * A.nblocks + j];
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
  for (int i = 0; i < A.nblocks; ++i) {
    row_offsets.push_back((i+1) * c_size);
    col_offsets.push_back((i+1) * c_size);
  }
  for (int i = 0; i < A.nblocks-1; ++i) {
    row_offsets.push_back(c_size*(A.nblocks) + (i+1) * A.rank);
    col_offsets.push_back(c_size*(A.nblocks) + (i+1) * A.rank);
  }

  auto U_splits = U.split(row_offsets, col_offsets);
  auto last_splits = last.split(A.nblocks, A.nblocks);
  int64_t permuted_nblocks = A.nblocks * 2;

  for (int i = 0; i < A.nblocks; ++i) {
    for (int j = i; j < A.nblocks; ++j) {
      if (!A.is_admissible(i, j)) {
        auto D_splits = A.D(i, j).split(std::vector<int64_t>(1, c_size),
                                        std::vector<int64_t>(1, c_size));
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
          U_splits[(i + A.nblocks) * permuted_nblocks + (j + A.nblocks)] = upper(last_splits[i * A.nblocks + j]);
        }
        else {
          U_splits[(i + A.nblocks) * permuted_nblocks + (j + A.nblocks)] = last_splits[i * A.nblocks + j];
        }
      }
      else {
        // Copy S blocks
        U_splits[(i + A.nblocks) * permuted_nblocks + (j + A.nblocks)] = last_splits[i * A.nblocks + j];
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
  A.print_structure();
  Hatrix::BLR2 A_expected(A);
  double construct_error = A.construction_relative_error(randpts);
  auto last = A.factorize(randpts);

  multiply_compliments(A_expected);

  // Matrix A0 = generate_A0(A, last);
  // Matrix L1 = generate_L0(A);
  // Matrix U1 = generate_U0(A);
  // Matrix full_dense = matmul(matmul(L1, A0), U1);
  // L1.print();
  // U1.print();
  // // Matrix full_dense = matmul(L0, U0);

  // double A0_accuracy = check_A0_accuracy(A0, A_expected);
  // double factorize_error = factorization_accuracy(full_dense, A_expected);


  Matrix L_permuted = generate_L_permuted(A, last);
  Matrix U_permuted = generate_U_permuted(A, last);
  Matrix tt = matmul(L_permuted, U_permuted);
  Matrix ff = generate_full_permuted(A_expected);

  L_permuted.print();
  U_permuted.print();
  (tt - ff).print();

  std::cout << factorization_accuracy(tt, A_expected) << std::endl;;

  Hatrix::Context::finalize();

  // std::cout << "N: " << N << " rank: " << rank << " nblocks: " << nblocks << " admis: " <<  admis
  //           << " construct error: " << construct_error
  //           << " factorization error: " << factorize_error
  //           << " A0 error: " << A0_accuracy << "\n";
}
