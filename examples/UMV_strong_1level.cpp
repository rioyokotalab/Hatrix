#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Hatrix/Hatrix.h"

using randvec_t = std::vector<std::vector<double> >;

std::vector<double> equally_spaced_vector(int N, double minVal, double maxVal) {
  std::vector<double> res(N, 0.0);
  double rnge = maxVal - minVal;
  for(int i=0; i<N; i++) {
    res[i] = minVal + ((double)i/(double)rnge);
  }
  return res;
}

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
    int64_t N, nblocks, rank;

    // Generate a row slice without the diagonal block specified by 'block'. The
    // nrows parameter determines at what level the slice is generated at. Returns
    // a block of size (nrows x (N - nrows)).
    Matrix generate_row_slice(int block, int nrows, const randvec_t& randpts) {
      Matrix row_slice(nrows, N-nrows);
      int64_t ncols_left_slice = block * nrows;
      Matrix left_slice = generate_laplacend_matrix(randpts, nrows, ncols_left_slice,
                                                    block * nrows, 0);
      int64_t ncols_right_slice = N - (block+1) * nrows;
      Matrix right_slice = generate_laplacend_matrix(randpts, nrows, ncols_right_slice,
                                                     block * nrows, (block+1) * nrows);

      // concat left and right slices
      for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols_left_slice; ++j) {
          row_slice(i, j) = left_slice(i, j);
        }

        for (int j = 0; j < ncols_right_slice; ++j) {
          row_slice(i, j + ncols_left_slice) = right_slice(i, j);
        }
      }

      return row_slice;
    }

    // Generate a column slice without the diagonal block.
    Matrix generate_column_slice(int block, int ncols, const randvec_t& randpts) {
      Matrix col_slice(N-ncols, ncols);
      int nrows_upper_slice = block * ncols;
      Matrix upper_slice = generate_laplacend_matrix(randpts, nrows_upper_slice, ncols,
                                                     0, block * ncols);
      int nrows_lower_slice = N - (block + 1) * ncols;
      Matrix lower_slice = generate_laplacend_matrix(randpts, nrows_lower_slice, ncols,
                                                     (block+1) * ncols, block * ncols);

      for (int j = 0; j < col_slice.cols; ++j) {
        for (int i = 0; i < nrows_upper_slice; ++i) {
          col_slice(i, j) = upper_slice(i, j);
        }

        for (int i = 0; i < nrows_lower_slice; ++i) {
          col_slice(i + nrows_upper_slice, j) = lower_slice(i, j);
        }
      }

      return col_slice;
    }

    // Generate U for the leaf.
    Matrix generate_column_bases(int block, int leaf_size, const randvec_t& randpts) {
      // Row slice since column bases should be cutting across the columns.
      Matrix row_slice = generate_row_slice(block, leaf_size, randpts);
      Matrix Ui, Si, Vi; double error;
      std::tie(Ui, Si, Vi, error) = truncated_svd(row_slice, rank);

      return Ui;
    }

    // Generate V for the leaf.
    Matrix generate_row_bases(int block, int leaf_size, const randvec_t& randpts) {
      // Col slice since row bases should be cutting across the rows.
      Matrix col_slice = generate_column_slice(block, leaf_size, randpts);
      Matrix Ui, Si, Vi; double error;
      std::tie(Ui, Si, Vi, error) = truncated_svd(col_slice, rank);

      return transpose(Vi);
    }


  public:
    BLR2(const randvec_t& randpts, int64_t N, int64_t nblocks, int64_t rank) :
      N(N), nblocks(nblocks), rank(rank) {
      int leaf_size = N / nblocks;

      // generate diagonal and off-diagonal dense blocks
      for (int irow = 0; irow < nblocks; ++irow) {
        for (int icol = 0; icol < nblocks; ++icol) {
          if (std::abs(irow - icol) <= 1) {
            D.insert(irow, icol,
                     Hatrix::generate_laplacend_matrix(randpts, leaf_size, leaf_size,
                                                       irow * leaf_size, icol * leaf_size));
            is_admissible(irow, icol) = false;
          }
        }
      }

      Matrix Ubig = generate_column_bases(block, leaf_size, randpts);
      U.insert(block, std::move(Ubig));
      Matrix Vbig = generate_row_bases(block, leaf_size, randpts);
      V.insert(block, std::move(Vbig));


      for (int row = 0; row < nblocks; ++row) {
        int col = row % 2 == 0 ? row + 1 : row - 1;
        Matrix D = generate_laplacend_matrix(randpts, leaf_size, leaf_size,
                                             row * leaf_size, col * leaf_size);
        S.insert(row, col, matmul(matmul(U(row), D, true, false), V(col)));
      }

    }

    double construction_error(const randvec_t& randpts) {
      double error = 0;
      int slice = N / nblocks;

      for (int block = 0; block < nblocks; ++block) {
        double diagonal_error = rel_error(D(block, block),
                                          Hatrix::generate_laplacend_matrix(randpts, slice, slice,
                                                                            slice * block, slice * block));
        error += pow(diagonal_error, 2);
      }

      for (int row = 0; row < num_nodes; ++row) {
        int col = row % 2 == 0 ? row + 1 : row - 1;
        Matrix Ubig = get_Ubig(row, level);
        Matrix Vbig = get_Vbig(col, level);
        Matrix expected = matmul(matmul(Ubig, S(row, col)), Vbig, false, true);
        Matrix actual = Hatrix::generate_laplacend_matrix(randpts, slice, slice,
                                                          row * slice, col * slice);

        error += Hatrix::norm(expected - actual);
      }
      return std::sqrt(error / N / N);
    }
  };

}

int main(int argc, char** argv) {
  int64_t N = atoi(argv[1]);
  int64_t nblocks = atoi(argv[2]);
  int64_t rank = atoi(argv[3]);

  Hatrix::Context::init();
  randvec_t randpts;
  randpts.push_back(equally_spaced_vector(N, 0.0, 1.0 * (N/2))); // 1D
  randpts.push_back(equally_spaced_vector(N, 0.0, 1.0 * (N/2))); // 2D
  randpts.push_back(equally_spaced_vector(N, 0.0, 1.0 * (N/2))); // 3D

  if (N % nblocks != 0) {
    std::cout << "N % nblocks != 0. Aborting.\n";
    abort();
  }

  Hatrix::BLR2 A(randpts, N, nblocks, rank);
  double construct_error = A.construction_error(randpts);

  Hatrix::Context::finalize();

  std::cout << "N: " << N << " rank: " << rank << " nblocks: " << nblocks
            << " Solution error: " << construct_error << "\n";
}
