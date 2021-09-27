#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <chrono>
#include <fstream>

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
  double A_norm = Hatrix::norm(A);
  double B_norm = Hatrix::norm(B);
  double diff = A_norm - B_norm;
  return std::sqrt((diff * diff) / (B_norm * B_norm));
}

namespace Hatrix {
  class HSS {
  private:
    ColLevelMap U;
    RowLevelMap V;
    RowColLevelMap D, S;
    int N, rank, height;

    // Generate U for the leaf.
    Matrix generate_column_bases(int block, int leaf_size, const randvec_t& randvec) {
      // Row slice since column bases should be cutting across the columns.
      Matrix row_slice(leaf_size, N-leaf_size);
      int64_t ncols_left_slice = block * leaf_size;
      Matrix left_slice = generate_laplacend_matrix(randvec, leaf_size, ncols_left_slice,
                                                    block * leaf_size, 0);
      int64_t ncols_right_slice = N - (block+1) * leaf_size;
      Matrix right_slice = generate_laplacend_matrix(randvec, leaf_size, ncols_right_slice,
                                                     block * leaf_size, (block+1) * leaf_size);

      // concat left and right slices
      for (int i = 0; i < leaf_size; ++i) {
        for (int j = 0; j < ncols_left_slice; ++j) {
          row_slice(i, j) = left_slice(i, j);
        }

        for (int j = 0; j < ncols_right_slice; ++j) {
          row_slice(i, j + ncols_left_slice) = right_slice(i, j);
        }
      }

      Matrix Ui, Si, Vi; double error;
      std::tie(Ui, Si, Vi, error) = truncated_svd(row_slice, rank);

      return Ui;
    }

    // Generate V for the leaf.
    Matrix generate_row_bases(int block, int leaf_size, const randvec_t& randvec) {
      // Col slice since row bases should be cutting across the rows.
      Matrix col_slice(N-leaf_size, leaf_size);

      return bases;
    }

    void generate_leaf_bases(const randvec_t& randpts) {
      int nblocks = pow(2, height);
      int leaf_size = N / nblocks;

      for (int block = 0; block < nblocks; ++block) {
        D.insert(block, block, height,
                 Hatrix::generate_laplacend_matrix(randpts, leaf_size, leaf_size,
                                                   block * leaf_size, block * leaf_size));
        Matrix Ubig = generate_column_bases(block, leaf_size, randpts);
        U.insert(block, height, std::move(Ubig));
        Matrix Vbig = generate_row_bases(block, leaf_size, randpts);
        V.insert(block, height, std::move(Vbig));
      }
    }

  public:


    HSS(const randvec_t& randpts, int _N, int _rank, int _height) :
      N(_N), rank(_rank), height(_height) {
      generate_leaf_bases(randpts);

    }

    double construction_relative_error(const randvec_t& randvec) {

    }
  };
}

int main(int argc, char* argv[]) {
  int N = atoi(argv[1]);
  int rank = atoi(argv[2]);
  int height = atoi(argv[3]);

  if (N % int(pow(2, height)) != 0 && rank > int(N / pow(2, height))) {
    std::cout << N << " % " << pow(2, height) << " != 0 || rank > leaf(" << int(N / pow(2, height))  << ")\n";
    abort();
  }

  Hatrix::Context::init();
  randvec_t randvec;
  randvec.push_back(equally_spaced_vector(N, 0.0, 1.0)); // 1D

  Hatrix::HSS A(randvec, N, rank, height);
  double error = A.construction_relative_error(randvec);

  Hatrix::Context::finalize();
  std::cout << "N= " << N << " rank= " << rank << " height=" << height <<  " construction error=" << error << std::endl;
}
