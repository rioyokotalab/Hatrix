#include <vector>
#include <iostream>
#include <cmath>

#include "Hatrix/Hatrix.h"

using randvec_t = std::vector<std::vector<double> >;

double rel_error(const double A_norm, const double B_norm) {
  double diff = A_norm - B_norm;
  return std::sqrt((diff * diff) / (B_norm * B_norm));
}

namespace Hatrix {
  class HSS {
    ColLevelMap U;
    RowLevelMap V;
    RowColLevelMap D, S;
    int N, rank, height;

    std::tuple<Matrix, Matrix> generate_column_bases(int block, int leaf_size, randvec_t& randvec) {
      Matrix row_slice(leaf_size, N - leaf_size);
      int ncols_left_slice = block * leaf_size;
      Matrix left_slice = generate_laplacend_matrix(randvec, leaf_size, ncols_left_slice,
                                                    block * leaf_size, 0);
      int ncols_right_slice = N - (block+1) * leaf_size;
      Matrix right_slice = generate_laplacend_matrix(randvec, leaf_size, ncols_right_slice,
                                                     block * leaf_size, (block+1) * leaf_size);

      // concat left and right slices
      for (int i = 0; i < leaf_size; ++i) {
        for (int j = 0; j < block * leaf_size; ++j) {
          row_slice(i, j) = left_slice(i, j);
        }

        for (int j = 0; j < ncols_right_slice; ++j) {
          row_slice(i, j + ncols_left_slice) = right_slice(i, j);
        }
      }


      Matrix U, S, V; double error;
      std::tie(U, S, V, error) = truncated_svd(row_slice, rank);

      return {U, Hatrix::matmul(S, V)};
    }

    std::tuple<Matrix, Matrix> generate_row_bases(int block, int leaf_size, randvec_t& randvec) {
      Matrix col_slice(N - leaf_size, leaf_size);
      int nrows_upper_slice = block * leaf_size;
      Matrix upper_slice = generate_laplacend_matrix(randvec, nrows_upper_slice, leaf_size,
                                                     0, block * leaf_size);
      int nrows_lower_slice = N - (block + 1) * leaf_size;
      Matrix lower_slice = generate_laplacend_matrix(randvec, nrows_lower_slice, leaf_size,
                                                     (block+1) * leaf_size, block * leaf_size);

      for (int j = 0; j < col_slice.cols; ++j) {
        for (int i = 0; i < nrows_upper_slice; ++i) {
          col_slice(i, j) = upper_slice(i, j);
        }

        for (int i = 0; i < nrows_lower_slice; ++i) {
          col_slice(i + nrows_upper_slice, j) = lower_slice(i, j);
        }
      }

      Matrix U, S, V; double error;
      Matrix col_slice_t = col_slice.transpose();
      std::tie(U, S, V, error) = truncated_svd(col_slice_t, rank);

      return {U, Hatrix::matmul(S, V)};
    }

    Matrix generate_coupling_matrix(randvec_t& randvec, int row, int col, int leaf_size, int level) {
      Matrix D = generate_laplacend_matrix(randvec, leaf_size, leaf_size,
                                           row * leaf_size, col * leaf_size);
      Matrix S = Hatrix::matmul(Hatrix::matmul(U(row, level), D, true), V(col, level));
      return S;
    }

    std::tuple<RowLevelMap, ColLevelMap> generate_leaf_nodes(randvec_t& randvec) {
      int nblocks = pow(height, 2);
      int leaf_size = N / nblocks;
      ColLevelMap Ugen;
      RowLevelMap Vgen;

      for (int block = 0; block < nblocks; ++block) {
        D.insert(block, block, height,
                 Hatrix::generate_laplacend_matrix(randvec, leaf_size, leaf_size,
                                                   block * leaf_size, block * leaf_size));
        Matrix U_temp, Ugen_temp;
        std::tie(U_temp, Ugen_temp) = generate_column_bases(block, leaf_size, randvec);
        U.insert(block, height, std::move(U_temp));
        Ugen.insert(block, height, std::move(Ugen_temp));

        Matrix V_temp, Vgen_temp;
        std::tie(V_temp, Vgen_temp) = generate_row_bases(block, leaf_size, randvec);
        V.insert(block, height, std::move(V_temp));
        Vgen.insert(block, height, std::move(Vgen_temp));
      }

      for (int block = 0; block < nblocks; ++block) {
        int s_col = block % 2 == 0 ? block + 1 : block - 1;
        S.insert(block, s_col, height, generate_coupling_matrix(randvec, block, s_col,
                                                                leaf_size, height));
      }

      return {Ugen, Vgen};
    }

    void generate_transfer_matrices(RowLevelMap& Ugen, ColLevelMap& Vgen) {
    }

  public:

    HSS(randvec_t& randpts, int _N, int _rank, int _height) :
      N(_N), rank(_rank), height(_height) {
      RowLevelMap Ugen; ColLevelMap Vgen;

      std::tie(Ugen, Vgen) = generate_leaf_nodes(randpts);
      generate_transfer_matrices(Ugen, Vgen);
    }

    double construction_relative_error(randvec_t& randvec) {
      int leaf_size = N / pow(height, 2);
      double error = 0;

      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
          Matrix A = generate_laplacend_matrix(randvec, leaf_size, leaf_size,
                                               i * leaf_size, j * leaf_size);
          if (i == j) {
            error += pow(rel_error(Hatrix::norm(A), Hatrix::norm(D(i, j, height)) ), 2);
          }
          else {
            Matrix Anew = Hatrix::matmul(Hatrix::matmul(U(i, height),
                                                        S(i, j, height)),
                                         V(j, height), false, true);
            error += pow(rel_error(Hatrix::norm(A),
                                   Hatrix::norm(Anew)), 2);
          }
        }
      }

      for (int i = 2; i < 4; ++i) {
        for (int j = 2; j < 4; ++j) {
          Matrix A = generate_laplacend_matrix(randvec, leaf_size, leaf_size,
                                               i * leaf_size, j * leaf_size);
          if (i == j) {
            error += pow(rel_error(Hatrix::norm(A), Hatrix::norm(D(i, j, height)) ), 2);
          }
          else {
            Matrix Anew = Hatrix::matmul(Hatrix::matmul(U(i, height),
                                                        S(i, j, height)),
                                         V(j, height), false, true);
            error += pow(rel_error(Hatrix::norm(A),
                                   Hatrix::norm(Anew)), 2);
          }
        }
      }

      return std::sqrt(error);
    }
  };
}

std::vector<double> equally_spaced_vector(int N, double minVal, double maxVal) {
  std::vector<double> res(N, 0.0);
  double rnge = maxVal - minVal;
  for(int i=0; i<N; i++) {
    res[i] = minVal + ((double)i/(double)rnge);
  }
  return res;
}

int main(int argc, char *argv[]) {
  int N = atoi(argv[1]);
  int rank = atoi(argv[2]);
  int height = 2;

  if (N % int(pow(height, 2)) != 0) {
    std::cout << N << " % " << pow(height, 2) << " != 0 \n";
    abort();
  }

  Hatrix::Context::init();
  randvec_t randvec;
  randvec.push_back(equally_spaced_vector(N, 0.0, 1.0)); // 1D

  Hatrix::HSS A(randvec, N, rank, height);
  double error = A.construction_relative_error(randvec);

  Hatrix::Context::finalize();

  std::cout << "construction error : " << error << std::endl;

}
