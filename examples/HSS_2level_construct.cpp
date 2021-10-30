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
    RowColLevelMap<Matrix> D, S;
    int64_t N, rank, height;

    std::tuple<Matrix, Matrix> generate_column_bases(int block, int leaf_size, const randvec_t& randvec) {
      Matrix row_slice(leaf_size, N - leaf_size);
      int64_t ncols_left_slice = block * leaf_size;
      Matrix left_slice = generate_laplacend_matrix(randvec, leaf_size, ncols_left_slice,
                                                    block * leaf_size, 0);
      int64_t ncols_right_slice = N - (block+1) * leaf_size;
      Matrix right_slice = generate_laplacend_matrix(randvec, leaf_size, ncols_right_slice,
                                                     block * leaf_size, (block+1) * leaf_size);
      std::vector<Matrix> row_slice_parts = row_slice.split(std::vector<int64_t>(1, 0),
                                                            std::vector<int64_t>(1, ncols_left_slice));

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

      return {Ui, Hatrix::matmul(Si, Vi)};
    }

    std::tuple<Matrix, Matrix> generate_row_bases(int block, int leaf_size, const randvec_t& randvec) {
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

      Matrix Ui, Si, Vi; double error;
      Matrix col_slice_t = transpose(col_slice);
      std::tie(Ui, Si, Vi, error) = truncated_svd(col_slice_t, rank);

      return {Ui, Hatrix::matmul(Si, Vi)};
    }

    Matrix generate_coupling_matrix(const randvec_t& randvec, int row, int col, int leaf_size, int level) {
      Matrix D = generate_laplacend_matrix(randvec, leaf_size, leaf_size,
                                           row * leaf_size, col * leaf_size);
      Matrix S = Hatrix::matmul(Hatrix::matmul(U(row, level), D, true), V(col, level));
      return S;
    }

    std::tuple<RowLevelMap, ColLevelMap> generate_leaf_nodes(const randvec_t& randvec) {
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

    Matrix generate_U_actual_bases(int p) {
      int child1 = p * 2;
      int child2 = p * 2 + 1;
      int leaf_size = int(N / 2);
      Matrix Ubig(rank, leaf_size);

      std::vector<Matrix> Ubig_splits = Ubig.split(1, 2);
      std::vector<Matrix> U_splits = U(p, height-1).split(2, 1);

      matmul(U_splits[0], U(child1, height), Ubig_splits[0], true, true);
      matmul(U_splits[1], U(child2, height), Ubig_splits[1], true, true);

      return transpose(Ubig);
    }

    Matrix generate_V_actual_bases(int p) {
      int child1 = p * 2;
      int child2 = p * 2 + 1;
      int leaf_size = int(N / 2);
      Matrix Vbig(leaf_size, rank);

      std::vector<Matrix> Vbig_splits = Vbig.split(2, 1);
      std::vector<Matrix> V_splits = V(p, height - 1).split(2, 1);

      matmul(V(child1, height), V_splits[0], Vbig_splits[0]);
      matmul(V(child2, height), V_splits[1], Vbig_splits[1]);

      return Vbig;
    }

    Matrix generate_non_leaf_coupling_matrix(const randvec_t& randvec, int row, int col, int leaf_size,
                                             Matrix& Ubig, Matrix& Vbig) {
      Matrix D = generate_laplacend_matrix(randvec, leaf_size, leaf_size,
                                           row * leaf_size, col * leaf_size);
      return matmul(matmul(Ubig, D, true, false), Vbig);
    }

    void generate_transfer_matrices(const randvec_t& randvec, RowLevelMap& Ugen, ColLevelMap& Vgen) {
      Matrix Ui, Si, Vi; double error;

      for (int p = 0; p < 2; ++p) {
        int child1 = p * 2;
        int child2 = p * 2 + 1;

        // Generate U transfer matrix.
        Matrix& Ugen_upper = Ugen(child1, height);
        Matrix& Ugen_lower = Ugen(child2, height);
        Matrix Ugen_concat(Ugen_upper.rows + Ugen_lower.rows, Ugen_upper.cols);
        std::vector<Matrix> Ugen_slices = Ugen_concat.split(2, 1);
        Ugen_slices[0] = Ugen_upper;
        Ugen_slices[1] = Ugen_lower;

        std::tie(Ui, Si, Vi, error) = truncated_svd(Ugen_concat, rank);
        U.insert(p, height-1, std::move(Ui));

        // Generate V transfer matrix.
        Matrix& Vgen_upper = Vgen(child1, height);
        Matrix& Vgen_lower = Vgen(child2, height);
        Matrix Vgen_concat(Vgen_upper.rows + Vgen_lower.rows, Vgen_upper.cols);
        std::vector<Matrix> Vgen_slices = Vgen_concat.split(2, 1);
        Vgen_slices[0] = Vgen_upper;
        Vgen_slices[1] = Vgen_lower;

        std::tie(Ui, Si, Vi, error) = truncated_svd(Vgen_concat, rank);
        V.insert(p, height-1, std::move(Ui));
      }

      for (int row = 0; row < 2; ++row) {
        int col = row % 2 == 0 ? row + 1 : row - 1;
        int leaf_size = int(N / 2);

        Matrix Ubig = generate_U_actual_bases(row);
        Matrix Vbig = generate_V_actual_bases(col);

        S.insert(row, col, height - 1,
                 generate_non_leaf_coupling_matrix(randvec, row, col, leaf_size,
                                                   Ubig, Vbig));
      }
    }

  public:

    HSS(const randvec_t& randpts, int _N, int _rank, int _height) :
      N(_N), rank(_rank), height(_height) {
      RowLevelMap Ugen; ColLevelMap Vgen;

      std::tie(Ugen, Vgen) = generate_leaf_nodes(randpts);
      generate_transfer_matrices(randpts, Ugen, Vgen);
    }

    double construction_relative_error(const randvec_t& randvec) {
      int leaf_size = N / pow(height, 2);
      double error = 0;

      // Check leaf level blocks.
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

      // Check off-diagonal non-leaf blocks

      // Upper right
      int row = 0, col = 1;
      leaf_size = N / 2;

      Matrix A = generate_laplacend_matrix(randvec, leaf_size, leaf_size,
                                           row * leaf_size, col * leaf_size);
      Matrix Ubig = generate_U_actual_bases(row);
      Matrix Vbig = generate_V_actual_bases(col);
      Matrix Anew = matmul(matmul(Ubig, S(row, col, height-1)), Vbig, false, true);
      error += pow(rel_error(norm(A), norm(Anew)), 2);

      // Lower left
      row = 1, col = 0;
      A = generate_laplacend_matrix(randvec, leaf_size, leaf_size,
                                           row * leaf_size, col * leaf_size);
      Ubig = generate_U_actual_bases(row);
      Vbig = generate_V_actual_bases(col);
      Anew = matmul(matmul(Ubig, S(row, col, height-1)), Vbig, false, true);
      error += pow(rel_error(norm(A), norm(Anew)), 2);

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

  if (N % int(pow(height, 2)) != 0 || rank > int(N / pow(height, 2))) {
    std::cout << N << " % " << pow(height, 2) << " != 0 || rank > leaf(" << int(N / pow(height, 2))  << ")\n";
    abort();
  }

  Hatrix::Context::init();
  randvec_t randvec;
  randvec.push_back(equally_spaced_vector(N, 0.0, 1.0)); // 1D

  Hatrix::HSS A(randvec, N, rank, height);
  double error = A.construction_relative_error(randvec);

  Hatrix::Context::finalize();

  std::cout << "N=" << N << " rank=" << rank << " construction error : " << error << std::endl;

}
