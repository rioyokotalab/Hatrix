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
      Matrix col_slice(N-leaf_size, leaf_size);
      int nrows_upper_slice = block * leaf_size;
      Matrix upper_slice = generate_laplacend_matrix(randpts, nrows_upper_slice, leaf_size,
                                                     0, block * leaf_size);
      int nrows_lower_slice = N - (block + 1) * leaf_size;
      Matrix lower_slice = generate_laplacend_matrix(randpts, nrows_lower_slice, leaf_size,
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
      std::tie(Ui, Si, Vi, error) = truncated_svd(col_slice, rank);

      return transpose(Vi);
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

      for (int row = 0; row < nblocks; ++row) {
        int col = row % 2 == 0 ? row + 1 : row - 1;
        Matrix D = generate_laplacend_matrix(randpts, leaf_size, leaf_size,
                                             row * leaf_size, col * leaf_size);
        S.insert(row, col, height, matmul(matmul(U(row, height), D, true, false), V(col, height)));
      }
    }

    std::tuple<RowLevelMap, ColLevelMap> generate_transfer_matrices(const randvec_t& randpts,
                                                                    int level, RowLevelMap& Uchild,
                                                                    ColLevelMap& Vchild) {
      int num_nodes = pow(2, level);
      int leaf_size = N / num_nodes;
      int child_level = level + 1;
      int c_num_nodes = pow(2, child_level);
      RowLevelMap Uparent;
      ColLevelMap Vparent;

      for (int p = 0; p < num_nodes; ++p) {
        int child1 = p * 2;
        int child2 = p * 2 + 1;

        Matrix& Ubig_child1 = Uchild(child1, child_level);
        Matrix& Ubig_child2 = Uchild(child2, child_level);
        Matrix Utransfer(rank*2, rank);
        std::vector<Matrix> Utransfer_splits = Utransfer.split(2, 1);

        Matrix Alevel_p_plus = generate_row_slice(p, leaf_size, randpts);
        std::vector<Matrix> Alevel_p_plus_splits = Alevel_p_plus.split(2, 1);
      }

      return {Uchild, Vchild};
    }

    Matrix get_Ubig(int node, int level) {
      if (level == height) {
        return U(node, level);
      }
      int child1 = node * 2;
      int child2 = node * 2 + 1;

      // int rank = leaf_size;

      Matrix Ubig_child1 = get_Ubig(child1, level+1);
      Matrix Ubig_child2 = get_Ubig(child2, level+1);

      int leaf_size = Ubig_child1.rows + Ubig_child2.rows;

      Matrix Ubig(leaf_size, rank);

      std::vector<Matrix> Ubig_splits =
        Ubig.split(
                   std::vector<int64_t>(1,
                                        Ubig_child1.rows), {});

      std::vector<Matrix> U_splits = U(node, level).split(2, 1);

      matmul(Ubig_child1, U_splits[0], Ubig_splits[0]);
      matmul(Ubig_child2, U_splits[1], Ubig_splits[1]);

      return Ubig;
    }

    Matrix get_Vbig(int node, int level) {
      if (level == height) {
        return V(node, level);
      }
      int child1 = node * 2;
      int child2 = node * 2 + 1;

      Matrix Vbig_child1 = get_Vbig(child1, level+1);
      Matrix Vbig_child2 = get_Vbig(child2, level+1);

      int leaf_size = Vbig_child1.rows + Vbig_child2.rows;
      //int rank = leaf_size;

      Matrix Vbig(leaf_size, rank);

      std::vector<Matrix> Vbig_splits = Vbig.split(std::vector<int64_t>(1, Vbig_child1.rows), {});
      std::vector<Matrix> V_splits = V(node, level).split(2, 1);

      matmul(Vbig_child1, V_splits[0], Vbig_splits[0]);
      matmul(Vbig_child2, V_splits[1], Vbig_splits[1]);

      return Vbig;
    }

  public:


    HSS(const randvec_t& randpts, int _N, int _rank, int _height) :
      N(_N), rank(_rank), height(_height) {
      RowLevelMap Uchild;
      ColLevelMap Vchild;

      generate_leaf_bases(randpts);

      Uchild = U;
      Vchild = V;

      for (int level = height-1; level > 0; --level) {
        std::tie(Uchild, Vchild) = generate_transfer_matrices(randpts, level, Uchild, Vchild);
      }
    }

    double construction_relative_error(const randvec_t& randpts) {
      double error = 0;
      int num_nodes = pow(2, height);

      for (int block = 0; block < num_nodes; ++block) {
        int slice = N / num_nodes;
        double diagonal_error = rel_error(D(block, block, height),
                                          Hatrix::generate_laplacend_matrix(randpts, slice, slice,
                                                                            slice * block, slice * block));
        error += pow(diagonal_error, 2);
      }

      for (int level = height; level > height-1; --level) {
        int num_nodes = pow(2, level);
        int slice = N / num_nodes;

        for (int row = 0; row < num_nodes; ++row) {
          int col = row % 2 == 0 ? row + 1 : row - 1;
          Matrix Ubig = get_Ubig(row, level);
          Matrix Vbig = get_Vbig(col, level);
          Matrix expected = matmul(matmul(Ubig, S(row, col, level)), Vbig, false, true);
          Matrix actual = Hatrix::generate_laplacend_matrix(randpts, slice, slice,
                                                            row * slice, col * slice);
          double offD_error = rel_error(expected, actual);
          error += pow(offD_error, 2);
        }
      }
      return std::sqrt(error);
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
  randvec_t randpts;
  randpts.push_back(equally_spaced_vector(N, 0.0, 1.0)); // 1D

  Hatrix::HSS A(randpts, N, rank, height);
  double error = A.construction_relative_error(randpts);

  Hatrix::Context::finalize();
  std::cout << "N= " << N << " rank= " << rank << " height=" << height <<  " construction error=" << error << std::endl;
}
