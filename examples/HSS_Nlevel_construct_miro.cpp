#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <chrono>
#include <fstream>

#include "Hatrix/Hatrix.h"

constexpr double PV = 1e-3;
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
  class HSS {
  private:
    ColLevelMap U;
    RowLevelMap V;
    RowColLevelMap<Matrix> D, S;
    int64_t N, rank, height;

    // Generate a row slice without the diagonal block specified by 'block'. The
    // nrows parameter determines at what level the slice is generated at. Returns
    // a block of size (nrows x (N - nrows)).
    Matrix generate_row_slice(int block, int nrows, const randvec_t& randpts) {
      Matrix row_slice(nrows, N-nrows);
      int64_t ncols_left_slice = block * nrows;
      Matrix left_slice = generate_laplacend_matrix(randpts, nrows, ncols_left_slice,
                                                    block * nrows, 0, PV);
      int64_t ncols_right_slice = N - (block+1) * nrows;
      Matrix right_slice = generate_laplacend_matrix(randpts, nrows, ncols_right_slice,
                                                     block * nrows, (block+1) * nrows, PV);

      // concat left and right slices
      for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols_left_slice; ++j) {
          row_slice(i, j) = left_slice(i, j);
        }

        for (int j = 0; j < ncols_right_slice; ++j) {
          row_slice(i, j + ncols_left_slice) = right_slice(i, j);
        }
      }

      row_slice.print();
      return row_slice;
    }

    // Generate a column slice without the diagonal block.
    Matrix generate_column_slice(int block, int ncols, const randvec_t& randpts) {
      Matrix col_slice(N-ncols, ncols);
      int nrows_upper_slice = block * ncols;
      Matrix upper_slice = generate_laplacend_matrix(randpts, nrows_upper_slice, ncols,
                                                     0, block * ncols, PV);
      int nrows_lower_slice = N - (block + 1) * ncols;
      Matrix lower_slice = generate_laplacend_matrix(randpts, nrows_lower_slice, ncols,
                                                     (block+1) * ncols, block * ncols, PV);

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

    void generate_leaf_bases(const randvec_t& randpts) {
      int nblocks = pow(2, height);
      int leaf_size = N / nblocks;

      for (int block = 0; block < nblocks; ++block) {
        D.insert(block, block, height,
                 Hatrix::generate_laplacend_matrix(randpts, leaf_size, leaf_size,
                                                   block * leaf_size, block * leaf_size, PV));
        Matrix Ubig = generate_column_bases(block, leaf_size, randpts);
        U.insert(block, height, std::move(Ubig));
        Matrix Vbig = generate_row_bases(block, leaf_size, randpts);
        V.insert(block, height, std::move(Vbig));
      }

      for (int row = 0; row < nblocks; ++row) {
        int col = row % 2 == 0 ? row + 1 : row - 1;
        Matrix D = generate_laplacend_matrix(randpts, leaf_size, leaf_size,
                                             row * leaf_size, col * leaf_size, PV);
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
      // Generate the actual bases for the upper level and pass it to this
      // function again for generating transfer matrices at successive levels.
      RowLevelMap Ubig_parent;
      ColLevelMap Vbig_parent;

      for (int node = 0; node < num_nodes; ++node) {
        int child1 = node * 2;
        int child2 = node * 2 + 1;

        {
          // Generate U transfer matrix.
          Matrix& Ubig_child1 = Uchild(child1, child_level);
          Matrix& Ubig_child2 = Uchild(child2, child_level);


          std::cout << "U transfer generate. n-> " << node << " l-> " << level << std::endl;
          Matrix Alevel_node_plus = generate_row_slice(node, leaf_size, randpts);
          std::vector<Matrix> Alevel_node_plus_splits = Alevel_node_plus.split(2, 1);

          Matrix temp(Ubig_child1.cols + Ubig_child2.cols, Alevel_node_plus.cols);
          std::vector<Matrix> temp_splits = temp.split(2, 1);

          matmul(Ubig_child1, Alevel_node_plus_splits[0], temp_splits[0], true, false, 1, 0);
          matmul(Ubig_child2, Alevel_node_plus_splits[1], temp_splits[1], true, false, 1, 0);

          Matrix Utransfer, Si, Vi; double error;
          std::tie(Utransfer, Si, Vi, error) = truncated_svd(temp, rank);
          U.insert(node, level, std::move(Utransfer));

          // Generate the full bases for passing onto the upper level.
          std::vector<Matrix> Utransfer_splits = U(node, level).split(2, 1);
          Matrix Ubig(leaf_size, rank);
          std::vector<Matrix> Ubig_splits = Ubig.split(2, 1);
          matmul(Ubig_child1, Utransfer_splits[0], Ubig_splits[0]);
          matmul(Ubig_child2, Utransfer_splits[1], Ubig_splits[1]);

          auto aa = get_Ubig(node, level);
          matmul(aa, aa, true, false).print();
          // Save the actual basis into the temporary Map to pass to generate
          // the S block and pass it to higher levels.
          Ubig_parent.insert(node, level, std::move(Ubig));
        }

        {
          // Generate V transfer matrix.
          Matrix& Vbig_child1 = Vchild(child1, child_level);
          Matrix& Vbig_child2 = Vchild(child2, child_level);

          Matrix Alevel_plus_node = generate_column_slice(node, leaf_size, randpts);
          std::vector<Matrix> Alevel_plus_node_splits = Alevel_plus_node.split(1, 2);

          Matrix temp(Alevel_plus_node.rows, Vbig_child1.cols + Vbig_child2.cols);
          std::vector<Matrix> temp_splits = temp.split(1, 2);

          matmul(Alevel_plus_node_splits[0], Vbig_child1, temp_splits[0]);
          matmul(Alevel_plus_node_splits[1], Vbig_child2, temp_splits[1]);

          Matrix Ui, Si, Vtransfer; double error;
          std::tie(Ui, Si, Vtransfer, error) = truncated_svd(temp, rank);
          V.insert(node, level, transpose(Vtransfer));

          // Generate the full bases for passing onto the upper level.
          std::vector<Matrix> Vtransfer_splits = V(node, level).split(2, 1);
          Matrix Vbig(rank, leaf_size);
          std::vector<Matrix> Vbig_splits = Vbig.split(1, 2);

          matmul(Vtransfer_splits[0], Vbig_child1, Vbig_splits[0], true, true, 1, 0);
          matmul(Vtransfer_splits[1], Vbig_child2, Vbig_splits[1], true, true, 1, 0);

          Vbig_parent.insert(node, level, transpose(Vbig));
        }
      }

      for (int row = 0; row < num_nodes; ++row) {
        int col = row % 2 == 0 ? row + 1 : row - 1;
        Matrix D = generate_laplacend_matrix(randpts, leaf_size, leaf_size,
                                             row * leaf_size, col * leaf_size, PV);
        S.insert(row, col, level, matmul(matmul(Ubig_parent(row, level), D, true, false),
                                         Vbig_parent(col, level)));
      }

      return {Ubig_parent, Vbig_parent};
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
      double dense = 0;
      int num_nodes = pow(2, height);

      for (int block = 0; block < num_nodes; ++block) {
        int slice = N / num_nodes;
        Matrix actual = Hatrix::generate_laplacend_matrix(randpts, slice, slice,
                                                          slice * block, slice * block, PV);
        error += pow(norm(D(block, block, height) - actual), 2);
        dense += pow(norm(actual), 2);
      }

      for (int level = height; level > 0; --level) {
        int num_nodes = pow(2, level);
        int slice = N / num_nodes;

        for (int row = 0; row < num_nodes; ++row) {
          int col = row % 2 == 0 ? row + 1 : row - 1;
          Matrix Ubig = get_Ubig(row, level);
          Matrix Vbig = get_Vbig(col, level);
          Matrix expected = matmul(matmul(Ubig, S(row, col, level)), Vbig, false, true);
          Matrix actual = Hatrix::generate_laplacend_matrix(randpts, slice, slice,
                                                            row * slice, col * slice, PV);

          error += pow(Hatrix::norm(expected - actual), 2);
          dense += pow(Hatrix::norm(actual), 2);
        }
      }
      return std::sqrt(error / dense);
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
  randpts.push_back(equally_spaced_vector(N, 0.0, 1.0 * N)); // 1D
  randpts.push_back(equally_spaced_vector(N, 0.0, 1.0 * N)); // 2D
  randpts.push_back(equally_spaced_vector(N, 0.0, 1.0 * N)); // 3D

  Hatrix::HSS A(randpts, N, rank, height);
  double error = A.construction_relative_error(randpts);

  Hatrix::Context::finalize();
  std::cout << "N= " << N << " rank= " << rank << " leaf=" << N / pow(2, height)
            << " height=" << height <<  " construction error=" << error << std::endl;
}
