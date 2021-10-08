#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <chrono>
#include <fstream>

#include "Hatrix/Hatrix.h"

// Implements the HSS construction algorithm from
// Chandrasekaran's 2006 paper.

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
  public:
    ColLevelMap U;
    RowLevelMap V;
    RowColLevelMap<Matrix> D, S;
    int N, rank, height;

  private:

    Matrix generate_column_bases(int block, int leaf_size, const randvec_t& randvec) {
      Matrix row_slice(leaf_size, N - leaf_size);
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

    Matrix generate_row_bases(int block, int leaf_size, const randvec_t& randvec) {
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

      return Ui;
    }

    void generate_leaf_nodes(const randvec_t& randvec) {
      int nblocks = pow(2, height);
      int leaf_size = N / nblocks;

      for (int block = 0; block < nblocks; ++block) {
        // Diagonal offset is used since the last block can have a different shape from
        // the rest.
        D.insert(block, block, height,
                 Hatrix::generate_laplacend_matrix(randvec, leaf_size, leaf_size,
                                                   block * leaf_size, block * leaf_size));

        Matrix U_temp = generate_column_bases(block,
                                              leaf_size,
                                              randvec);
        U.insert(block, height, std::move(U_temp));

        Matrix V_temp = generate_row_bases(block,
                                           leaf_size,
                                           randvec);
        V.insert(block, height, std::move(V_temp));
      }

      for (int row = 0; row < nblocks; ++row) {
        int col = row % 2 == 0 ? row + 1 : row - 1;
        Matrix D = generate_laplacend_matrix(randvec, leaf_size, leaf_size,
                                             row * leaf_size, col * leaf_size);
        S.insert(row, col, height, matmul(matmul(U(row, height), D, true, false), V(col, height)));
      }
    }

    Matrix get_Ubig(int p, int level) {
      if (level == height) {
        return U(p, level);
      }
      int child1 = p * 2;
      int child2 = p * 2 + 1;
      int num_nodes = pow(2, level);

      // int rank = leaf_size;

      Matrix Ubig_child1 = get_Ubig(child1, level+1);
      Matrix Ubig_child2 = get_Ubig(child2, level+1);

      int leaf_size = Ubig_child1.rows + Ubig_child2.rows;

      Matrix Ubig(leaf_size, rank);

      std::vector<Matrix> Ubig_splits =
        Ubig.split(
                   std::vector<int64_t>(1,
                                        Ubig_child1.rows), {});

      std::vector<Matrix> U_splits = U(p, level).split(2, 1);

      matmul(Ubig_child1, U_splits[0], Ubig_splits[0]);
      matmul(Ubig_child2, U_splits[1], Ubig_splits[1]);

      return Ubig;
    }

    Matrix get_Vbig(int p, int level) {
      if (level == height) {
        return V(p, level);
      }
      int child1 = p * 2;
      int child2 = p * 2 + 1;
      int num_nodes = pow(2, level);

      Matrix Vbig_child1 = get_Vbig(child1, level+1);
      Matrix Vbig_child2 = get_Vbig(child2, level+1);

      int leaf_size = Vbig_child1.rows + Vbig_child2.rows;
      //int rank = leaf_size;

      Matrix Vbig(leaf_size, rank);

      std::vector<Matrix> Vbig_splits = Vbig.split(std::vector<int64_t>(1, Vbig_child1.rows), {});
      std::vector<Matrix> V_splits = V(p, level).split(2, 1);

      matmul(Vbig_child1, V_splits[0], Vbig_splits[0]);
      matmul(Vbig_child2, V_splits[1], Vbig_splits[1]);

      return Vbig;
    }

    void generate_transfer_matrices(const randvec_t& randvec, const int level) {
      int num_nodes = pow(2, level);
      int leaf_size = N / num_nodes;
      int child_level = level + 1;
      int c_num_nodes = pow(2, child_level);
      int child_leaf_size = N / c_num_nodes;

      std::vector<Matrix> Ublocks, Vblocks;

      for (int p = 0; p < num_nodes; ++p) {
        int child1 = p * 2;
        int child2 = p * 2 + 1;

        // Generate U transfer matrix.
        Matrix U_child1 = generate_column_bases(child1,
                                                child_leaf_size,
                                                randvec);// get_Ubig(child1, child_level);
        Matrix U_child2 = generate_column_bases(child2,
                                                child_leaf_size,
                                                randvec);
        Matrix Ubig = generate_column_bases(p, leaf_size, randvec);
        std::vector<Matrix> Ubig_splits = Ubig.split(std::vector<int64_t>(1, U_child1.rows), {});

        Matrix Utransfer(rank * 2, rank);
        std::vector<Matrix> Utransfer_splits = Utransfer.split(2, 1);

        matmul(U_child1, Ubig_splits[0], Utransfer_splits[0], true, false, 1.0, 0.0);
        matmul(U_child2, Ubig_splits[1], Utransfer_splits[1], true, false, 1.0, 0.0);

        U.insert(p, level, std::move(Utransfer));
        Ublocks.push_back(Ubig);

        // Generate V transfer matrix.
        Matrix Vbig_child1 = generate_row_bases(child1,
                                                child_leaf_size,
                                                randvec);// get_Ubig(child1, child_level);
        Matrix Vbig_child2 = generate_row_bases(child2,
                                                child_leaf_size,
                                                randvec);// get_Ubig(child1, child_level);
        Matrix Vbig  = generate_row_bases(p, leaf_size, randvec);
        std::vector<Matrix> Vbig_splits = Vbig.split(std::vector<int64_t>(1, Vbig_child1.rows), {});
        Matrix Vtransfer(rank * 2, rank);
        std::vector<Matrix> Vtransfer_splits = Vtransfer.split(2, 1);

        matmul(Vbig_child1, Vbig_splits[0], Vtransfer_splits[0], true, false, 1.0, 0.0);
        matmul(Vbig_child2, Vbig_splits[1], Vtransfer_splits[1], true, false, 1.0, 0.0);

        V.insert(p, level, std::move(Vtransfer));
        Vblocks.push_back(Vbig);
      }

      for (int row = 0; row < num_nodes; ++row) {
        int col = row % 2 == 0 ? row + 1 : row - 1;
        Matrix& Ubig = Ublocks[row];
        Matrix& Vbig = Vblocks[col];

        Matrix D = generate_laplacend_matrix(randvec, leaf_size, leaf_size,
                                             row * leaf_size, col * leaf_size);
        S.insert(row, col, level, matmul(matmul(Ubig, D, true, false), Vbig));
      }
    }

  public:

    HSS(const randvec_t& randpts, int _N, int _rank, int _height) :
      N(_N), rank(_rank), height(_height) {
      generate_leaf_nodes(randpts);

      for (int level = height-1; level > 0; --level) {
        generate_transfer_matrices(randpts, level);
      }
    }

    double construction_relative_error(const randvec_t& randvec) {
      // verify diagonal matrix block constructions at the leaf level.
      double error = 0;
      int num_nodes = pow(2, height);
      for (int block = 0; block < num_nodes; ++block) {
        int slice = N / num_nodes;
        error += Hatrix::norm(D(block, block, height) - Hatrix::generate_laplacend_matrix(randvec, slice, slice,
                                                                                          slice * block, slice * block));
      }

      // regenerate off-diagonal blocks and test for correctness.
      for (int level = height; level > 0; --level) {
        int num_nodes = pow(2, level);
        int slice = N / num_nodes;

        for (int row = 0; row < num_nodes; ++row) {
          int col = row % 2 == 0 ? row + 1 : row - 1;
          Matrix Ubig = get_Ubig(row, level);
          Matrix Vbig = get_Vbig(col, level);
          Matrix expected = matmul(matmul(Ubig, S(row, col, level)), Vbig, false, true);
          Matrix actual = Hatrix::generate_laplacend_matrix(randvec, slice, slice,
                                                            row * slice, col * slice);

          error += Hatrix::norm(expected - actual);
        }
      }

      return std::sqrt(error / N / N);
    }
  };
} // namespace Hatrix

int main(int argc, char *argv[]) {
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

  auto start_construct = std::chrono::system_clock::now();
  Hatrix::HSS A(randvec, N, rank, height);
  auto stop_construct = std::chrono::system_clock::now();

  double error = A.construction_relative_error(randvec);

  Hatrix::Context::finalize();

  std::ofstream file;
  file.open("output.txt", std::ios::app | std::ios::out);
  std::cout << "N= " << N << " rank= " << rank << " height=" << height <<  " const. error=" << error << std::endl;
  file.close();

}
