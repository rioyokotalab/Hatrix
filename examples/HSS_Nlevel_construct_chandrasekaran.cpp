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
  public:
    ColLevelMap U;
    RowLevelMap V;
    RowColLevelMap D, S;
    int N, rank, height;

  private:

    Matrix generate_column_bases(int block,
                                 int leaf_size,
                                 int diagonal_offset,
                                 int slice,
                                 int level,
                                 const randvec_t& randvec) {
      int num_nodes = pow(2, level);
      Matrix row_slice(leaf_size, N - leaf_size);
      int64_t ncols_left_slice = block * slice;
      Matrix left_slice = generate_laplacend_matrix(randvec, leaf_size, ncols_left_slice,
                                                    diagonal_offset, 0);

      int64_t ncols_right_slice = block == num_nodes - 1 ? 0 : N - (block+1) * slice;
      Matrix right_slice = generate_laplacend_matrix(randvec, leaf_size, ncols_right_slice,
                                                     diagonal_offset, (block+1) * slice);

      std::vector<Matrix> row_slice_parts = row_slice.split({},
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

      return Ui;
    }

    Matrix generate_row_bases(int block,
                              int leaf_size,
                              int diagonal_offset,
                              int slice,
                              int level,
                              const randvec_t& randvec) {
      int num_nodes = pow(2, level);
      Matrix col_slice(N - leaf_size, leaf_size);
      int nrows_upper_slice = block * slice;
      Matrix upper_slice = generate_laplacend_matrix(randvec, nrows_upper_slice, leaf_size,
                                                     0, diagonal_offset);

      int nrows_lower_slice = block == num_nodes - 1 ? 0 : N - (block + 1) * slice;
      Matrix lower_slice = generate_laplacend_matrix(randvec, nrows_lower_slice, leaf_size,
                                                     (block+1) * slice, diagonal_offset);

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

    Matrix generate_coupling_matrix(const randvec_t& randvec, const int row, const int col,
                                    const int level) {

      Matrix Ubig = get_Ubig(row, level);
      Matrix Vbig = get_Vbig(col, level);
      int block_nrows = Ubig.rows;
      int block_ncols = Vbig.rows;
      int slice = N / int(pow(2, level));
      Matrix D = generate_laplacend_matrix(randvec, block_nrows, block_ncols,
                                           row * slice, col * slice);
      return matmul(matmul(Ubig, D, true, false), Vbig);
    }


    void generate_leaf_nodes(const randvec_t& randvec) {
      int nblocks = pow(2, height);

      for (int block = 0; block < nblocks; ++block) {
        int slice = N / nblocks;
        int leaf_size = (block == (nblocks-1)) ? (N - (slice * block)) :  slice;

        // Diagonal offset is used since the last block can have a different shape from
        // the rest.
        int diagonal_offset = slice * block;
        D.insert(block, block, height,
                 Hatrix::generate_laplacend_matrix(randvec, leaf_size, leaf_size,
                                                   diagonal_offset, diagonal_offset));

        Matrix U_temp = generate_column_bases(block,
                                               leaf_size,
                                               diagonal_offset,
                                               slice,
                                               height,
                                               randvec);
        U.insert(block, height, std::move(U_temp));

        Matrix V_temp = generate_row_bases(block,
                                           leaf_size,
                                           diagonal_offset,
                                           slice,
                                           height,
                                           randvec);
        V.insert(block, height, std::move(V_temp));
      }

      for (int row = 0; row < nblocks; ++row) {
        int col = row % 2 == 0 ? row + 1 : row - 1;
        S.insert(row, col, height, generate_coupling_matrix(randvec, row, col, height));
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

      Matrix Ubig(rank, leaf_size);

      std::vector<Matrix> Ubig_splits = Ubig.split({}, std::vector<int64_t>(1, Ubig_child1.rows));
      std::vector<Matrix> U_splits = U(p, level).split(2, 1);

      matmul(U_splits[0], Ubig_child1, Ubig_splits[0], true, true);
      matmul(U_splits[1], Ubig_child2, Ubig_splits[1], true, true);

      return transpose(Ubig);
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
      int slice = N / num_nodes;

      std::vector<Matrix> Ublocks, Vblocks;

      for (int p = 0; p < num_nodes; ++p) {
        int child1 = p * 2;
        int child2 = p * 2 + 1;
        int child_level = level + 1;
        int leaf_size = (p == (num_nodes-1)) ? (N - (slice * p)) :  slice;
        int diagonal_offset = slice * p;

        // Generate U transfer matrix.
        int c_num_nodes = pow(2, child_level);
        int c1_slice = N / c_num_nodes;
        int c1_leaf_size = (child1 == (c_num_nodes-1)) ? (N - (c1_slice * child1)) :  c1_slice;
        int c1_diagonal_offset = c1_slice * child1;
        Matrix U_child1 = generate_column_bases(child1,
                                                c1_leaf_size,
                                                c1_diagonal_offset,
                                                c1_slice,
                                                child_level,
                                                randvec);// get_Ubig(child1, child_level);

        int c2_slice = N / c_num_nodes;
        int c2_leaf_size = (child2 == (c_num_nodes-1)) ? (N - (c2_slice * child2)) :  c2_slice;
        int c2_diagonal_offset = c2_slice * child2;
        Matrix U_child2 = generate_column_bases(child2,
                                                c2_leaf_size,
                                                c2_diagonal_offset,
                                                c2_slice,
                                                child_level,
                                                randvec);

        // Matrix U_child2 = get_Ubig(child2, child_level);

        Matrix Ubig = generate_column_bases(p,
                                            leaf_size,
                                            diagonal_offset,
                                            slice,
                                            level,
                                            randvec);
        std::vector<Matrix> Ubig_splits = Ubig.split(std::vector<int64_t>(1, U_child1.rows), {});

        Matrix Utransfer(rank * 2, rank);
        std::vector<Matrix> Utransfer_splits = Utransfer.split(2, 1);

        matmul(U_child1, Ubig_splits[0], Utransfer_splits[0], true, false, 1.0, 0.0);
        matmul(U_child2, Ubig_splits[1], Utransfer_splits[1], true, false, 1.0, 0.0);

        U.insert(p, level, std::move(Utransfer));
        Ublocks.push_back(Ubig);

        // Generate V transfer matrix.
        Matrix Vbig_child1 = generate_row_bases(child1,
                                                c1_leaf_size,
                                                c1_diagonal_offset,
                                                c1_slice,
                                                child_level,
                                                randvec);// get_Ubig(child1, child_level);

        // Matrix Vbig_child1 = get_Vbig(child1, child_level);
        Matrix Vbig_child2 = generate_row_bases(child2,
                                                c2_leaf_size,
                                                c2_diagonal_offset,
                                                c2_slice,
                                                child_level,
                                                randvec);// get_Ubig(child1, child_level);
        // Matrix Vbig_child2 = get_Vbig(child2, child_level);
        Matrix Vbig  = generate_row_bases(p,
                                          leaf_size,
                                          diagonal_offset,
                                          slice,
                                          level,
                                          randvec);
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
        int block_nrows = Ubig.rows;
        int block_ncols = Vbig.rows;
        int slice = N / int(pow(2, level));

        Matrix D = generate_laplacend_matrix(randvec, block_nrows, block_ncols,
                                             row * slice, col * slice);
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
        int diagonal_offset = slice * block;
        int leaf_size = (block == (num_nodes-1)) ? (N - (slice * block)) :  slice;

        double diagonal_error = rel_error(D(block, block, height),
                                          Hatrix::generate_laplacend_matrix(randvec, leaf_size, leaf_size,
                                                                            diagonal_offset, diagonal_offset));
        error += pow(diagonal_error, 2);
      }

      // regenerate off-diagonal blocks and test for correctness.
      for (int level = height; level > 0; --level) {
        int num_nodes = pow(2, level);
        int slice = N / num_nodes;

        for (int row = 0; row < num_nodes; ++row) {
          int col = row % 2 == 0 ? row + 1 : row - 1;
          Matrix Ubig = get_Ubig(row, level);
          Matrix Vbig = get_Vbig(col, level);
          int block_nrows = Ubig.rows;
          int block_ncols = Vbig.rows;
          Matrix expected = matmul(matmul(Ubig, S(row, col, level)), Vbig, false, true);
          Matrix actual = Hatrix::generate_laplacend_matrix(randvec, block_nrows, block_ncols,
                                                            row * slice, col * slice);
          double offD_error = rel_error(expected, actual);

          std::cout << "level=" << level << " row=" << row << " error=" << offD_error << std::endl;

          error += pow(offD_error, 2);
        }
      }

      return std::sqrt(error);
    }
  };
} // namespace Hatrix

int main(int argc, char *argv[]) {
  int N = atoi(argv[1]);
  int rank = atoi(argv[2]);
  int height = atoi(argv[3]);

  if (rank > int(N / pow(2, height))) {
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
