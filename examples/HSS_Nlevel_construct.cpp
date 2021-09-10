#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <chrono>
#include <ofstream>

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

    Matrix generate_coupling_matrix(const randvec_t& randvec, const int row, const int col,
                                    const int level) {
      Matrix Ubig = get_Ubig(row, level);
      Matrix Vbig = get_Vbig(col, level);
      int block_size = N / int(pow(2, level));
      Matrix D = generate_laplacend_matrix(randvec, block_size, block_size,
                                           row * block_size, col * block_size);
      return matmul(matmul(Ubig, D, true, false), Vbig);
    }


    std::tuple<RowLevelMap, ColLevelMap> generate_leaf_nodes(const randvec_t& randvec) {
      int nblocks = pow(2, height);
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

      for (int row = 0; row < nblocks; ++row) {
        int col = row % 2 == 0 ? row + 1 : row - 1;
        S.insert(row, col, height, generate_coupling_matrix(randvec, row, col, height));
      }

      return {Ugen, Vgen};
    }

    Matrix get_Ubig(int p, int level) {
      if (level == height) {
        return U(p, level);
      }
      int child1 = p * 2;
      int child2 = p * 2 + 1;
      int num_nodes = pow(2, level);
      int leaf_size = int(N / num_nodes);
      // int rank = leaf_size;

      Matrix Ubig(rank, leaf_size);

      std::vector<Matrix> Ubig_splits = Ubig.split(1, 2);
      std::vector<Matrix> U_splits = U(p, level).split(2, 1);

      matmul(U_splits[0], get_Ubig(child1, level+1), Ubig_splits[0], true, true);
      matmul(U_splits[1], get_Ubig(child2, level+1), Ubig_splits[1], true, true);

      return transpose(Ubig);
    }

    Matrix get_Vbig(int p, int level) {
      if (level == height) {
        return V(p, level);
      }
      int child1 = p * 2;
      int child2 = p * 2 + 1;
      int num_nodes = pow(2, level);
      int leaf_size = int(N / num_nodes);
      // int rank = leaf_size;

      Matrix Vbig(leaf_size, rank);

      std::vector<Matrix> Vbig_splits = Vbig.split(2, 1);
      std::vector<Matrix> V_splits = V(p, level).split(2, 1);

      matmul(get_Vbig(child1, level+1), V_splits[0], Vbig_splits[0]);
      matmul(get_Vbig(child2, level+1), V_splits[1], Vbig_splits[1]);

      return Vbig;
    }

    std::tuple<RowLevelMap, ColLevelMap> generate_transfer_matrices(const randvec_t& randvec, RowLevelMap& Ugen,
                                                                    ColLevelMap& Vgen, const int level) {
      Matrix Ui, Si, Vi; double error;
      RowLevelMap Ugen_transfer; ColLevelMap Vgen_transfer;
      int num_nodes = pow(2, level);

      for (int p = 0; p < num_nodes; ++p) {
        int child1 = p * 2;
        int child2 = p * 2 + 1;
        int child_level = level + 1;

        // Generate U transfer matrix.
        Matrix& Ugen_upper = Ugen(child1, child_level);
        Matrix& Ugen_lower = Ugen(child2, child_level);
        Matrix Ugen_concat(Ugen_upper.rows + Ugen_lower.rows, Ugen_upper.cols);
        // int rank = Ugen_concat.rows;
        std::vector<Matrix> Ugen_slices = Ugen_concat.split(2, 1);
        Ugen_slices[0] = Ugen_upper;
        Ugen_slices[1] = Ugen_lower;

        std::tie(Ui, Si, Vi, error) = truncated_svd(Ugen_concat, rank);
        U.insert(p, level, std::move(Ui));
        Ugen_transfer.insert(p, level, matmul(Si, Vi));

        // Generate V transfer matrix.
        Matrix& Vgen_upper = Vgen(child1, child_level);
        Matrix& Vgen_lower = Vgen(child2, child_level);
        Matrix Vgen_concat(Vgen_upper.rows + Vgen_lower.rows, Vgen_upper.cols);
        std::vector<Matrix> Vgen_slices = Vgen_concat.split(2, 1);
        Vgen_slices[0] = Vgen_upper;
        Vgen_slices[1] = Vgen_lower;

        std::tie(Ui, Si, Vi, error) = truncated_svd(Vgen_concat, rank);
        V.insert(p, level, std::move(Ui));
        Vgen_transfer.insert(p, level, matmul(Si, Vi));
      }

      for (int row = 0; row < num_nodes; ++row) {
        int col = row % 2 == 0 ? row + 1 : row - 1;
        S.insert(row, col, level,
                 generate_coupling_matrix(randvec, row, col, level));
      }

      return {Ugen_transfer, Vgen_transfer};
    }

  public:

    HSS(const randvec_t& randpts, int _N, int _rank, int _height) :
      N(_N), rank(_rank), height(_height) {
      RowLevelMap Ugen; ColLevelMap Vgen;
      std::tie(Ugen, Vgen) = generate_leaf_nodes(randpts);
      for (int level = height-1; level > 0; --level) {
        std::tie(Ugen, Vgen) = generate_transfer_matrices(randpts, Ugen, Vgen, level);
      }
    }

    double construction_relative_error(const randvec_t& randvec) {
      // verify diagonal matrix block constructions at the leaf level.
      double error = 0;
      int num_nodes = pow(2, height);
      for (int block = 0; block < num_nodes; ++block) {
        int leaf_size = N / num_nodes;
        double diagonal_error = rel_error(D(block, block, height),
                                          Hatrix::generate_laplacend_matrix(randvec, leaf_size, leaf_size,
                                                                            block * leaf_size, block * leaf_size));
        error += pow(diagonal_error, 2);
      }

      // regenerate off-diagonal blocks and test for correctness.
      for (int level = height; level > 0; --level) {
        int num_nodes = pow(2, level);
        int block_size = N / num_nodes;

        for (int row = 0; row < num_nodes; ++row) {
          int col = row % 2 == 0 ? row + 1 : row - 1;
          Matrix Ubig = get_Ubig(row, level);
          Matrix Vbig = get_Vbig(col, level);
          Matrix expected = matmul(matmul(Ubig, S(row, col, level)), Vbig, false, true);
          Matrix actual = Hatrix::generate_laplacend_matrix(randvec, block_size, block_size,
                                                            row * block_size, col * block_size);
          double offD_error = rel_error(expected, actual);
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

  if (N % int(pow(2, height)) != 0 || rank > int(N / pow(2, height))) {
    std::cout << N << " % " << pow(2, height) << " != 0 || rank > leaf(" << int(N / pow(2, height))  << ")\n";
    abort();
  }

  Hatrix::Context::init();
  randvec_t randvec;
  randvec.push_back(equally_spaced_vector(N, 0.0, 1.0)); // 1D

  auto start_construct = std::chrono::system_clock::now();
  Hatrix::HSS A(randvec, N, rank, height);
  auto stop_construct = std::chrono::system_clock::now();

  std::cout << " r: " << A.U(0, 2).rows << " c: " << A.U(0, 1).cols << std::endl;

  double error = A.construction_relative_error(randvec);

  Hatrix::Context::finalize();

  std::ofstream file;
  file.open("output.txt", std::ios::app | std::ios::out);
  file << "N= " << N << " rank= " << rank << " height=" << height <<  " const. error=" << error << std::endl;
  file.close();

}
