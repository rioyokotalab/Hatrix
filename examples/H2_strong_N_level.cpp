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
  class H2 {
  public:
    ColLevelMap U;
    RowLevelMap V;
    RowColLevelMap<Matrix> D, S;
    int64_t N, rank, height, admis;
    RowColLevelMap<bool> is_admissible;

  private:

    std::tuple<Matrix, Matrix> generate_column_bases(int block,
                                                     int leaf_size,
                                                     int diagonal_offset,
                                                     int slice,
                                                     const randvec_t& randvec) {
      int num_nodes = pow(2, height);
      Matrix row_slice(leaf_size, N - leaf_size);
      int64_t ncols_left_slice = block * slice;
      Matrix left_slice = generate_laplacend_matrix(randvec, leaf_size, ncols_left_slice,
                                                    diagonal_offset, 0);

      int64_t ncols_right_slice = block == num_nodes - 1 ? 0 : N - (block+1) * slice;
      Matrix right_slice = generate_laplacend_matrix(randvec, leaf_size, ncols_right_slice,
                                                     diagonal_offset, (block+1) * slice);

      std::vector<Matrix> row_slice_parts =
        row_slice.split(std::vector<int64_t>(1, 0),
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

    std::tuple<Matrix, Matrix> generate_row_bases(int block, int leaf_size,
                                                  int diagonal_offset,
                                                  int slice,
                                                  const randvec_t& randvec) {
      int num_nodes = pow(2, height);
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

      return {Ui, Hatrix::matmul(Si, Vi)};
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


    std::tuple<RowLevelMap, ColLevelMap> generate_leaf_nodes(const randvec_t& randpts) {
      int nblocks = pow(2, height);
      int64_t block_size = N / nblocks;
      ColLevelMap Ugen;
      RowLevelMap Vgen;

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          if (!is_admissible(i, j, height)) {
            D.insert(i, j, height,
                     Hatrix::generate_laplacend_matrix(randpts,
                                                       block_size, block_size,
                                                       i*block_size, j*block_size));
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
        for (int64_t j = 0; j < nblocks; ++j) {
          if (is_admissible(i, j, height)) {
            Hatrix::Matrix dense = Hatrix::generate_laplacend_matrix(randpts,
                                                                     block_size, block_size,
                                                                     i*block_size, j*block_size);
            Hatrix::matmul(dense, Y[j], AY);
          }
        }
        std::tie(Utemp, Stemp, Vtemp, error) = Hatrix::truncated_svd(AY, rank);
        U.insert(i, height, std::move(Utemp));
        Ugen.insert(i, height, matmul(Stemp, Vtemp));
      }

      for (int64_t j = 0; j < nblocks; ++j) {
        Hatrix::Matrix YtA(rank + oversampling, block_size);
        for (int64_t i = 0; i < nblocks; ++i) {
          if (is_admissible(i, j, height)) {
            Hatrix::Matrix dense = Hatrix::generate_laplacend_matrix(randpts,
                                                                     block_size, block_size,
                                                                     i*block_size, j*block_size);
            Hatrix::matmul(Y[i], dense, YtA, true);
          }
        }
        std::tie(Utemp, Stemp, Vtemp, error) = Hatrix::truncated_svd(transpose(YtA), rank);
        V.insert(j, height, std::move(Utemp));
        Vgen.insert(j, height, matmul(Stemp, Vtemp));
      }

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          if (is_admissible(i, j, height)) {
            Hatrix::Matrix dense = Hatrix::generate_laplacend_matrix(randpts,
                                                                     block_size, block_size,
                                                                     i*block_size, j*block_size);
            S.insert(i, j, height,
                     Hatrix::matmul(Hatrix::matmul(U(i, height), dense, true), V(j, height)));
          }
        }
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

      // int rank = leaf_size;

      Matrix Ubig_child1 = get_Ubig(child1, level+1);
      Matrix Ubig_child2 = get_Ubig(child2, level+1);

      int leaf_size = Ubig_child1.rows + Ubig_child2.rows;

      Matrix Ubig(rank, leaf_size);

      std::vector<Matrix> Ubig_splits =
        Ubig.split({}, std::vector<int64_t>(1, Ubig_child1.rows));
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

      std::vector<Matrix> Vbig_splits =
        Vbig.split(std::vector<int64_t>(1, Vbig_child1.rows), {});
      std::vector<Matrix> V_splits = V(p, level).split(2, 1);

      matmul(Vbig_child1, V_splits[0], Vbig_splits[0]);
      matmul(Vbig_child2, V_splits[1], Vbig_splits[1]);

      return Vbig;
    }

    std::tuple<RowLevelMap, ColLevelMap> generate_transfer_matrices(const randvec_t& randvec,
                                                                    RowLevelMap& Ugen,
                                                                    ColLevelMap& Vgen,
                                                                    const int level) {
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
        // Use max since the last block might differ in length.
        Matrix Ugen_concat(Ugen_upper.rows + Ugen_lower.rows,
                           std::max(Ugen_upper.cols, Ugen_lower.cols));
        // int rank = Ugen_concat.rows;
        std::vector<Matrix> Ugen_slices = Ugen_concat.split(2, 1);

        // Cannot use slices since the large matrix can have a larger
        //   dimension than the smaller ones.
        for (int i = 0; i < Ugen_upper.rows; i++) {
          for (int j = 0; j < Ugen_upper.cols; j++) {
            Ugen_slices[0](i, j) = Ugen_upper(i, j);
          }
        }
        for (int i = 0; i < Ugen_lower.rows; i++) {
          for (int j = 0; j < Ugen_lower.cols; j++) {
            Ugen_slices[1](i, j) = Ugen_lower(i, j);
          }
        }

        std::tie(Ui, Si, Vi, error) = truncated_svd(Ugen_concat, rank);
        U.insert(p, level, std::move(Ui));
        Ugen_transfer.insert(p, level, matmul(Si, Vi));

        // Generate V transfer matrix.
        Matrix& Vgen_upper = Vgen(child1, child_level);
        Matrix& Vgen_lower = Vgen(child2, child_level);
        Matrix Vgen_concat(Vgen_upper.rows + Vgen_lower.rows,
                           std::max(Vgen_upper.cols, Vgen_lower.cols));
        std::vector<Matrix> Vgen_slices = Vgen_concat.split(2, 1);
        for (int i = 0; i < Vgen_upper.rows; i++) {
          for (int j = 0; j < Vgen_upper.cols; j++) {
            Vgen_slices[0](i, j) = Vgen_upper(i, j);
          }
        }
        for (int i = 0; i < Vgen_lower.rows; i++) {
          for (int j = 0; j < Vgen_lower.cols; j++) {
            Vgen_slices[1](i, j) = Vgen_lower(i, j);
          }
        }

        std::tie(Ui, Si, Vi, error) = truncated_svd(Vgen_concat, rank);
        V.insert(p, level, std::move(Ui));
        Vgen_transfer.insert(p, level, matmul(Si, Vi));
      }

      for (int row = 0; row < num_nodes; ++row) {
        int col = row % 2 == 0 ? row + 1 : row - 1;
        if (is_admissible(row, col, level)) {
          S.insert(row, col, level,
                   generate_coupling_matrix(randvec, row, col, level));
        }
      }

      return {Ugen_transfer, Vgen_transfer};
    }

    void compute_matrix_structure(int64_t level) {
      if (level == 0) { return; }
      int64_t nodes = pow(2, level);
      if (level == height) {
        for (int i = 0; i < nodes; ++i) {
          for (int j = 0; j < nodes; ++j) {
            is_admissible.insert(i, j, level, std::abs(i - j) > admis);
          }
        }
      }
      else {
        int64_t child_level = level + 1;
        for (int i = 0; i < nodes; ++i) {
          std::vector<int> row_children({i * 2, i * 2 + 1});
          for (int j = 0; j < nodes; ++j) {
            std::vector<int> col_children({j * 2, j * 2 + 1});

            bool admis_block = true;
            for (int c1 = 0; c1 < 2; ++c1) {
              for (int c2 = 0; c2 < 2; ++c2) {
                if (is_admissible.exists(row_children[c1], col_children[c2], child_level) &&
                    !is_admissible(row_children[c1], col_children[c2], child_level)) {
                  admis_block = false;
                }
              }
            }

            if (admis_block) {
              for (int c1 = 0; c1 < 2; ++c1) {
                for (int c2 = 0; c2 < 2; ++c2) {
                  is_admissible.erase(row_children[c1], col_children[c2], child_level);
                }
              }
            }

            is_admissible.insert(i, j, level, std::move(admis_block));
          }
        }
      }

      compute_matrix_structure(level-1);
    }

    void actually_print_structure(int64_t level) {
      if (level == 0) { return; }
      int64_t nodes = pow(2, level);
      std::cout << "LEVEL: " << level << std::endl;
      for (int i = 0; i < nodes; ++i) {
        std::cout << "| " ;
        for (int j = 0; j < nodes; ++j) {
          if (is_admissible.exists(i, j, level)) {
            std::cout << is_admissible(i, j, level) << " | " ;
          }
          else {
            std::cout << "  | ";
          }
        }
        std::cout << std::endl;
      }

      actually_print_structure(level-1);
    }

  public:

    H2(const randvec_t& randpts, int64_t _N, int64_t _rank, int64_t _height,
       int64_t _admis) :
      N(_N), rank(_rank), height(_height), admis(_admis) {
      compute_matrix_structure(height);
      print_structure();

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
      for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
          if (!is_admissible(i, j, height)) {
            std::cout << "D-> r: " << i << " c: " << j << std::endl;
            int slice = N / num_nodes;
            double diagonal_error =
              rel_error(D(i, j, height),
                        Hatrix::generate_laplacend_matrix(randvec, slice, slice,
                                                          slice * i, slice * j));
            error += pow(diagonal_error, 2);
          }
        }
      }

      // regenerate off-diagonal blocks and test for correctness.
      for (int level = height; level > 1; --level) {
        int num_nodes = pow(2, level);
        int slice = N / num_nodes;

        for (int row = 0; row < num_nodes; ++row) {
          int col = row % 2 == 0 ? row + 1 : row - 1;

          if (is_admissible(row, col, level)) {
            Matrix Ubig = get_Ubig(row, level);
            Matrix Vbig = get_Vbig(col, level);
            int block_nrows = Ubig.rows;
            int block_ncols = Vbig.rows;
            Matrix expected = matmul(matmul(Ubig, S(row, col, level)), Vbig, false, true);
            Matrix actual = Hatrix::generate_laplacend_matrix(randvec, block_nrows, block_ncols,
                                                              row * slice, col * slice);
            double offD_error = rel_error(expected, actual);

            std::cout << "r: " << row << " c: " << col << std::endl;
            error += pow(offD_error, 2);
          }
        }
      }

      return std::sqrt(error);
    }

    void print_structure() {
      actually_print_structure(height);
    }
  };
} // namespace Hatrix

int main(int argc, char *argv[]) {
  int64_t N = atoi(argv[1]);
  int64_t rank = atoi(argv[2]);
  int64_t height = atoi(argv[3]);
  int64_t admis = atoi(argv[4]);

  if (rank > int(N / pow(2, height))) {
    std::cout << N << " % " << pow(2, height)
              << " != 0 || rank > leaf(" << int(N / pow(2, height))  << ")\n";
    abort();
  }

  Hatrix::Context::init();
  randvec_t randvec;
  randvec.push_back(equally_spaced_vector(N, 0.0, 1.0 * N)); // 1D
  randvec.push_back(equally_spaced_vector(N, 0.0, 1.0 * N)); // 2D
  randvec.push_back(equally_spaced_vector(N, 0.0, 1.0 * N)); // 3D

  auto start_construct = std::chrono::system_clock::now();
  Hatrix::H2 A(randvec, N, rank, height, admis);
  A.print_structure();
  auto stop_construct = std::chrono::system_clock::now();

  double error = A.construction_relative_error(randvec);

  Hatrix::Context::finalize();

  std::cout << "N= " << N << " rank= " << rank
            << " height=" << height <<  " const. error=" << error << std::endl;

}
