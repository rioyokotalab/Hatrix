#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <chrono>
#include <fstream>

#include "Hatrix/Hatrix.h"

constexpr double PV = 1;
using randvec_t = std::vector<std::vector<double> >;

double rel_error(const Hatrix::Matrix& A, const Hatrix::Matrix& B) {
  return Hatrix::norm(A - B) / Hatrix::norm(B);
}

// TODO: Make a better copy constructor for Matrix and replace this macro with a function.
#define SPLIT_DENSE(dense, row_split, col_split)        \
  dense.split(std::vector<int64_t>(1, row_split),       \
              std::vector<int64_t>(1, col_split));

std::vector<double> equally_spaced_vector(int N, double minVal, double maxVal) {
  std::vector<double> res(N, 0.0);
  double rnge = maxVal - minVal;
  for(int i=0; i<N; i++) {
    res[i] = minVal + ((double)i/(double)rnge);
  }
  return res;
}

namespace Hatrix {
  class H2 {
  public:
    ColLevelMap U;
    RowLevelMap V;
    RowColLevelMap<Matrix> D, S;
    int64_t N, rank, height, admis;
    RowColLevelMap<bool> is_admissible;

    int64_t oversampling = 5;

  private:
    bool row_has_admissible_blocks(int row, int level) {
      bool has_admis = false;
      for (int i = 0; i < pow(2, level); ++i) {
        if (is_admissible.exists(row, i, level) && is_admissible(row, i, level)) {
          has_admis = true;
          break;
        }
      }

      return has_admis;
    }

    bool col_has_admissible_blocks(int col, int level) {
      bool has_admis = false;
      for (int j = 0; j < pow(2, level); ++j) {
        if (is_admissible.exists(j, col, level) && is_admissible(j, col, level)) {
          has_admis = true;
          break;
        }
      }

      return has_admis;
    }

    // Generate U for the leaf.
    Matrix generate_column_bases(int block, int leaf_size, const randvec_t& randpts, std::vector<Matrix>& Y, int level) {
      // Row slice since column bases should be cutting across the columns.
      Matrix Ui, Si, Vi; double error;
      int nblocks = pow(2, level);
      Hatrix::Matrix AY(leaf_size, rank + oversampling);
      for (int64_t j = 0; j < nblocks; ++j) {
        if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) { continue; }
        Hatrix::Matrix dense = Hatrix::generate_laplacend_matrix(randpts,
                                                                 leaf_size, leaf_size,
                                                                 block*leaf_size, j*leaf_size);
        Hatrix::matmul(dense, Y[j], AY);
      }
      std::tie(Ui, Si, Vi, error) = Hatrix::truncated_svd(AY, rank);

      return Ui;
    }

    // Generate V for the leaf.
    Matrix generate_row_bases(int block, int leaf_size, const randvec_t& randpts, std::vector<Matrix>& Y, int level) {
      Matrix Ui, Si, Vi; double error;
      int nblocks = pow(2, level);

      // Col slice since row bases should be cutting across the rows.
      Hatrix::Matrix YtA(rank + oversampling, leaf_size);
      for (int64_t i = 0; i < nblocks; ++i) {
        if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) { continue; }
        Hatrix::Matrix dense = Hatrix::generate_laplacend_matrix(randpts,
                                                                 leaf_size, leaf_size,
                                                                 i*leaf_size, block*leaf_size);
        Hatrix::matmul(Y[i], dense, YtA, true);
      }
      std::tie(Ui, Si, Vi, error) = Hatrix::truncated_svd(transpose(YtA), rank);

      return Ui;
    }

    void generate_leaf_nodes(const randvec_t& randpts) {
      int nblocks = pow(2, height);
      int64_t block_size = N / nblocks;
      std::vector<Hatrix::Matrix> Y;

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          if (is_admissible.exists(i, j, height) && !is_admissible(i, j, height)) {
            D.insert(i, j, height,
                     Hatrix::generate_laplacend_matrix(randpts,
                                                       block_size, block_size,
                                                       i*block_size, j*block_size));
          }
        }
      }

      Hatrix::Matrix Utemp, Stemp, Vtemp;
      double error;

      // Generate a bunch of random matrices.
      for (int64_t i = 0; i < nblocks; ++i) {
        Y.push_back(
                    Hatrix::generate_random_matrix(block_size, rank + oversampling));
      }

      for (int64_t i = 0; i < nblocks; ++i) {
        U.insert(i, height, generate_column_bases(i, block_size, randpts, Y, height));
      }

      for (int64_t j = 0; j < nblocks; ++j) {
        V.insert(j, height, generate_row_bases(j, block_size, randpts, Y, height));
      }

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          if (is_admissible.exists(i, j, height) && is_admissible(i, j, height)) {
            Hatrix::Matrix dense = Hatrix::generate_laplacend_matrix(randpts,
                                                                     block_size, block_size,
                                                                     i*block_size, j*block_size);

            S.insert(i, j, height,
                     Hatrix::matmul(Hatrix::matmul(U(i, height), dense, true), V(j, height)));
          }
        }
      }
    }

  std::tuple<RowLevelMap, ColLevelMap>
  generate_transfer_matrices(const randvec_t& randpts,
                             int level, RowLevelMap& Uchild,
                             ColLevelMap& Vchild) {
      int num_nodes = pow(2, level);
      int leaf_size = N / num_nodes;
      int child_level = level + 1;
      int c_num_nodes = pow(2, child_level);
      std::vector<Matrix> Y;
      // Generate the actual bases for the upper level and pass it to this
      // function again for generating transfer matrices at successive levels.
      RowLevelMap Ubig_parent;
      ColLevelMap Vbig_parent;

      for (int i = 0; i < num_nodes; ++i) {
        Y.push_back(
                    Hatrix::generate_random_matrix(leaf_size, rank + oversampling));
      }

      for (int node = 0; node < num_nodes; ++node) {
        int child1 = node * 2;
        int child2 = node * 2 + 1;

        if (row_has_admissible_blocks(node, level)) {
          // Generate U transfer matrix.
          Matrix& Ubig_child1 = Uchild(child1, child_level);
          Matrix& Ubig_child2 = Uchild(child2, child_level);

          Matrix Ubig = generate_column_bases(node, leaf_size, randpts, Y, level);
          std::vector<Matrix> Ubig_splits = Ubig.split(2, 1);

          Matrix temp(Ubig_child1.cols + Ubig_child2.cols, Ubig.cols);
          std::vector<Matrix> temp_splits = temp.split(2, 1);

          matmul(Ubig_child1, Ubig_splits[0], temp_splits[0], true, false, 1, 0);
          matmul(Ubig_child2, Ubig_splits[1], temp_splits[1], true, false, 1, 0);

          Matrix Utransfer, Si, Vi; double error;
          std::tie(Utransfer, Si, Vi, error) = truncated_svd(temp, rank);
          U.insert(node, level, std::move(Utransfer));

          // Save the actual basis into the temporary Map to pass to generate
          // the S block and pass it to higher levels.
          Ubig_parent.insert(node, level, std::move(Ubig));
        }

        if (col_has_admissible_blocks(node, level)) {
          // Generate V transfer matrix.
          Matrix& Vbig_child1 = Vchild(child1, child_level);
          Matrix& Vbig_child2 = Vchild(child2, child_level);

          Matrix Vbig = generate_row_bases(node, leaf_size, randpts, Y, level);
          std::vector<Matrix> Vbig_splits = Vbig.split(2, 1);

          Matrix temp(Vbig_child1.cols + Vbig_child2.cols, Vbig.cols);
          std::vector<Matrix> temp_splits = temp.split(2, 1);

          matmul(Vbig_splits[0], Vbig_child1, temp_splits[0], true, false, 1, 0);
          matmul(Vbig_splits[1], Vbig_child2, temp_splits[1], true, false, 1, 0);

          Matrix Ui, Si, Vtransfer; double error;
          std::tie(Vtransfer, Si, Ui, error) = truncated_svd(temp, rank);
          V.insert(node, level, std::move(Vtransfer));

          Vbig_parent.insert(node, level, std::move(Vbig));
        }
      }

      for (int row = 0; row < num_nodes; ++row) {
        for (int col = 0; col < num_nodes; ++col) {
          if (is_admissible.exists(row, col, level) && is_admissible(row, col, level)) {
            Matrix D = generate_laplacend_matrix(randpts, leaf_size, leaf_size,
                                                 row * leaf_size, col * leaf_size);
            S.insert(row, col, level, matmul(matmul(Ubig_parent(row, level), D, true, false),
                                             Vbig_parent(col, level)));
          }
        }
      }

      return {Ubig_parent, Vbig_parent};
    }

    Matrix get_Ubig(int node, int level) {
      if (level == height) {
        return U(node, level);
      }
      int child1 = node * 2;
      int child2 = node * 2 + 1;

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

      Matrix Vbig(leaf_size, rank);

      std::vector<Matrix> Vbig_splits = Vbig.split(std::vector<int64_t>(1, Vbig_child1.rows), {});
      std::vector<Matrix> V_splits = V(node, level).split(2, 1);

      matmul(Vbig_child1, V_splits[0], Vbig_splits[0]);
      matmul(Vbig_child2, V_splits[1], Vbig_splits[1]);

      return Vbig;
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

    Matrix make_complement(const Hatrix::Matrix &Q) {
      Matrix Q_F(Q.rows, Q.rows);
      Matrix Q_full, R;
      std::tie(Q_full, R) = qr(Q, Hatrix::Lapack::QR_mode::Full, Hatrix::Lapack::QR_ret::OnlyQ);

      for (int i = 0; i < Q_F.rows; ++i) {
        for (int j = 0; j < Q_F.cols - Q.cols; ++j) {
          Q_F(i, j) = Q_full(i, j + Q.cols);
        }
      }

      for (int i = 0; i < Q_F.rows; ++i) {
        for (int j = 0; j < Q.cols; ++j) {
          Q_F(i, j + (Q_F.cols - Q.cols)) = Q(i, j);
        }
      }
      return Q_F;
    }


  public:

    H2(const randvec_t& randpts, int64_t _N, int64_t _rank, int64_t _height,
       int64_t _admis) :
      N(_N), rank(_rank), height(_height), admis(_admis) {
      RowLevelMap Uchild; ColLevelMap Vchild;

      compute_matrix_structure(height);
      generate_leaf_nodes(randpts);
      Uchild = U;
      Vchild = V;

      for (int level = height-1; level > 0; --level) {
        std::tie(Uchild, Vchild) = generate_transfer_matrices(randpts, level, Uchild, Vchild);
      }
    }

    double construction_relative_error(const randvec_t& randvec) {
      // verify diagonal matrix block constructions at the leaf level.
      double error = 0;
      double actual = 0, expected = 0;
      int num_nodes = pow(2, height);
      for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
          if (is_admissible.exists(i, j, height) && !is_admissible(i, j, height)) {
            int slice = N / num_nodes;
            actual += norm(Hatrix::generate_laplacend_matrix(randvec, slice, slice,
                                                             slice * i, slice * j));
            expected += norm(D(i, j, height));
          }
        }
      }

      // regenerate off-diagonal blocks and test for correctness.
      for (int level = height; level > 0; --level) {
        int num_nodes = pow(2, level);
        int slice = N / num_nodes;

        for (int row = 0; row < num_nodes; ++row) {
          for (int col = 0; col < num_nodes; ++col) {
            if (is_admissible.exists(row, col, height) && is_admissible(row, col, level)) {
              Matrix Ubig = get_Ubig(row, level);
              Matrix Vbig = get_Vbig(col, level);
              int block_nrows = Ubig.rows;
              int block_ncols = Vbig.rows;
              Matrix expected_matrix = matmul(matmul(Ubig, S(row, col, level)), Vbig, false, true);
              Matrix actual_matrix = Hatrix::generate_laplacend_matrix(randvec, block_nrows, block_ncols,
                                                                row * slice, col * slice);
              actual += norm(actual_matrix);
              expected += norm(expected_matrix);
            }
          }
        }
      }

      return std::sqrt(pow(std::abs(actual - expected) / expected, 2));
    }

    void print_structure() {
      actually_print_structure(height);
    }

    void factorize(const randvec_t &randpts) {
      for (int64_t level = height; level > height-1; --level) {
        int num_nodes = pow(2, level);
        int64_t block_size = N / num_nodes;

        for (int64_t block = 0; block < num_nodes; ++block) {
          // Step 0: Recompress fill-ins on the off-diagonals.

          // Step 1: Generate UF and VF blocks.
          Matrix UF = make_complement(U(block, level));
          Matrix VF = make_complement(V(block, level));

          // Step 2: Multiply the A with UF and VF.
          for (int j = 0; j < num_nodes; ++j) {
            if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) {
              D(block, j, level) = matmul(UF, D(block, j, level), true);
            }
          }

          for (int i = 0; i < num_nodes; ++i) {
            if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) {
              D(i, block, level) = matmul(D(i, block, level), VF);
            }
          }

          int64_t row_rank = U(block, level).cols, col_rank = V(block, level).cols;
        }
      }
    }

    Matrix solve(Matrix& b) {
      Matrix x(b);

      return x;
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
  randvec_t randpts;
  randpts.push_back(equally_spaced_vector(N, 0.0, 1.0 * N)); // 1D
  randpts.push_back(equally_spaced_vector(N, 0.0, 1.0 * N)); // 2D
  randpts.push_back(equally_spaced_vector(N, 0.0, 1.0 * N)); // 3D

  Hatrix::Matrix b = Hatrix::generate_random_matrix(N, 1);

  auto start_construct = std::chrono::system_clock::now();
  Hatrix::H2 A(randpts, N, rank, height, admis);
  A.factorize(randpts);
  auto stop_construct = std::chrono::system_clock::now();

  A.print_structure();

  Hatrix::Matrix x = A.solve(b);

  Hatrix::Matrix Adense = Hatrix::generate_laplacend_matrix(randpts, N, N, 0, 0, PV);
  Hatrix::Matrix x_solve = lu_solve(Adense, b);

  double construct_error = A.construction_relative_error(randpts);

  Hatrix::Context::finalize();

  double solve_error = Hatrix::norm(x - x_solve) / Hatrix::norm(x_solve);

  std::cout << "N= " << N << " rank= " << rank << " admis= " << admis << " leaf= " << int(N / pow(2, height))
            << " height=" << height <<  " const. error="
            << construct_error << " solve error=" << solve_error << std::endl;

}