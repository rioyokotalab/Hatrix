#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <chrono>
#include <fstream>

#include "Hatrix/Hatrix.h"

// UMV factorization and substitution of N-level HSS matrix. Compression
// done by following the technique in the Miro board. Implementation in
// HSS_Nlevel_construct_miro.cpp

// Algorithm taken from paper
// "Accuracy Controlled Direct Integral Equation Solver of Linear Complexity with Change
// of Basis for Large-Scale Interconnect Extraction"

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

    // Generate a column slice without the diagonal block.
    Matrix generate_column_slice(int block, int ncols, const randvec_t& randpts) {
      Matrix col_slice(N-ncols, ncols);
      int nrows_upper_slice = block * ncols;
      Matrix upper_slice = generate_laplacend_matrix(randpts, nrows_upper_slice, ncols,
                                                     0, block * ncols);
      int nrows_lower_slice = N - (block + 1) * ncols;
      Matrix lower_slice = generate_laplacend_matrix(randpts, nrows_lower_slice, ncols,
                                                     (block+1) * ncols, block * ncols);

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
                                             row * leaf_size, col * leaf_size);
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

    Hatrix::Matrix make_complement(const Hatrix::Matrix &Q) {
      Hatrix::Matrix Q_F(Q.rows, Q.rows);
      Hatrix::Matrix Q_full, R;
      std::tie(Q_full, R) = qr(Q, Hatrix::Lapack::QR_mode::Full,
        Hatrix::Lapack::QR_ret::OnlyQ);

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

    Matrix& unsolved_chunk(int block, int level, int rank) {
      Hatrix::Matrix& Diag = D(block, block, level);
      int c_size = Diag.rows - rank;
      std::vector<Hatrix::Matrix> Diag_splits = Diag.split(std::vector<int64_t>(1, c_size),
                                                           std::vector<int64_t>(1, c_size));
      return Diag_splits[3];
    }

    // permute the vector forward and return the offset at which the new vector begins.
    int permute_forward(Hatrix::Matrix& x, const int level, int rank_offset) {
      Hatrix::Matrix copy(x);
      int num_nodes = int(pow(2, level));
      int c_offset = rank_offset;
      for (int block = 0; block < num_nodes; ++block) {
        rank_offset += D(block, block, level).rows - U(block, level).cols;
      }

      for (int block = 0; block < num_nodes; ++block) {
        int rows = D(block, block, level).rows;
        int rank = U(block, level).cols;
        int c_size = rows - rank;

        // copy the complement part of the vector into the temporary vector
        for (int i = 0; i < c_size; ++i) {
          copy(c_offset + c_size * block + i, 0) = x(c_offset + block * rows + i, 0);
        }
        // copy the rank part of the vector into the temporary vector
        for (int i = 0; i < rank; ++i) {
          copy(rank_offset + rank * block + i, 0) = x(c_offset + block * rows + c_size + i, 0);
        }
      }

      x = copy;

      return rank_offset;
    }

    int permute_backward(Hatrix::Matrix& x, const int level, int rank_offset) {
      Hatrix::Matrix copy(x);
      int num_nodes = pow(2, level);
      int c_offset = rank_offset;
      for (int block = 0; block < num_nodes; ++block) {
        c_offset -= D(block, block, level).rows - U(block, level).cols;
      }

      for (int block = 0; block < num_nodes; ++block) {
        int rows = D(block, block, level).rows;
        int rank = U(block, level).cols;
        int c_size = rows - rank;

        for (int i = 0; i < c_size; ++i) {
          copy(c_offset + block * rows + i, 0) = x(c_offset + block * c_size + i, 0);
        }

        for (int i = 0; i < rank; ++i) {
          copy(c_offset + block * rows + c_size + i, 0) = x(rank_offset + rank * block + i, 0);
        }
      }

      x = copy;

      return c_offset;
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
        error += Hatrix::norm(D(block, block, height) - Hatrix::generate_laplacend_matrix(randpts, slice, slice,
                                                                                          slice * block, slice * block));
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
          error += Hatrix::norm(expected - actual);
        }
      }
      return std::sqrt(error / N / N);
    }

    // UMV factorization of this HSS matrix.
    void factorize() {
      for (int level = height; level > 0; --level) {
        int num_nodes = pow(2, level);

        // Perform multiplication of U_F and V_F along with partial LU.
        for (int block = 0; block < num_nodes; ++block) {
          Hatrix::Matrix& diagonal = D(block, block, level);

          Hatrix::Matrix U_F = make_complement(U(block, level));
          Hatrix::Matrix V_F = make_complement(V(block, level));

          diagonal = matmul(matmul(U_F, diagonal, true, false), V_F);

          // in case of full rank, dont perform partial LU
          if (rank == diagonal.rows) { continue; }

          int c_size = diagonal.rows - rank;
          std::vector<Hatrix::Matrix> diagonal_splits = diagonal.split(std::vector<int64_t>(1, c_size),
                                                                       std::vector<int64_t>(1, c_size));
          Hatrix::Matrix& Dcc = diagonal_splits[0];
          Hatrix::Matrix& Dco = diagonal_splits[1];
          Hatrix::Matrix& Doc = diagonal_splits[2];
          Hatrix::Matrix& Doo = diagonal_splits[3];


          Hatrix::lu(Dcc);
          solve_triangular(Dcc, Dco, Hatrix::Left, Hatrix::Lower, true, false, 1.0);
          solve_triangular(Dcc, Doc, Hatrix::Right, Hatrix::Upper, false, false, 1.0);
          matmul(Doc, Dco, Doo, false, false, -1.0, 1.0);
        }

        // Merge the unfactorized parts.
        int parent_level = level - 1;
        for (int block = 0; block < int(pow(2, parent_level)); ++block) {
          Hatrix::Matrix D_unsolved(rank * 2, rank * 2);
          std::vector<Hatrix::Matrix> D_unsolved_splits = D_unsolved.split(2, 2);

          int child1 = block * 2; int child2 = block * 2 + 1;
          D_unsolved_splits[0] = unsolved_chunk(child1, level, rank);
          D_unsolved_splits[3] = unsolved_chunk(child2, level, rank);

          int col_child1 = child1 % 2 == 0 ? child1 + 1 : child1 - 1;
          int col_child2 = child2 % 2 == 0 ? child2 + 1 : child2 - 1;

          D_unsolved_splits[1] = S(child1, col_child1, level);
          D_unsolved_splits[2] = S(child2, col_child2, level);

          D.insert(block, block, parent_level, std::move(D_unsolved));
        }
      }
      Hatrix::lu(D(0, 0, 0));
    }

    // Forward/backward substitution of UMV factorized HSS matrix.
    Hatrix::Matrix solve(const Hatrix::Matrix& b) {
      std::vector<Hatrix::Matrix> x_splits;
      Hatrix::Matrix x(b);
      int rhs_offset = 0, c_size, offset;

      // forward
      for (int level = height; level > 0; --level) {
        int num_nodes = pow(2, level);
        for (int node = 0; node < num_nodes; ++node) {
          Matrix& Diag = D(node, node, level);
          c_size = Diag.rows - rank;
          offset = rhs_offset + node * Diag.rows;

          Matrix temp(Diag.rows, 1);
          for (int i = 0; i < Diag.rows; ++i) {
            temp(i, 0) = x(offset + i, 0);
          }
          Matrix U_F = make_complement(U(node, level));
          Matrix product = matmul(U_F, temp, true);
          for (int i = 0; i < Diag.rows; ++i) {
            x(offset + i, 0) = product(i, 0);
          }

          // don't compute partial LU if full rank.
          if (rank == Diag.rows) { continue; }

          x_splits = x.split(
                             {offset, offset + c_size, offset + Diag.rows}, {});
          Hatrix::Matrix& c = x_splits[1];
          Hatrix::Matrix& o = x_splits[2];

          std::vector<Matrix> D_splits = Diag.split(std::vector<int64_t>(1, c_size),
                                                    std::vector<int64_t>(1, c_size));
          Matrix& Dcc = D_splits[0];
          Matrix& Doc = D_splits[2];

          solve_triangular(Dcc, c, Hatrix::Left, Hatrix::Lower, true);
          matmul(Doc, c, o, false, false, -1.0, 1.0);
        }

        rhs_offset = permute_forward(x, level, rhs_offset);
      }

      x_splits = x.split(std::vector<int64_t>(1, rhs_offset), {});
      solve_triangular(A.D(0, 0, 0), x_splits[1], Hatrix::Left, Hatrix::Lower, true);
      solve_triangular(A.D(0, 0, 0), x_splits[1], Hatrix::Left, Hatrix::Upper, false);

      // backward
      for (int level = 1; level < height; ++level) {

      }

      return x;
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
  Hatrix::Matrix b = Hatrix::generate_random_matrix(N, 1);

  double construct_error = A.construction_relative_error(randpts);

  // UMV factorization and substitution of HSS matrix.
  A.factorize();
  Hatrix::Matrix x = A.solve(b);

  // Verification with dense solver.
  Hatrix::Matrix Adense = Hatrix::generate_laplacend_matrix(randpts, N, N, 0, 0);
  Hatrix::Matrix x_solve(b);
  Hatrix::lu(Adense);
  Hatrix::solve_triangular(Adense, x_solve, Hatrix::Left, Hatrix::Lower, true);
  Hatrix::solve_triangular(Adense, x_solve, Hatrix::Left, Hatrix::Upper, false);

  double solve_error = rel_error(x, x_solve);

  Hatrix::Context::finalize();
  std::cout << "N= " << N << " rank= " << rank << " height=" << height
            << " construction error=" << construct_error
            << " solve error=" << solve_error << std::endl;
}
