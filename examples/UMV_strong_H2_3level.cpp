#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <chrono>
#include <fstream>
#include <numeric>

#include "Hatrix/Hatrix.h"

double PV = 1e-3;
using randvec_t = std::vector<std::vector<double> >;
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
    void actually_print_structure(int64_t level);
    int64_t permute_forward(Matrix& x, const int64_t level, int64_t rank_offset);
    int64_t permute_backward(Matrix& x, const int64_t level, int64_t rank_offset);
    bool row_has_admissible_blocks(int row, int level);
    bool col_has_admissible_blocks(int col, int level);
    Matrix generate_column_block(int block, int block_size, const randvec_t& randpts, int level);
    Matrix generate_column_bases(int block, int block_size, const randvec_t& randpts,
                                       std::vector<Matrix>& Y, int level);
    Matrix generate_row_block(int block, int block_size, const randvec_t& randpts, int level);
    Matrix generate_row_bases(int block, int block_size, const randvec_t& randpts,
                              std::vector<Matrix>& Y, int level);
    void generate_leaf_nodes(const randvec_t& randpts);
    std::tuple<RowLevelMap, ColLevelMap> generate_transfer_matrices(const randvec_t& randpts,
                                                                    int level, RowLevelMap& Uchild,
                                                                    ColLevelMap& Vchild);
    Matrix get_Ubig(int node, int level);
    Matrix get_Vbig(int node, int level);
    void compute_matrix_structure(int64_t level);

  public:
    H2(const randvec_t& randpts, int64_t _N, int64_t _rank, int64_t _height,
             int64_t _admis);
    H2(const H2& A);
    double construction_relative_error(const randvec_t& randvec);
    void print_structure();
    void factorize(const randvec_t &randpts);
    Matrix solve(Matrix& b);
  };
}

using namespace Hatrix;

double rel_error(const Hatrix::Matrix& A, const Hatrix::Matrix& B) {
  return Hatrix::norm(A - B) / Hatrix::norm(B);
}

// TODO: Make a better copy constructor for Matrix and replace this macro with a function.
#define SPLIT_DENSE(dense, row_split, col_split)        \
  dense.split(std::vector<int64_t>(1, row_split),       \
              std::vector<int64_t>(1, col_split));

Hatrix::Matrix make_complement(const Hatrix::Matrix &Q) {
  Hatrix::Matrix Q_F(Q.rows, Q.rows);
  Hatrix::Matrix Q_full, R;
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

Hatrix::Matrix lower(Hatrix::Matrix A) {
  Hatrix::Matrix mat(A.rows, A.cols);

  for (int i = 0; i < A.rows; ++i) {
    mat(i, i) = 1.0;
    for (int j = 0; j < i; ++j) {
      mat(i, j) = A(i, j);
    }
  }

  return mat;
}

Hatrix::Matrix upper(Hatrix::Matrix A) {
  Hatrix::Matrix mat(A.rows, A.cols);

  for (int i = 0; i < A.rows; ++i) {
    for (int j = i; j < A.cols; ++j) {
      mat(i, j) = A(i, j);
    }
  }

  return mat;
}


std::vector<int64_t>
generate_offsets(Hatrix::H2& A, int level) {
  std::vector<int64_t> offsets;
  int c_size = A.N / pow(2, A.height) - A.rank;
  int size_offset = 0;

  // add offsets for level offet
  for (int l = A.height; l > level; --l) {
    if (A.U.exists(0, l)) {
      size_offset += (A.U(0, l).rows - A.U(0, l).cols) * pow(2, l);
      offsets.push_back(size_offset);
    }
  }

  // add offsets for the layout of the permuted matrix.
  int num_nodes = pow(2, level);

  // offsets for compliment sizes.
  for (int i = 0; i < num_nodes; ++i) {
    c_size = A.U(i, level).rows - A.U(i, level).cols;
    size_offset += c_size;
    offsets.push_back(size_offset);
  }

  // offsets for rank sizes.
  for (int i = 1; i < num_nodes; ++i) {
    size_offset += (A.U(i, level).cols);
    offsets.push_back(size_offset);
  }

  return offsets;
}

// Generate offsets for the large dense matrix that will hold the last
// block that is factorized by the UMV.
std::vector<int64_t>
generate_top_level_offsets(Hatrix::H2& A) {
  std::vector<int64_t> offsets;
  int size_offset = 0;
  int level = A.height;

  for (; level > 0; --level) {
    if (A.U.exists(0, level)) {
      size_offset += (A.U(0, level).rows -
                      A.U(0, level).cols) * pow(2, level);
      offsets.push_back(size_offset);
    }
    else {
      break;
    }
  }

  int num_nodes = pow(2, level);

  for (int i = 0; i < num_nodes; ++i) {
    size_offset += A.D(i, i, level).rows;
    offsets.push_back(size_offset);
  }

  return offsets;
}

namespace Hatrix {
  void H2::actually_print_structure(int64_t level) {
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

  // permute the vector forward and return the offset at which the new vector begins.
  int64_t H2::permute_forward(Matrix& x, const int64_t level, int64_t rank_offset) {
    Matrix copy(x);
    int64_t num_nodes = int64_t(pow(2, level));
    int64_t c_offset = rank_offset;
    for (int64_t block = 0; block < num_nodes; ++block) {
      rank_offset += D(block, block, level).rows - U(block, level).cols;
    }

    for (int64_t block = 0; block < num_nodes; ++block) {
      int64_t rows = D(block, block, level).rows;
      int64_t rank = U(block, level).cols;
      int64_t c_size = rows - rank;

      // copy the complement part of the vector into the temporary vector
      for (int64_t i = 0; i < c_size; ++i) {
        copy(c_offset + c_size * block + i, 0) = x(c_offset + block * rows + i, 0);
      }
      // copy the rank part of the vector into the temporary vector
      for (int64_t i = 0; i < rank; ++i) {
        copy(rank_offset + rank * block + i, 0) = x(c_offset + block * rows + c_size + i, 0);
      }
    }

    x = copy;

    return rank_offset;
  }

  int64_t H2::permute_backward(Matrix& x, const int64_t level, int64_t rank_offset) {
    Matrix copy(x);
    int64_t num_nodes = pow(2, level);
    int64_t c_offset = rank_offset;
    for (int64_t block = 0; block < num_nodes; ++block) {
      c_offset -= D(block, block, level).rows - U(block, level).cols;
    }

    for (int64_t block = 0; block < num_nodes; ++block) {
      int64_t rows = D(block, block, level).rows;
      int64_t rank = U(block, level).cols;
      int64_t c_size = rows - rank;

      for (int64_t i = 0; i < c_size; ++i) {
        copy(c_offset + block * rows + i, 0) = x(c_offset + block * c_size + i, 0);
      }

      for (int64_t i = 0; i < rank; ++i) {
        copy(c_offset + block * rows + c_size + i, 0) = x(rank_offset + rank * block + i, 0);
      }
    }

    x = copy;

    return c_offset;
  }

  bool H2::row_has_admissible_blocks(int row, int level) {
    bool has_admis = false;
    for (int i = 0; i < pow(2, level); ++i) {
      if (is_admissible.exists(row, i, level) && is_admissible(row, i, level)) {
        has_admis = true;
        break;
      }
    }

    return has_admis;
  }

  bool H2::col_has_admissible_blocks(int col, int level) {
    bool has_admis = false;
    for (int j = 0; j < pow(2, level); ++j) {
      if (is_admissible.exists(j, col, level) && is_admissible(j, col, level)) {
        has_admis = true;
        break;
      }
    }

    return has_admis;
  }

  Matrix H2::generate_column_block(int block, int block_size, const randvec_t& randpts, int level) {
    Matrix AY(block_size, 0);
    int nblocks = pow(2, level);
    for (int64_t j = 0; j < nblocks; ++j) {
      if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) { continue; }
      Hatrix::Matrix dense = Hatrix::generate_laplacend_matrix(randpts,
                                                               block_size, block_size,
                                                               block*block_size, j*block_size, PV);

      AY = concat(AY, dense, 1);
    }
    return AY;
  }

  // Generate U for the leaf.
  Matrix H2::generate_column_bases(int block, int block_size, const randvec_t& randpts,
                                   std::vector<Matrix>& Y, int level) {
    // Row slice since column bases should be cutting across the columns.
    Matrix AY = generate_column_block(block, block_size, randpts, level);

    Matrix Ui, Si, Vi; double error;
    std::tie(Ui, Si, Vi, error) = Hatrix::truncated_svd(AY, rank);

    return Ui;
  }

  Matrix H2::generate_row_block(int block, int block_size, const randvec_t& randpts, int level) {
    Hatrix::Matrix YtA(0, block_size);
    for (int64_t i = 0; i < pow(2, level); ++i) {
      if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) { continue; }
      Hatrix::Matrix dense = Hatrix::generate_laplacend_matrix(randpts,
                                                               block_size, block_size,
                                                               i*block_size, block*block_size, PV);
      YtA = concat(YtA, dense, 0);
    }

    return YtA;
  }

  // Generate V for the leaf.
  Matrix H2::generate_row_bases(int block, int block_size, const randvec_t& randpts,
                                std::vector<Matrix>& Y, int level) {
    Matrix YtA = generate_row_block(block, block_size, randpts, level);

    Matrix Ui, Si, Vi; double error;
    std::tie(Ui, Si, Vi, error) = Hatrix::truncated_svd(YtA, rank);

    return transpose(Vi);
  }

  void H2::generate_leaf_nodes(const randvec_t& randpts) {
    int nblocks = pow(2, height);
    int64_t block_size = N / nblocks;
    std::vector<Hatrix::Matrix> Y;

    for (int i = 0; i < nblocks; ++i) {
      for (int j = 0; j < nblocks; ++j) {
        if (is_admissible.exists(i, j, height) && !is_admissible(i, j, height)) {
          D.insert(i, j, height,
                   Hatrix::generate_laplacend_matrix(randpts,
                                                     block_size, block_size,
                                                     i*block_size, j*block_size, PV));
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
                                                                   i*block_size, j*block_size, PV);

          S.insert(i, j, height,
                   Hatrix::matmul(Hatrix::matmul(U(i, height), dense, true), V(j, height)));
        }
      }
    }
  }

  std::tuple<RowLevelMap, ColLevelMap>
  H2::generate_transfer_matrices(const randvec_t& randpts,
                                 int level, RowLevelMap& Uchild,
                                 ColLevelMap& Vchild) {
    int num_nodes = pow(2, level);
    int block_size = N / num_nodes;
    int child_level = level + 1;
    int c_num_nodes = pow(2, child_level);
    std::vector<Matrix> Y;
    // Generate the actual bases for the upper level and pass it to this
    // function again for generating transfer matrices at successive levels.
    RowLevelMap Ubig_parent;
    ColLevelMap Vbig_parent;

    for (int i = 0; i < num_nodes; ++i) {
      Y.push_back(
                  Hatrix::generate_random_matrix(block_size, rank + oversampling));
    }

    for (int node = 0; node < num_nodes; ++node) {
      int child1 = node * 2;
      int child2 = node * 2 + 1;

      if (row_has_admissible_blocks(node, level)) {
        // Generate U transfer matrix.
        Matrix& Ubig_child1 = Uchild(child1, child_level);
        Matrix& Ubig_child2 = Uchild(child2, child_level);

        // Shown as Alevel_node_plus on miro.
        Matrix col_block = generate_column_block(node, block_size, randpts, level);
        auto col_block_splits = col_block.split(2, 1);

        Matrix temp(Ubig_child1.cols + Ubig_child2.cols, col_block.cols);
        auto temp_splits = temp.split(2, 1);

        matmul(Ubig_child1, col_block_splits[0], temp_splits[0], true, false, 1, 0);
        matmul(Ubig_child2, col_block_splits[1], temp_splits[1], true, false, 1, 0);

        Matrix Utransfer, Si, Vi; double error;
        std::tie(Utransfer, Si, Vi, error) = truncated_svd(temp, rank);
        U.insert(node, level, std::move(Utransfer));

        // Generate the full bases to pass onto the parent.
        auto Utransfer_splits = U(node, level).split(2, 1);
        Matrix Ubig(block_size, rank);
        auto Ubig_splits = Ubig.split(2, 1);

        matmul(Ubig_child1, Utransfer_splits[0], Ubig_splits[0]);
        matmul(Ubig_child2, Utransfer_splits[1], Ubig_splits[1]);

        Ubig_parent.insert(node, level, std::move(Ubig));
      }

      if (col_has_admissible_blocks(node, level)) {
        // Generate V transfer matrix.
        Matrix& Vbig_child1 = Vchild(child1, child_level);
        Matrix& Vbig_child2 = Vchild(child2, child_level);

        Matrix row_block = generate_row_block(node, block_size, randpts, level);
        auto row_block_splits = row_block.split(1, 2);

        Matrix temp(row_block.rows, Vbig_child1.cols + Vbig_child2.cols);
        auto temp_splits = temp.split(1, 2);

        matmul(row_block_splits[0], Vbig_child1, temp_splits[0]);
        matmul(row_block_splits[1], Vbig_child2, temp_splits[1]);

        Matrix Ui, Si, Vtransfer; double error;
        std::tie(Ui, Si, Vtransfer, error) = truncated_svd(temp, rank);
        V.insert(node, level, transpose(Vtransfer));

        // Generate the full bases for passing onto the upper level.
        std::vector<Matrix> Vtransfer_splits = V(node, level).split(2, 1);
        Matrix Vbig(rank, block_size);
        std::vector<Matrix> Vbig_splits = Vbig.split(1, 2);

        matmul(Vtransfer_splits[0], Vbig_child1, Vbig_splits[0], true, true, 1, 0);
        matmul(Vtransfer_splits[1], Vbig_child2, Vbig_splits[1], true, true, 1, 0);

        Vbig_parent.insert(node, level, transpose(Vbig));

      }
    }

    for (int row = 0; row < num_nodes; ++row) {
      for (int col = 0; col < num_nodes; ++col) {
        if (is_admissible.exists(row, col, level) && is_admissible(row, col, level)) {
          Matrix D = generate_laplacend_matrix(randpts, block_size, block_size,
                                               row * block_size, col * block_size, PV);
          S.insert(row, col, level, matmul(matmul(Ubig_parent(row, level), D, true, false),
                                           Vbig_parent(col, level)));
        }
      }
    }

    return {Ubig_parent, Vbig_parent};
  }

  Matrix H2::get_Ubig(int node, int level) {
    if (level == height) {
      return U(node, level);
    }
    int child1 = node * 2;
    int child2 = node * 2 + 1;

    Matrix Ubig_child1 = get_Ubig(child1, level+1);
    Matrix Ubig_child2 = get_Ubig(child2, level+1);

    int block_size = Ubig_child1.rows + Ubig_child2.rows;

    Matrix Ubig(block_size, rank);

    std::vector<Matrix> Ubig_splits = Ubig.split(
                                                 std::vector<int64_t>(1,
                                                                      Ubig_child1.rows),
                                                 {});

    std::vector<Matrix> U_splits = U(node, level).split(2, 1);

    matmul(Ubig_child1, U_splits[0], Ubig_splits[0]);
    matmul(Ubig_child2, U_splits[1], Ubig_splits[1]);

    return Ubig;
  }

  Matrix H2::get_Vbig(int node, int level) {
    if (level == height) {
      return V(node, level);
    }
    int child1 = node * 2;
    int child2 = node * 2 + 1;

    Matrix Vbig_child1 = get_Vbig(child1, level+1);
    Matrix Vbig_child2 = get_Vbig(child2, level+1);

    int block_size = Vbig_child1.rows + Vbig_child2.rows;

    Matrix Vbig(block_size, rank);

    std::vector<Matrix> Vbig_splits = Vbig.split(std::vector<int64_t>(1, Vbig_child1.rows), {});
    std::vector<Matrix> V_splits = V(node, level).split(2, 1);

    matmul(Vbig_child1, V_splits[0], Vbig_splits[0]);
    matmul(Vbig_child2, V_splits[1], Vbig_splits[1]);

    return Vbig;
  }

  void H2::compute_matrix_structure(int64_t level) {
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


  H2::H2(const randvec_t& randpts, int64_t _N, int64_t _rank, int64_t _height,
         int64_t _admis) :
    N(_N), rank(_rank), height(_height), admis(_admis) {
    RowLevelMap Uchild; ColLevelMap Vchild;

    compute_matrix_structure(height);
    // Add (0,0,0) as an inadmissible block to aid the last merge + factorize of UMV.
    is_admissible.insert(0, 0, 0, false);
    generate_leaf_nodes(randpts);
    Uchild = U;
    Vchild = V;

    for (int level = height-1; level > 0; --level) {
      std::tie(Uchild, Vchild) = generate_transfer_matrices(randpts, level, Uchild, Vchild);
    }
  }

  H2::H2(const H2& A) : N(A.N), rank(A.rank), height(A.height), admis(A.admis),
                        U(A.U), V(A.V), D(A.D), S(A.S), is_admissible(A.is_admissible) {}

  double H2::construction_relative_error(const randvec_t& randvec) {
    // verify diagonal matrix block constructions at the leaf level.
    double error = 0;
    double dense_norm = 0;
    int num_nodes = pow(2, height);
    for (int i = 0; i < num_nodes; ++i) {
      for (int j = 0; j < num_nodes; ++j) {
        if (is_admissible.exists(i, j, height) && !is_admissible(i, j, height)) {
          int slice = N / num_nodes;
          Matrix actual = Hatrix::generate_laplacend_matrix(randvec, slice, slice,
                                                            slice * i, slice * j, PV);
          Matrix expected = D(i, j, height);
          error += pow(norm(actual - expected), 2);
          dense_norm += pow(norm(actual), 2);
        }
      }
    }

    // regenerate off-diagonal blocks and test for correctness.
    for (int level = height; level > 0; --level) {
      int num_nodes = pow(2, level);
      int slice = N / num_nodes;

      for (int row = 0; row < num_nodes; ++row) {
        for (int col = 0; col < num_nodes; ++col) {
          if (is_admissible.exists(row, col, level) && is_admissible(row, col, level)) {
            Matrix Ubig = get_Ubig(row, level);
            Matrix Vbig = get_Vbig(col, level);
            int block_nrows = Ubig.rows;
            int block_ncols = Vbig.rows;
            Matrix expected_matrix = matmul(matmul(Ubig, S(row, col, level)), Vbig, false, true);
            Matrix actual_matrix = Hatrix::generate_laplacend_matrix(randvec, block_nrows,
                                                                     block_ncols,
                                                                     row * slice, col * slice, PV);

            dense_norm += pow(norm(actual_matrix), 2);
            error += pow(norm(expected_matrix - actual_matrix), 2);
          }
        }
      }
    }

    return std::sqrt(error / dense_norm);
  }

  void H2::print_structure() {
    actually_print_structure(height);
  }

  Matrix A1_global;

  void H2::factorize(const randvec_t &randpts) {
    // For verification of A1 matrix.
    A1_global = generate_identity_matrix(N, N);
    auto dim_offsets = generate_offsets(*this, 1);
    auto A1_global_splits = A1_global.split(dim_offsets, dim_offsets);

    for (int i = 0; i < dim_offsets.size(); ++i) {
      std::cout <<  dim_offsets[i] <<  " ";
    }
    std::cout << std::endl;

    int64_t level = height;
    RowColLevelMap<Matrix> F;

    for (; level > 0; --level) {
      int num_nodes = pow(2, level);
      bool is_all_dense_level = false;
      for (int i = 0; i < num_nodes; ++i) {
        if (!U.exists(i, level)) {
          is_all_dense_level = true;
        }
      }
      if (is_all_dense_level) {
        break;
      }

      for (int64_t block = 0; block < num_nodes; ++block) {
        // Assume that the block size for this level is the number of rows in the bases.
        if (!U.exists(block, level)) { continue; }
        int64_t block_size = U(block, level).rows;
        // Step 0: Recompress fill-ins on the off-diagonals.
        if (false) {
          {
            // Compress fill-ins on the same row as the <block,level> pair.
            Matrix row_concat(block_size, 0);
            std::vector<int64_t> VN1_col_splits;
            bool found_row_fill_in = false;
            for (int j = 0; j < num_nodes; ++j) {
              if (F.exists(block, j, level)) {
                found_row_fill_in = true;
                break;
              }
            }

            if (found_row_fill_in) {
              for (int j = 0; j < num_nodes; ++j) {
                if (is_admissible.exists(block, j, level) && is_admissible(block, j, level)) {
                  row_concat = concat(row_concat, matmul(U(block, level),
                                                         S(block, j, level)), 1);
                  if (F.exists(block, j, level)) {
                    Matrix Fp = matmul(F(block, j, level), V(j, level), false, true);
                    row_concat = concat(row_concat, Fp, 1);
                  }
                }
              }

              Matrix UN1, _SN1, _VN1T; double error;
              std::tie(UN1, _SN1, _VN1T, error) = truncated_svd(row_concat, rank);

              for (int j = 0; j < num_nodes; ++j) {
                if (is_admissible.exists(block, j, level) && is_admissible(block, j, level)) {
                  Matrix r_block_j = matmul(UN1, U(block, level), true, false);
                  Matrix Sbar_block_j = matmul(r_block_j, S(block, j, level));

                  Matrix SpF(rank, rank);
                  if (F.exists(block, j, level)) {
                    Matrix Fp = matmul(F(block, j, level), V(j, level), false, true);

                    SpF = matmul(matmul(UN1, Fp, true, false), V(j, level));
                    Sbar_block_j = Sbar_block_j + SpF;
                  }

                  S.erase(block, j, level);
                  S.insert(block, j, level, std::move(Sbar_block_j));

                  if (F.exists(block, j, level)) {
                    F.erase(block, j, level);
                  }
                }
              }
            }
          }

          {
            // Compress fill-ins on the same col as the <block, level> pair.
            Matrix col_concat(0, block_size);
            std::vector<int64_t> UN2_row_splits;
            bool found_col_fill_in = false;

            for (int i = 0; i < num_nodes; ++i) {
              if (F.exists(i, block, level)) {
                found_col_fill_in = true;
                break;
              }
            }

            if (found_col_fill_in) {
              for (int i = 0; i < num_nodes; ++i) {
                if (is_admissible.exists(i, block, level) && is_admissible(i, block, level)) {
                  col_concat = concat(col_concat, matmul(S(i, block, level),
                                                         transpose(V(block, level))),
                                      0);
                  if (F.exists(i, block, level)) {
                    Matrix Fp = matmul(U(i, level), F(i, block, level));
                    col_concat = concat(col_concat, Fp, 0);
                  }
                }
              }
            }
          }
        } // if (block > 0)

          // Step 1: Generate UF and VF blocks.
        Matrix UF = make_complement(U(block, level));
        Matrix VF = make_complement(V(block, level));

                // debug code to check the A1_global matrix
        if (level == 1 && block == 0) {
          int block_size = N / pow(2, level+1);
          int block_split = rank;
          auto D0_splits = SPLIT_DENSE(D(0,0,1), block_split, block_split);
          A1_global_splits[1 * 5 + 1] =  D0_splits[0];
          A1_global_splits[1 * 5 + 3] = D0_splits[1];
          A1_global_splits[3 * 5 + 1] = D0_splits[2];
          A1_global_splits[3 * 5 + 3] =  D0_splits[3];
        }

        if (level == 1 && block == 1) {
          int block_size = N / pow(2, level+1);
          int block_split = rank;
          auto D1_splits = SPLIT_DENSE(D(1,1,1), block_split, block_split);
          A1_global_splits[2 * 5 + 2] =  D1_splits[0];
          A1_global_splits[2 * 5 + 4] = D1_splits[1];
          A1_global_splits[4 * 5 + 2] = D1_splits[2];
          A1_global_splits[4 * 5 + 4] =  D1_splits[3];
        }


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
        int64_t row_split = block_size - row_rank, col_split = block_size - col_rank;

        // Step 3: Partial LU factorization
        std::cout << "performing partial LU: " << block << " l: " << level << std::endl;
        auto diagonal_splits = SPLIT_DENSE(D(block, block, level), row_split, col_split);
        Matrix& Dcc = diagonal_splits[0];
        lu(Dcc);

        // TRSM with other cc blocks on this row
        for (int j = block + 1; j < num_nodes; ++j) {
          if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) {
            int64_t col_split = block_size - V(j, level).cols;
            auto D_splits = SPLIT_DENSE(D(block, j, level), row_split, col_split);
            solve_triangular(Dcc, D_splits[0], Hatrix::Left, Hatrix::Lower, true);
          }
        }

        // TRSM with co blocks on this row.
        for (int j = 0; j < num_nodes; ++j) {
          if (is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) {
            int64_t col_split = block_size - V(j, level).cols;
            auto D_splits = SPLIT_DENSE(D(block, j, level), row_split, col_split);
            solve_triangular(Dcc, D_splits[1], Hatrix::Left, Hatrix::Lower, true);
          }
        }

        // TRSM with cc blocks on the column
        for (int i = block + 1; i < num_nodes; ++i) {
          if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) {
            int64_t row_split = block_size - U(i, level).cols;
            auto D_splits = SPLIT_DENSE(D(i, block, level), row_split, col_split);
            solve_triangular(Dcc, D_splits[0], Hatrix::Right, Hatrix::Upper, false);
          }
        }

        // TRSM with oc blocks on the column
        for (int i = 0; i < num_nodes; ++i) {
          if (is_admissible.exists(i, block, level) && !is_admissible(i, block, level)) {
            auto D_splits = SPLIT_DENSE(D(i, block, level),
                                        block_size - U(i, level).cols, col_split);
            solve_triangular(Dcc, D_splits[2], Hatrix::Right, Hatrix::Upper, false);
          }
        }

        if (level == 1 && block == 1) {
          int block_size = N / pow(2, level+1);
          int block_split = rank;
          auto D1_splits = SPLIT_DENSE(D(1,1,1), block_split, block_split);
          // A1_global_splits[2 * 5 + 2] =  D1_splits[0];
        }


        // Schur's compliment between oc and co blocks and update into oo block.
        for (int i = 0; i < num_nodes; ++i) {
          for (int j = 0; j < num_nodes; ++j) {
            if ((is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) &&
                (is_admissible.exists(i, block, level) && !is_admissible(i, block, level))) {
              auto lower_splits = SPLIT_DENSE(D(i, block, level),
                                              block_size - U(i, level).cols, col_split);
              auto right_splits = SPLIT_DENSE(D(block, j, level),
                                              row_split, block_size - V(j, level).cols);

              if (is_admissible.exists(i, j, level) && !is_admissible(i, j, level)) {
                auto reduce_splits = SPLIT_DENSE(D(i, j, level),
                                                 block_size - U(i, level).cols,
                                                 block_size - V(j, level).cols);
                matmul(lower_splits[2], right_splits[1], reduce_splits[3], false, false,
                       -1.0, 1.0);
              }
            }
          }
        }

        // Schur's compliment between co and cc block.
        for (int i = block + 1; i < num_nodes; ++i) {
          for (int j = 0; j < num_nodes; ++j) {
            if ((is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) &&
                (is_admissible.exists(i, block, level) && !is_admissible(i, block, level))) {
              auto lower_splits = SPLIT_DENSE(D(i, block, level), block_size -
                                              U(i, level).cols, col_split);
              auto right_splits = SPLIT_DENSE(D(block, j, level), row_split, block_size -
                                              V(j, level).cols);
              // Product and operands are dense.
              if (is_admissible.exists(i, j, level) && !is_admissible(i, j, level)) {
                auto reduce_splits = SPLIT_DENSE(D(i, j, level),
                                                 block_size - U(i, level).cols,
                                                 block_size - V(j, level).cols);
                matmul(lower_splits[0], right_splits[1], reduce_splits[1], false, false,
                       -1.0, 1.0);
              }
              // Operands are dense and product is fill-in.
              // The product is a (co; oo)-sized matrix.
              else {
                // Need to create a new fill-in block.
                if (!F.exists(i, j, level)) {
                  Matrix fill_in(block_size, rank);
                  auto fill_splits = fill_in.split(std::vector<int64_t>(1, block_size - rank),
                                                   {});
                  // Update the co block within the fill-in.
                  matmul(lower_splits[0], right_splits[1], fill_splits[0], false, false,
                         -1.0, 1.0);

                  // Update the oo block within the fill-in.
                  matmul(lower_splits[2], right_splits[1], fill_splits[1], false, false,
                         -1.0, 1.0);

                  F.insert(i, j, level, std::move(fill_in));
                }
                // Update an existing fill-in block.
                else {
                  Matrix &fill_in = F(i, j, level);
                  auto fill_splits = fill_in.split(std::vector<int64_t>(1, block_size - rank), {});
                  // Update the co block within the fill-in.
                  matmul(lower_splits[0], right_splits[1], fill_splits[0], false, false,
                         -1.0, 1.0);
                  // Update the oo block within the fill-in.
                  matmul(lower_splits[2], right_splits[1], fill_splits[1], false, false,
                         -1.0, 1.0);
                }
              }
            }
          }
        }

        // Schur's compliment between oc and cc blocks.
        for (int i = 0; i < num_nodes; ++i) {
          for (int j = block + 1; j < num_nodes; ++j) {
            if ((is_admissible.exists(block, j, level) && !is_admissible(block, j, level)) &&
                (is_admissible.exists(i, block, level) && !is_admissible(i, block, level))) {
              auto lower_splits = SPLIT_DENSE(D(i, block, level), block_size -
                                              U(i, level).cols, col_split);
              auto right_splits = SPLIT_DENSE(D(block, j, level), row_split, block_size -
                                              V(j, level).cols);
              if (is_admissible.exists(i, j, level) && !is_admissible(i, j, level)) {
                auto reduce_splits = SPLIT_DENSE(D(i, j, level),
                                                 block_size - U(i, level).cols,
                                                 block_size - V(j, level).cols);
                matmul(lower_splits[2], right_splits[0], reduce_splits[2],
                       false, false, -1.0, 1.0);
              }
              else {
                // Schur's compliement between oc and cc blocks where a new fill-in is created.
                // The product is a (oc, oo)-sized block.
                if (!F.exists(i, j, level)) {
                  Matrix fill_in(rank, block_size);
                  auto fill_splits = fill_in.split({},
                                                   std::vector<int64_t>(1, block_size - rank));
                  // Update the oc block within the fill-ins.
                  matmul(lower_splits[2], right_splits[0], fill_splits[0], false, false,
                         -1.0, 1.0);
                  // Update the oo block within the fill-ins.
                  matmul(lower_splits[2], right_splits[1], fill_splits[1], false, false,
                         -1.0, 1.0);
                  F.insert(i, j, level, std::move(fill_in));
                }
                else {
                  Matrix& fill_in = F(i, j, level);
                  auto fill_splits = fill_in.split({},
                                                   std::vector<int64_t>(1, block_size - rank));
                  // Update the oc block within the fill-ins.
                  matmul(lower_splits[2], right_splits[0], fill_splits[0], false, false,
                         -1.0, 1.0);
                  // Update the oo block within the fill-ins.
                  matmul(lower_splits[2], right_splits[1], fill_splits[1], false, false,
                         -1.0, 1.0);
                }
              }
            }
          }
        }
      } // for (block=0; block < num_nodes; ++block)

        // Merge the unfactorized parts.
      int64_t parent_level = level - 1;
      for (int i = 0; i < int(pow(2, parent_level)); ++i) {
        for (int j = 0; j < int(pow(2, parent_level)); ++j) {
          if (is_admissible.exists(i, j, parent_level) && !is_admissible(i, j, parent_level)) {
            std::vector<int64_t> i_children({i * 2, i * 2 + 1}), j_children({j * 2, j * 2 + 1});
            Matrix D_unelim(rank*2, rank*2);
            auto D_unelim_splits = D_unelim.split(2, 2);

            for (int ic1 = 0; ic1 < 2; ++ic1) {
              for (int jc2 = 0; jc2 < 2; ++jc2) {
                int64_t c1 = i_children[ic1], c2 = j_children[jc2];
                if (!U.exists(c1, level)) { continue; }

                int64_t block_size = U(c1, level).rows;

                if (!is_admissible(c1, c2, level)) {
                  std::cout << "c1: " << c1 << " c2: " << c2 << " l: " <<  level << std::endl;
                  auto D_splits = SPLIT_DENSE(D(c1, c2, level), block_size-rank, block_size-rank);
                  D_unelim_splits[ic1 * 2 + jc2] = D_splits[3];
                }
                else {
                  D_unelim_splits[ic1 * 2 + jc2] = S(c1, c2, level);
                }
              }
            }

            D.insert(i, j, parent_level, std::move(D_unelim));
          }
        }
      }
    } // for (int level=height; level > 0; --level)

    int64_t last_nodes = pow(2, level);
    for (int d = 0; d < last_nodes; ++d) {
      lu(D(d, d, level));
      for (int j = d+1; j < last_nodes; ++j) {
        solve_triangular(D(d, d, level), D(d, j, level), Hatrix::Left, Hatrix::Lower, true);
      }
      for (int i = d+1; i < last_nodes; ++i) {
        solve_triangular(D(d, d, level), D(i, d, level), Hatrix::Right, Hatrix ::Upper, false);
      }

      for (int i = d+1; i < last_nodes; ++i) {
        for (int j = d+1; j < last_nodes; ++j) {
          matmul(D(i, d, level), D(d, j, level), D(i, j, level), false, false, -1.0, 1.0);
        }
      }
    }
  }

  Matrix H2::solve(Matrix& b) {
    int64_t level = height;
    int64_t c_size, offset, rhs_offset = 0;
    Matrix x(b);
    std::vector<Matrix> x_splits;

    // Design philosophy: when working with permuted vectors in a loop, make a copy
    // of the working vector before starting actual work on it. Then copy into
    // that local copy for all subsequent operations. Finally copy out the copied
    // local vector into the overall global vector.

    // Forward
    for (; level > 0; --level) {
      int64_t num_nodes = pow(2, level);
      bool lr_exists = false;
      for (int block = 0; block < num_nodes; ++block) {
        if (U.exists(block, level)) {
          lr_exists = true;
        }
      }
      if (!lr_exists) { break; }

      // Contrary to the paper, multiply all the UF from the left side before
      // the forward substitution starts.
      for (int block = 0; block < num_nodes; ++block) {
        Matrix& Uo = U(block, level);
        Matrix U_F = make_complement(Uo);
        int64_t block_size = Uo.rows;
        c_size = block_size - rank;
        offset = rhs_offset + block * block_size;

        Matrix temp(block_size, 1);
        for (int i = 0; i < block_size; ++i) {
          temp(i, 0) = x(offset + i, 0);
        }
        Matrix product = matmul(U_F, temp, true);
        for (int i = 0; i < block_size; ++i) {
          x(offset + i, 0) = product(i, 0);
        }
      }

      // Copy part of the workspace vector into x_level so that we can only split
      // and work with the parts that correspond to the permuted vector that need
      // to be worked upon.
      Matrix x_level(N - rhs_offset, 1);
      for (int i = 0; i < x_level.rows; ++i) {
        x_level(i, 0) = x(rhs_offset + i, 0);
      }
      auto x_level_splits = x_level.split(num_nodes, 1);

      // Forward with cc blocks.
      for (int block = 0; block < num_nodes; ++block) {
        Matrix& Uo = U(block, level);
        Matrix& Vo = V(block, level);
        Matrix x_block(x_level_splits[block]);

        int64_t block_size = Uo.rows;

        c_size = block_size - rank;

        auto block_splits = SPLIT_DENSE(D(block, block, level), c_size, c_size);
        auto x_block_splits = x_block.split(std::vector<int64_t>(1, c_size), {});

        solve_triangular(block_splits[0], x_block_splits[0], Hatrix::Left, Hatrix::Lower, true);
        matmul(block_splits[2], x_block_splits[0], x_block_splits[1], false, false, -1.0, 1.0);

        // copy back x_block into the right place in x_level
        for (int i = 0; i < block_size; ++i) {
          x_level(block * block_size + i, 0) = x_block(i, 0);
        }

        // Forward with the big C block on the lower part. These are the dense blocks
        // that exist below the diagonal block.
        for (int irow = block+1; irow < num_nodes; ++irow) {
          if (is_admissible.exists(irow, block, level) && !is_admissible(irow, block, level)) {
            int64_t block_split = block_size - U(irow, level).cols;
            auto lower_splits = D(irow, block, level).split({},
                                                            std::vector<int64_t>(1,
                                                                                 block_split));

            Matrix x_block(x_level_splits[block]);
            auto x_block_splits = x_block.split(std::vector<int64_t>(1, block_split), {});

            Matrix x_irow(x_level_splits[irow]);
            matmul(lower_splits[0], x_block_splits[0], x_irow, false, false, -1.0, 1.0);
            for (int64_t i = 0; i < block_size; ++i) {
              x_level(irow * block_size + i, 0) = x_irow(i, 0);
            }
          }
        }

        // Forward with the oc parts of the block that are actually in the upper
        // part of the matrix.
        for (int irow = 0; irow < block; ++irow) {
          if (is_admissible.exists(irow, block, level) &&
              !is_admissible(irow, block, level)) {
            int64_t block_split = block_size - V(irow, level).cols;
            auto top_splits = SPLIT_DENSE(D(irow, block, level), block_split, block_split);

            Matrix x_irow(x_level_splits[irow]), x_block(x_level_splits[block]);

            auto x_irow_splits = x_irow.split(std::vector<int64_t>(1, block_split), {});
            auto x_block_splits = x_block.split(std::vector<int64_t>(1, block_split), {});

            matmul(top_splits[2], x_block_splits[0], x_irow_splits[1],
                   false, false, -1.0, 1.0);

            for (int i = 0; i < block_size; ++i) {
              x_level(irow * block_size + i, 0) = x_irow(i, 0);
            }
          }
        }
      }

      for (int i = 0; i < x_level.rows; ++i) {
        x(rhs_offset + i, 0) = x_level(i, 0);
      }

      rhs_offset = permute_forward(x, level, rhs_offset);
    }

    // Work with L0 and U0
    x_splits = x.split(std::vector<int64_t>(1, rhs_offset), {});
    Matrix x_last(x_splits[1]);
    int64_t last_nodes = pow(2, level);
    auto x_last_splits = x_last.split(last_nodes, 1);

    for (int i = 0; i < last_nodes; ++i) {
      for (int j = 0; j < i; ++j) {
        matmul(D(i, j, level), x_last_splits[j], x_last_splits[i], false, false, -1.0, 1.0);
      }
      solve_triangular(D(i, i, level), x_last_splits[i], Hatrix::Left, Hatrix::Lower, true);
    }

    for (int i = last_nodes-1; i >= 0; --i) {
      for (int j = last_nodes-1; j > i; --j) {
        matmul(D(i, j, level), x_last_splits[j], x_last_splits[i], false, false, -1.0, 1.0);
      }
      solve_triangular(D(i, i, level), x_last_splits[i], Hatrix::Left, Hatrix::Upper, false);
    }
    x_splits[1] = x_last;

    level++;
    // Backward
    for (; level <= height; ++level) {
      int64_t num_nodes = pow(2, level);

      bool lr_exists = false;
      for (int block = 0; block < num_nodes; ++block) {
        if (V.exists(block, level)) {
          lr_exists = true;
        }
      }
      if (!lr_exists) { break; }

      rhs_offset = permute_backward(x, level, rhs_offset);

      Matrix x_level(N - rhs_offset, 1);
      for (int i = 0; i < x_level.rows; ++i) {
        x_level(i, 0) = x(rhs_offset + i, 0);
      }
      auto x_level_splits = x_level.split(num_nodes, 1);

      // Upper triangle solve of the cc blocks.
      for (int block = num_nodes-1; block >= 0; --block) {
        Matrix& Vo = V(block, level);
        int64_t block_size = Vo.rows;
        int64_t c_size = block_size - rank;
        // Apply co block part of the dense blocks to the corresponding parts
        // in the RHS.
        for (int left_col = block-1; left_col >= 0; --left_col) {
          if (is_admissible.exists(block, left_col, level) &&
              !is_admissible(block, left_col, level)) {
            int64_t block_split = block_size - V(left_col, level).cols;
            auto left_splits = SPLIT_DENSE(D(block, left_col, level), block_split, block_split);

            Matrix x_block(x_level_splits[block]), x_left_col(x_level_splits[left_col]);
            auto x_block_splits = x_block.split(std::vector<int64_t>(1, block_split), {});
            auto x_left_col_splits = x_left_col.split(std::vector<int64_t>(1, block_split), {});

            matmul(left_splits[1], x_left_col_splits[1], x_block_splits[0], false, false,
                   -1.0, 1.0);
            for (int i = 0; i < block_size; ++i) {
              x_level(left_col * block_size + i, 0) = x_left_col(i, 0);
            }
          }
        }

        // Apply c block present on the right of this diagonal block.
        for (int right_col = num_nodes-1; right_col > block; --right_col) {
          if (is_admissible.exists(block, right_col, level) &&
              !is_admissible(block, right_col, level)) {
            int64_t block_split = block_size - V(block, level).cols;
            auto right_splits = D(block, right_col,
                                  level).split(std::vector<int64_t>(1,
                                                                    block_split),
                                               {});
            Matrix x_block(x_level_splits[block]);
            auto x_block_splits = x_block.split(std::vector<int64_t>(1, block_split), {});

            matmul(right_splits[0], x_level_splits[right_col], x_block_splits[0], false, false,
                   -1.0, 1.0);
            for (int i = 0; i < block_size; ++i) {
              x_level(right_col * block_size + i, 0) = x_block(i, 0);
            }
          }
        }

        Matrix x_block(x_level_splits[block]);

        auto block_splits = SPLIT_DENSE(D(block, block, level), c_size, c_size);
        auto x_block_splits = x_block.split(std::vector<int64_t>(1, c_size), {});
        matmul(block_splits[1], x_block_splits[1], x_block_splits[0], false, false, -1.0, 1.0);
        solve_triangular(block_splits[0], x_block_splits[0], Hatrix::Left, Hatrix::Upper, false);

        // copy back x_block into the right place in x_level
        for (int i = 0; i < block_size; ++i) {
          x_level(block * block_size + i, 0) = x_block(i, 0);
        }
      }

      for (int i = 0; i < x_level.rows; ++i) {
        x(rhs_offset + i, 0) = x_level(i, 0);
      }

      // Multiply VF with the respective block in x
      for (int block = num_nodes-1; block >= 0; --block) {
        Matrix& Vo = V(block, level);
        int64_t block_size = Vo.rows;
        c_size = block_size - rank;
        offset = rhs_offset + block * block_size;

        Matrix temp(block_size, 1);
        for (int i = 0; i < block_size; ++i) {
          temp(i, 0) = x(offset + i, 0);
        }

        auto V_F = make_complement(Vo);
        Matrix product = matmul(V_F, temp);
        for (int i = 0; i < block_size; ++i) {
          x(offset + i, 0) = product(i, 0);
        }
      }
    }

    return x;
  }

} // namespace Hatrix

// Build an array of all the UF matrices starting with the leaf level
// progressing toward upper levels.
std::vector<Hatrix::Matrix> generate_UF_chain(Hatrix::H2& A) {
  std::vector<Hatrix::Matrix> U_F;

  for (int level = 0; level <= A.height; ++level) {
    int num_nodes = pow(2, level);
    for (int block = 0; block < num_nodes; ++block) {
      Matrix UF_full = generate_identity_matrix(A.N, A.N);
      if (A.U.exists(block, level)) {
        std::vector<int64_t> dim_offsets = generate_offsets(A, level);
        int block_size = A.U(block, level).rows;
        Matrix UF_block = make_complement(A.U(block, level));

        auto UF_full_splits = UF_full.split(dim_offsets, dim_offsets);
        auto UF_block_splits = SPLIT_DENSE(UF_block, block_size - A.U(block, level).cols,
                                           block_size - A.U(block, level).cols);


        int permuted_nblocks = dim_offsets.size() + 1;
        int level_offset = A.height - level;

        int prow = block + level_offset;
        int pcol = block + level_offset;

        UF_full_splits[prow * permuted_nblocks + pcol] = UF_block_splits[0];
        UF_full_splits[(prow + num_nodes) * permuted_nblocks + pcol] = UF_block_splits[2];
        UF_full_splits[prow * permuted_nblocks + (pcol + num_nodes)] = UF_block_splits[1];
        UF_full_splits[(prow + num_nodes) * permuted_nblocks + pcol + num_nodes] =
          UF_block_splits[3];
      }
      U_F.push_back(UF_full);
    }
  }

  return U_F;
}

std::vector<Hatrix::Matrix> generate_VF_chain(Hatrix::H2& A) {
  std::vector<Hatrix::Matrix> V_F;

  for (int level = 0; level <= A.height; ++level) {
    int num_nodes = pow(2, level);
    for (int block = 0; block < num_nodes; ++block) {
      auto VF_full = generate_identity_matrix(A.N, A.N);
      if (A.V.exists(block, level)) {
        std::vector<int64_t> dim_offsets = generate_offsets(A, level);
        int block_size = A.V(block, level).rows;
        Matrix VF_block = transpose(make_complement(A.V(block, level)));

        auto VF_full_splits = VF_full.split(dim_offsets, dim_offsets);
        auto VF_block_splits = SPLIT_DENSE(VF_block, block_size - A.V(block, level).cols,
                                           block_size - A.V(block, level).cols);


        int permuted_nblocks = dim_offsets.size() + 1;
        int level_offset = A.height - level;

        int prow = block + level_offset;
        int pcol = block + level_offset;

        VF_full_splits[prow * permuted_nblocks + pcol] = VF_block_splits[0];
        VF_full_splits[(prow + num_nodes) * permuted_nblocks + pcol] = VF_block_splits[2];
        VF_full_splits[prow * permuted_nblocks + (pcol + num_nodes)] = VF_block_splits[1];
        VF_full_splits[(prow + num_nodes) * permuted_nblocks + pcol + num_nodes] =
          VF_block_splits[3];

      }

      V_F.push_back(VF_full);
    }
  }

  return V_F;
}

std::vector<Matrix> generate_L_chain(Hatrix::H2& A) {
  std::vector<Matrix> L;

  for (int level = 0; level <= A.height; ++level) {
    int num_nodes = pow(2, level);
    for (int block = 0; block < num_nodes; ++block) {
      Matrix L_block = generate_identity_matrix(A.N, A.N);
      if (A.U.exists(block, level)) {
        auto dim_offsets = generate_offsets(A, level);
        auto L_block_splits = L_block.split(dim_offsets, dim_offsets);
        int level_offset = A.height - level;
        int permuted_nblocks = dim_offsets.size() + 1;
        int prow = block + level_offset;

        for (int j = 0; j <= block; ++j) {
          if (A.is_admissible.exists(block, j, level) && !A.is_admissible(block, j, level)) {
            int block_split = A.U(block, level).rows - A.U(block, level).cols;
            int pcol = j + level_offset;

            auto D_splits = SPLIT_DENSE(A.D(block, j, level), block_split, block_split);

            if (block == j) {
              L_block_splits[prow * permuted_nblocks + pcol] = lower(D_splits[0]);
            }
            else {
              L_block_splits[prow * permuted_nblocks + pcol] = D_splits[0];
            }

            L_block_splits[(prow + num_nodes) * permuted_nblocks + pcol] = D_splits[2];
          }
        }


        // Copy oc parts belonging to the 'upper' parts of the matrix
        for (int i = 0; i < block; ++i) {
          if (A.U.exists(i, level)) {
            if (A.is_admissible.exists(i, block, level) &&
                !A.is_admissible(i, block, level)) {
              int block_split = A.U(i, level).rows - A.U(i, level).cols;
              int pcol = i + level_offset;
              auto D_splits = SPLIT_DENSE(A.D(i, block, level), block_split, block_split);

              L_block_splits[(prow + num_nodes) * permuted_nblocks + pcol] = D_splits[2];
            }
          }
        }
      }

      L.push_back(L_block);
    }
  }

  return L;
}

std::vector<Matrix> generate_U_chain(Hatrix::H2& A) {
  std::vector<Matrix> U;

  for (int level = 0; level <= A.height; ++level) {
    int num_nodes = pow(2, level);

    for (int block = 0; block < num_nodes; ++block) {
      Matrix U_block = generate_identity_matrix(A.N, A.N);
      if (A.V.exists(block, level)) {
        auto dim_offsets = generate_offsets(A, level);
        auto U_block_splits = U_block.split(dim_offsets, dim_offsets);
        int level_offset = A.height - level;
        int permuted_nblocks = dim_offsets.size() + 1;
        int pcol = block + level_offset;
        for (int i = 0; i <= block; ++i) {
          if (A.is_admissible.exists(i, block, level) && !A.is_admissible(i, block, level)) {
            int block_split = A.U(block, level).rows - A.U(block, level).cols;
            int prow = i + level_offset;

            auto D_splits = SPLIT_DENSE(A.D(i, block, level), block_split, block_split);

            // Copy the cc parts
            if (block == i) {
              U_block_splits[prow * permuted_nblocks + pcol] = upper(D_splits[0]);
            }
            else {
              U_block_splits[prow * permuted_nblocks + pcol] = D_splits[0];
            }

            // Copy the co parts
            U_block_splits[prow * permuted_nblocks + pcol + num_nodes] = D_splits[1];
          }
        }

        for (int j = 0; j < block; ++j) {
          if (A.V.exists(j, level)) {
            if (A.is_admissible.exists(block, j, level) && !A.is_admissible(block, j, level)) {
              int block_split = A.V(j, level).rows - A.V(j, level).cols;
              int prow = block + level_offset;
              int pcol = j + level_offset;
              auto D_splits = SPLIT_DENSE(A.D(block, j, level), block_split, block_split);
              U_block_splits[prow * permuted_nblocks + pcol + num_nodes] = D_splits[1];
            }
          }
        }
      }

      U.push_back(U_block);
    }
  }

  return U;
}

// Generate L0 by collecting blocks that correspond to L0 in the factorized matrix A.
Hatrix::Matrix generate_L0_permuted(H2& A) {
  Matrix L0 = generate_identity_matrix(A.N, A.N);
  std::vector<int64_t> top_level_offsets = generate_top_level_offsets(A);

  int level = A.height;
  for (; level > 0; --level) {
    if (!A.U.exists(0, level)) {
      break;
    }
  }

  auto L0_splits = L0.split(top_level_offsets, top_level_offsets);
  int num_nodes = pow(2, level);
  int level_offset = A.height - level;
  int permuted_nblocks = num_nodes + level_offset;

  for (int i = 0; i < num_nodes; ++i) {
    for (int j = 0; j <= i; ++j) {
      int prow = i + level_offset;
      int pcol = j + level_offset;
      if (i == j) {
        L0_splits[prow * permuted_nblocks + pcol] = lower(A.D(i, j, level));
      }
      else {
        L0_splits[prow * permuted_nblocks + pcol] = A.D(i, j, level);
      }
    }
  }

  return L0;
}

Hatrix::Matrix generate_U0_permuted(H2& A) {
  Matrix U0 = generate_identity_matrix(A.N, A.N);
  std::vector<int64_t> top_level_offsets = generate_top_level_offsets(A);

  int level = A.height;
  for (; level > 0; --level) {
    if (!A.V.exists(0, level)) {
      break;
    }
  }

  auto U0_splits = U0.split(top_level_offsets, top_level_offsets);
  int num_nodes = pow(2, level);
  int level_offset = A.height - level;
  int permuted_nblocks = num_nodes + level_offset;

  for (int i = 0; i < num_nodes; ++i) {
    for (int j = i;  j < num_nodes; ++j) {
      int prow = i + level_offset;
      int pcol = j + level_offset;

      if (i == j) {
        U0_splits[prow * permuted_nblocks + pcol] = upper(A.D(i, j, level));
      }
      else {
        U0_splits[prow * permuted_nblocks + pcol] = A.D(i, j, level);
      }
    }
  }

  return U0;
}

Matrix chained_product(std::vector<Matrix>& U_F,
                       std::vector<Matrix>& L,
                       Matrix& L0,
                       Matrix& U0,
                       std::vector<Matrix>& U,
                       std::vector<Matrix>& V_F,
                       H2& A) {
  Matrix product = generate_identity_matrix(A.N, A.N);

  for (int level = A.height; level > 0; --level) {
    if (A.U.exists(0, level)) {
      int num_nodes = pow(2, level);
      for (int i = 0; i < num_nodes; ++i) {
        int index = num_nodes + i - 1;
        product = matmul(product, U_F[index]);
        product = matmul(product, L[index]);
      }
    }
    else {
      break;
    }
  }

  product = matmul(product, L0);
  product = matmul(product, U0);

  for (int level = 1; level <= A.height; ++level) {
    if (A.V.exists(0, level)) {
      int num_nodes = pow(2, level);
      for (int i = num_nodes-1; i >= 0; --i) {
        int index = num_nodes + i - 1;
        product = matmul(product, U[index]);
        product = matmul(product, V_F[index]);
      }
    }
    else {
      break;
    }
  }

  return product;
}

Matrix unpermute_matrix(Matrix permuted, H2& A) {
  Matrix unpermuted(permuted);

  return unpermuted;
}

Matrix verify_A0(Matrix& L0, Matrix& U0, H2& A) {
  Matrix A0_actual = generate_identity_matrix(A.N, A.N);
  std::vector<int64_t> top_level_offsets = generate_top_level_offsets(A);

  auto A0_actual_splits = A0_actual.split(top_level_offsets, top_level_offsets);
  auto A0_expected = matmul(L0, U0);

  Matrix temp(A.rank * 2, A.rank * 2);
  auto temp_split = temp.split(2, 2);
  temp_split[1] = A.S(0, 1, 1);
  temp_split[2] = A.S(1, 0, 1);

  int block_split = A.U(0, 2).rows - A.U(0, 2).cols;

  auto D0_split = SPLIT_DENSE(A.D(0, 0, 1), A.rank, A.rank);
  auto D1_split = SPLIT_DENSE(A.D(1, 1, 1), A.rank, A.rank);

  temp_split[0] = D0_split[3];
  temp_split[3] = D1_split[3];

  A0_actual_splits[8] = temp;

  std::cout << "A0 verification norm -> " << Hatrix::norm(A0_expected - A0_actual) << std::endl;
  return A0_actual;
}

Matrix verify_A1(Matrix& A0, std::vector<Matrix>& L,
               std::vector<Matrix>& U,
               std::vector<Matrix>& U_F,
               std::vector<Matrix>& V_F, H2& A) {
  // Check if the product L1 x L2 x A0 x U2 x U1 is equal to the corresponding parts
  // in the factorized matrix.
  std::vector<int64_t> level1_offsets = generate_offsets(A, 1);
  auto D0_split = SPLIT_DENSE(A.D(0, 0, 1), A.rank, A.rank);
  auto D1_split = SPLIT_DENSE(A.D(1, 1, 1), A.rank, A.rank);

  auto L_side = matmul(matmul(matmul(U_F[1], L[1]), U_F[2]), L[2]);
  auto U_side = matmul(matmul(matmul(U[2], V_F[2]), U[1]), V_F[1]);
  auto A1_actual = matmul(matmul(L_side, A0), U_side);

  auto dim_offsets = generate_offsets(A, 1);
  auto A1_global_splits = A1_global.split(dim_offsets, dim_offsets);
  auto A0_splits = A0.split(dim_offsets, dim_offsets);

  auto A1_01 = matmul(matmul(A.U(0, 1), A.S(0, 1, 1)), A.V(1, 1), false, true);
  auto A1_01_splits = A1_01.split(2, 2);

  A1_global_splits[1 * 5 + 2] = A1_01_splits[0];
  A1_global_splits[1 * 5 + 4] = A1_01_splits[1];
  A1_global_splits[3 * 5 + 2] = A1_01_splits[2];
  A1_global_splits[3 * 5 + 4] = A1_01_splits[3];

  // Set these blocks here because they represent the pre-LU factorized last block.

  auto A1_10 = matmul(matmul(A.U(1, 1), A.S(1, 0, 1)), A.V(0, 1), false, true);
  auto A1_10_splits = A1_10.split(2, 2);

  A1_global_splits[2 * 5 + 1] = A1_10_splits[0];
  A1_global_splits[4 * 5 + 1] = A1_10_splits[1];
  A1_global_splits[2 * 5 + 3] = A1_10_splits[2];
  A1_global_splits[4 * 5 + 3] = A1_10_splits[3];

  (A1_actual - A1_global).print();

  std::cout << "norm: " << norm(A1_actual - A1_global) << std::endl;

  return A1_actual;
}

Matrix verify_A2(Matrix& A0, Matrix& A1, std::vector<Matrix>& L,
                 std::vector<Matrix>& U,
                 std::vector<Matrix>& U_F,
                 std::vector<Matrix>& V_F, H2& A) {
  auto L_prod = matmul(matmul(matmul(L[3], L[4]), L[5]), L[6]);
  auto M = matmul(matmul(matmul(matmul(matmul(L_prod, A1), U[6]), U[5]), U[4]), U[3]);

  // (M - A1_global).print();

  return M;
}

Hatrix::Matrix verify_factorization(Hatrix::H2& A) {
  auto U_F = generate_UF_chain(A);
  auto V_F = generate_VF_chain(A);
  auto L = generate_L_chain(A);
  auto U = generate_U_chain(A);
  auto L0 = generate_L0_permuted(A);
  auto U0 = generate_U0_permuted(A);

  Matrix A_actual_permuted = chained_product(U_F, L, L0, U0, U, V_F, A);

  auto A0 = verify_A0(L0, U0, A);

  auto A1 = verify_A1(A0, L, U, U_F, V_F, A);

  verify_A2(A0, A1, L, U, U_F, V_F, A);

  return A_actual_permuted;
}

Matrix permutation_matrix(H2& A) {
  Matrix P(A.N, A.N);

  std::vector<int64_t> array(A.N);
  std::vector<int64_t> new_array(A.N);
  std::iota(array.begin(), array.end(), 0);

  for (int level = A.height; level > 1; --level) {
    if (A.U.exists(0, level)) {
      int block_size = A.U(0, level).rows;
      int rank = A.U(0, level).cols;
      int c_size = block_size - rank;
      int num_nodes = pow(2, level);

      int index, post_rank_index, new_index = 0;

      for (int b = 0; b < num_nodes; ++b) {
        index = b * c_size;
        new_index = index;

        for (int c = 0; c < c_size; ++c) {
          new_array[new_index] = array[index];
          ++index;
          ++new_index;
        }

        post_rank_index = index + rank;

        for (int i = 0; i < (A.N - (b+1) * block_size) + rank * b; ++i) {
          new_array[new_index] = array[post_rank_index];
          post_rank_index += 1;
          ++new_index;
        }

        for (int r = 0; r < rank; ++r) {
          new_array[new_index] = array[index];
          index += 1;
          new_index += 1;
        }

        array = new_array;
      }
    }
  }

  // Generate P according to new_array
  for (int i = 0; i < A.N; ++i) {
    P(i, new_array[i]) = 1.0;
  }

  return P;
}

Matrix permute_dense(Matrix& Adense, H2& A) {
  Matrix P_level = permutation_matrix(A);

  Matrix A_permuted = matmul(matmul(P_level, Adense), P_level, false, true);

  return A_permuted;
}

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
  PV = 1e-2 * (1 / pow(10, height));

  Hatrix::Matrix b = Hatrix::generate_random_matrix(N, 1);

  auto start_construct = std::chrono::system_clock::now();
  Hatrix::H2 A(randpts, N, rank, height, admis);
  Hatrix::H2 A_copy(A);
  double construct_error = A.construction_relative_error(randpts);
  auto stop_construct = std::chrono::system_clock::now();

  A.factorize(randpts);

  Hatrix::Matrix Adense = Hatrix::generate_laplacend_matrix(randpts, N, N, 0, 0, PV);
  auto A_actual_permuted = verify_factorization(A);

  auto Adense_permuted = permute_dense(Adense, A);
  // Adense_permuted.print();

  A_actual_permuted.print();
  (A_actual_permuted - Adense_permuted).print();
  double factorization_error = Hatrix::norm(A_actual_permuted - Adense) / Hatrix::norm(Adense);

  Hatrix::Matrix x = A.solve(b);
  Hatrix::Matrix x_solve = lu_solve(Adense, b);


  // std::cout << "X solve\n";
  // x_solve.print();
  // std::cout << "X-X_solve\n";
  // (x-x_solve).print();

  Hatrix::Context::finalize();

  double solve_error = Hatrix::norm(x - x_solve) / Hatrix::norm(x_solve);

  std::cout << "N= " << N << " rank= " << rank << " admis= " << admis << " leaf= "
            << int(N / pow(2, height))
            << " height=" << height <<  " const. error="
            << construct_error
            << " factorization error= " << factorization_error
            << " solve error=" << solve_error << std::endl;

}
