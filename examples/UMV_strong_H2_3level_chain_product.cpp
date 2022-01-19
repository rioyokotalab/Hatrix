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
    void solve_forward_level(Matrix& x_level, int level);
    void solve_backward_level(Matrix& x_level, int level);
    Matrix generate_U_transfer_matrix(Matrix& Ubig_child1, Matrix& Ubig_child2, int node,
                                      int block_size, const randvec_t& randpts, int level);
    Matrix generate_V_transfer_matrix(Matrix& Vbig_child1, Matrix& Vbig_child2, int node,
                                      int block_size, const randvec_t& randpts, int level);

  public:
    H2(const randvec_t& randpts, int64_t _N, int64_t _rank, int64_t _height,
             int64_t _admis);
    H2(const H2& A);
    double construction_relative_error(const randvec_t& randvec);
    void print_structure();
    void factorize(const randvec_t &randpts);
    Matrix solve(Matrix& b, int level);
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

  Matrix
  H2::generate_U_transfer_matrix(Matrix& Ubig_child1, Matrix& Ubig_child2, int node,
                                        int block_size, const randvec_t& randpts, int level) {
    // Shown as Alevel_node_plus on miro.
    Matrix col_block = generate_column_block(node, block_size, randpts, level);
    auto col_block_splits = col_block.split(2, 1);

    Matrix temp(Ubig_child1.cols + Ubig_child2.cols, col_block.cols);
    auto temp_splits = temp.split(2, 1);

    matmul(Ubig_child1, col_block_splits[0], temp_splits[0], true, false, 1, 0);
    matmul(Ubig_child2, col_block_splits[1], temp_splits[1], true, false, 1, 0);

    Matrix Utransfer, Si, Vi; double error;
    std::tie(Utransfer, Si, Vi, error) = truncated_svd(temp, rank);

    return Utransfer;
  }

  Matrix
  H2::generate_V_transfer_matrix(Matrix& Vbig_child1, Matrix& Vbig_child2, int node,
                                        int block_size, const randvec_t& randpts, int level) {
    Matrix row_block = generate_row_block(node, block_size, randpts, level);
    auto row_block_splits = row_block.split(1, 2);

    Matrix temp(row_block.rows, Vbig_child1.cols + Vbig_child2.cols);
    auto temp_splits = temp.split(1, 2);

    matmul(row_block_splits[0], Vbig_child1, temp_splits[0]);
    matmul(row_block_splits[1], Vbig_child2, temp_splits[1]);

    Matrix Ui, Si, Vtransfer; double error;
    std::tie(Ui, Si, Vtransfer, error) = truncated_svd(temp, rank);

    return transpose(Vtransfer);
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

      // Generate U transfer matrix.
      if (row_has_admissible_blocks(node, level)) {
        Matrix& Ubig_child1 = Uchild(child1, child_level);
        Matrix& Ubig_child2 = Uchild(child2, child_level);

        Matrix Utransfer = generate_U_transfer_matrix(Ubig_child1, Ubig_child2, node,
                                                      block_size, randpts, level);
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

        Matrix Vtransfer = generate_V_transfer_matrix(Vbig_child1, Vbig_child2, node,
                                                      block_size, randpts, level);
        V.insert(node, level, std::move(Vtransfer));

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

    // if (node == 2 && level == 2) {
    //   std::cout << "AFTER GENERATION Ubig23: " << norm(generate_identity_matrix(rank, rank) - matmul(Ubig_splits[0], Ubig_splits[0], true, false)) << std::endl;;
    //   std::cout << "AFTER GENERATION Ubig23: " << norm(generate_identity_matrix(rank, rank) - matmul(Ubig, Ubig, true, false)) << std::endl;;

    // }

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

    // std::cout << "CONSTRUCTION UPDATE block -> "
    //           << 1 << " nrm -> "
    //           << norm(generate_identity_matrix(rank, rank) -
    //                   matmul(U(1, 2), U(1, 2),
    //                          true, false)) << std::endl;
  }

  // Copy constructor.
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

  Matrix A2_expected, A1_expected;

  void H2::factorize(const randvec_t &randpts) {
    // For verification of A1 matrix.
    Matrix mat(200, 5);
    auto mat_split = mat.split(2, 1);
    auto Utransfer_splits = U(1, 2).split(2, 1);
    matmul(U(2, 3), Utransfer_splits[0], mat_split[0]);
    matmul(U(3, 3), Utransfer_splits[1], mat_split[1]);


    std::cout << "PRE FACTOR Ubig23: " << norm(generate_identity_matrix(rank, rank) - matmul(mat, mat, true, false)) << std::endl;;


    int64_t level = height;
    RowColLevelMap<Matrix> F;
    RowMap r, t;     // matrices for storing the updates for coupling blocks at each level.
    A2_expected = generate_identity_matrix(rank * 8, rank * 8);
    auto A2_expected_splits = A2_expected.split(4, 4);

    A1_expected = generate_identity_matrix(rank * 8, rank * 8);
    auto A1_expected_splits = A1_expected.split(4, 4);

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

      if (level == 2) {
        for (int i = 0; i < 4; ++i) {
          for (int j = 0; j < 4; ++j) {
            if (D.exists(i, j, level)) {
              A2_expected_splits[i * 4 + j] = D(i, j, level);
            }
          }
        }
      }

      for (int64_t block = 0; block < num_nodes; ++block) {
        // Assume that the block size for this level is the number of rows in the bases.
        if (!U.exists(block, level)) { continue; }
        int64_t block_size = U(block, level).rows;
        if (level == 2) {
          std::cout << "PRE CLUSTER UPDATE: block -> "
                    << block << " nrm -> "
                    << norm(generate_identity_matrix(rank, rank) -
                            matmul(U(block, level), U(block, level),
                                   true, false)) << std::endl;

        }
        // Step 0: Recompress fill-ins on the off-diagonals.
        if (block > 0) {
          {
            // Compress fill-ins on the same row as the <block,level> pair.
            Matrix row_concat(block_size, 0);
            std::vector<int64_t> VN1_col_splits;
            bool found_row_fill_in = false;
            for (int j = 0; j < block; ++j) {
              if (F.exists(block, j, level)) {
                found_row_fill_in = true;
                break;
              }
            }

            // if (found_row_fill_in || level != height) {
            if (found_row_fill_in) {
              // Recompress and update the basis on the same level.
              for (int j = 0; j < num_nodes; ++j) {
                if (is_admissible.exists(block, j, level) && is_admissible(block, j, level)) {
                  row_concat = concat(row_concat, matmul(U(block, level),
                                                         S(block, j, level)), 1);
                  if (F.exists(block, j, level)) {
                    Matrix Fp = matmul(F(block, j, level), V(j, level), false, true);
                    row_concat = concat(row_concat, Fp, 1);
                  }
                }
                else if (!is_admissible.exists(block, j, level)) { // if you have an off-diagonal dense block,
                  // std::cout << "include block: " << block << ", " << j << ", " << level << std::endl;
                  Matrix lr_block = Hatrix::generate_laplacend_matrix(randpts, block_size, block_size,
                                                                   block * block_size, j * block_size, PV);
                  row_concat = concat(row_concat, lr_block, 1);
                }
              }

              Matrix UN1, _SN1, _VN1T; double error;
              std::tie(UN1, _SN1, _VN1T, error) = truncated_svd(row_concat, rank);

              Matrix r_block = matmul(UN1, U(block, level), true, false);
              r.insert(block, std::move(r_block));

              U.erase(block, level);
              U.insert(block, level, std::move(UN1));
            }
          }

          {
            // Compress fill-ins on the same col as the <block, level> pair.
            Matrix col_concat(0, block_size);
            std::vector<int64_t> UN2_row_splits;
            bool found_col_fill_in = false;

            for (int i = 0; i < block; ++i) {
              if (F.exists(i, block, level)) {
                found_col_fill_in = true;
                break;
              }
            }

            // if (found_col_fill_in || level != height) {
            if (found_col_fill_in) {
              // Recompress and update the column on the same level.
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
                else if (!is_admissible.exists(i, block, level)) {
                  Matrix lr_block = Hatrix::generate_laplacend_matrix(randpts, block_size, block_size,
                                                                      i * block_size, block * block_size, PV);
                  col_concat = concat(col_concat, lr_block, 0);
                }
              }

              Matrix _UN2, _SN2, VN2T; double error;
              std::tie(_UN2, _SN2, VN2T, error)= truncated_svd(col_concat, rank);

              Matrix t_block = matmul(V(block, level), VN2T, true, true);

              t.insert(block, std::move(t_block));

              V.erase(block, level);
              V.insert(block, level, transpose(VN2T));
            }
          }
        } // if (block > 0)

        if (level == 2) {
          std::cout << "POST CLUSTER UPDATE block -> "
                    << block << " nrm -> "
                    << norm(generate_identity_matrix(rank, rank) -
                            matmul(U(block, level), U(block, level),
                                   true, false)) << std::endl;

        }

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
        int64_t row_split = block_size - row_rank, col_split = block_size - col_rank;

        // Step 3: Partial LU factorization
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

      int parent_level = level - 1;
      if (level == 3) {
        for (int b = 0; b < 4; ++b) {
          std::cout << "PRE UPDATE block -> "
                    << b << " nrm -> "
                    << norm(generate_identity_matrix(rank, rank) -
                            matmul(U(b, 2), U(b, 2),
                                   true, false)) << std::endl;
        }
      }

      // Update coupling matrices on this level and transfer matrices one level higher.
      for (int block = 0; block < num_nodes; ++block) {
        if (r.exists(block)) {
          Matrix &r_block = r(block);
          for (int j = 0; j < block; ++j) {
            if (is_admissible.exists(block, j, level) && is_admissible(block, j, level)) {
              Matrix Sbar_block_j = matmul(r_block, S(block, j, level));

              Matrix SpF(rank, rank);
              if (F.exists(block, j, level)) {
                // std::cout << "block: " << block << " j: " << j << " lvel: " << level << std::endl;
                Matrix Fp = matmul(F(block, j, level), V(j, level), false, true);
                SpF = matmul(matmul(U(block, level), Fp, true, false), V(j, level));
                Sbar_block_j = Sbar_block_j + SpF;
                F.erase(block, j, level);
              }
              S.erase(block, j, level);
              S.insert(block, j, level, std::move(Sbar_block_j));
            }
          }

          // int parent_node = block / 2;
          // int slice_index = block % 2;
          // if (U.exists(parent_node, parent_level)) {

          //   // if (block == 2 && level == 3) {
          //   //   auto Utransfer_splits = U(parent_node, parent_level).split(2, 1);
          //   //   auto Uhat_big_23 = matmul(Ubig_23, Utransfer_splits[0]);
          //   //   auto proj = matmul(U(2, 3), U(2, 3), true, false);
          //   //   std::cout << "ident: " << norm(generate_identity_matrix(rank, rank) - matmul(Uhat_big_23, Uhat_big_23, true, false)) << std::endl;;
          //   // }
          //   if (block == 2 && level == 3) {
          //     std::cout << "NORM OF NEW MATRIX: " << norm(generate_identity_matrix(rank, rank) - matmul(U(2, 3), U(2, 3), true, false)) << std::endl;
          //   }
          //   auto Utransfer_splits = U(parent_node, parent_level).split(2, 1);
          //   auto Utransfer_new_part = matmul(r_block, Utransfer_splits[slice_index]);
          //   Utransfer_splits[slice_index] = Utransfer_new_part;
          // }
          r.erase(block);
        }

        if (t.exists(block)) {
          Matrix& t_block = t(block);
          for (int i = 0; i < block; ++i) {
            if (is_admissible.exists(i, block, level) && is_admissible(i, block, level)) {
              Matrix Sbar_i_block = matmul(S(i, block, level), t_block);

              if (F.exists(i, block, level)) {
                // std::cout << "i-> " << i << " block-> " << block << " level-> " << level << std::endl;
                Matrix Fp = matmul(U(i, level), F(i, block, level));
                Matrix SpF = matmul(matmul(U(i, level), Fp, true, false), V(block, level));
                Sbar_i_block = Sbar_i_block + SpF;
                F.erase(i, block, level);
              }

              S.erase(i, block, level);
              S.insert(i, block, level, std::move(Sbar_i_block));
            }
          }

          // int parent_node = block / 2;
          // int slice_index = block % 2;
          // if (V.exists(parent_node, parent_level)) {
          //   auto Vtransfer_splits = V(parent_node, parent_level).split(2, 1);
          //   auto Vtransfer_new_part = matmul(t_block, Vtransfer_splits[slice_index]);
          //   Vtransfer_splits[slice_index] = Vtransfer_new_part;
          // }
          t.erase(block);
        }
      } // for (int block = 0; block < num_nodes; ++block)

      // Update transfer matrices.
      for (int block = 0; block < num_nodes; ++block) {
        int parent_node = block / 2;
        int slice_index = block % 2;


      }

      if (level == 3) {
        for (int i = 0; i < 4; ++i) {
          for (int j = 0; j < 4; ++j) {
            if (is_admissible(i, j, 2)) {
              A2_expected_splits[i * 4 + j] =
                matmul(matmul(U(i, 2), S(i, j, 2)), V(j, 2), false, true);
            }
          }
        }
      }

      if (level == 3) {
        for (int b = 0; b < 4; ++b) {
          std::cout << "POST UPDATE block -> "
                    << b << " nrm -> "
                    << norm(generate_identity_matrix(rank, rank) -
                            matmul(U(b, 2), U(b, 2),
                                   true, false)) << std::endl;
        }

      }


      // Merge the unfactorized parts.
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

    // Capture unfactorized A1 block.
    for (int i = 0; i < last_nodes; ++i) {
      for (int j = 0; j < last_nodes; ++j) {
        A1_expected_splits[(i + 2) * 4 + j + 2] = D(i, j, level);
      }
    }


    for (int d = 0; d < last_nodes; ++d) {
      lu(D(d, d, level));
      for (int j = d+1; j < last_nodes; ++j) {
        solve_triangular(D(d, d, level), D(d, j, level), Hatrix::Left, Hatrix::Lower, true);
      }
      for (int i = d+1; i < last_nodes; ++i) {
        solve_triangular(D(d, d, level), D(i, d, level), Hatrix::Right, Hatrix::Upper, false);
      }

      for (int i = d+1; i < last_nodes; ++i) {
        for (int j = d+1; j < last_nodes; ++j) {
          matmul(D(i, d, level), D(d, j, level), D(i, j, level), false, false, -1.0, 1.0);
        }
      }
    }

    // std::cout << "post factorization U transfer matrices\n";
    // for (int i = 0; i < 4; ++i) {
    //   std::cout << "i -> "
    //             << i << " "
    //             << norm(generate_identity_matrix(rank, rank) - matmul(U(i, 2), U(i, 2),
    //                                                                   true, false)) << "\n";
    // }

    // std::cout << "post factorization V transfer matrices\n";
    // for (int i = 0; i < 4; ++i) {
    //   std::cout << "i -> "
    //             << i << " "
    //             <<  norm(generate_identity_matrix(rank, rank) - matmul(V(i, 2), V(i, 2),
    //                                                                    true, false)) << "\n";
    // }
  }

  void H2::solve_forward_level(Matrix& x_level, int level) {
    int nblocks = pow(2, level);
    std::vector<Matrix> x_level_split = x_level.split(nblocks, 1);

    for (int block = 0; block < nblocks; ++block) {
      int block_size = U(block, level).rows;
      Matrix U_F = make_complement(U(block, level));
      Matrix prod = matmul(U_F, x_level_split[block], true);
      for (int64_t i = 0; i < block_size; ++i) {
        x_level(block * block_size + i, 0) = prod(i, 0);
      }
    }

    // forward substitution with cc blocks
    for (int block = 0; block < nblocks; ++block) {
      int block_size = U(block, level).rows;
      int c_size = block_size - rank;

      int64_t row_split = block_size - U(block, level).cols,
        col_split = block_size - V(block, level).cols;
      auto block_splits = SPLIT_DENSE(D(block, block, level), row_split, col_split);

      Matrix x_block(x_level_split[block]);
      auto x_block_splits = x_block.split(std::vector<int64_t>(1, row_split), {});

      solve_triangular(block_splits[0], x_block_splits[0], Hatrix::Left, Hatrix::Lower, true);
      matmul(block_splits[2], x_block_splits[0], x_block_splits[1], false, false, -1.0, 1.0);
      for (int64_t i = 0; i < block_size; ++i) {
        x_level(block * block_size + i, 0) = x_block(i, 0);
      }

      // Forward with the big c blocks on the lower part.
      for (int irow = block+1; irow < nblocks; ++irow) {
        if (is_admissible.exists(irow, block, level) && !is_admissible(irow, block, level)) {
          int64_t row_split = block_size - U(irow, level).cols;
          int64_t col_split = block_size - V(block, level).cols;
          auto lower_splits = D(irow, block, level).split({}, std::vector<int64_t>(1, row_split));

          Matrix x_block(x_level_split[block]), x_level_irow(x_level_split[irow]);
          auto x_block_splits = x_block.split(std::vector<int64_t>(1, col_split), {});

          matmul(lower_splits[0], x_block_splits[0], x_level_irow, false, false, -1.0, 1.0);
          for (int64_t i = 0; i < block_size; ++i) {
            x_level(irow * block_size + i, 0) = x_level_irow(i, 0);
          }
        }
      }

      // Forward with the oc parts of the block that are actually in the upper part of the matrix.
      for (int irow = 0; irow < block; ++irow) {
        if (is_admissible.exists(irow, block, level) && !is_admissible(irow, block, level)) {
          int64_t row_split = block_size - U(irow, level).cols;
          int64_t col_split = block_size - V(block, level).cols;
          auto top_splits = SPLIT_DENSE(D(irow, block, level), row_split, col_split);

          Matrix x_irow(x_level_split[irow]), x_block(x_level_split[block]);
          auto x_irow_splits = x_irow.split(std::vector<int64_t>(1, row_split), {});
          auto x_block_splits = x_block.split(std::vector<int64_t>(1, col_split), {});

          matmul(top_splits[2], x_block_splits[0], x_irow_splits[1], false, false, -1.0, 1.0);

          for (int64_t i = 0; i < block_size; ++i) {
            x_level(irow * block_size + i, 0) = x_irow(i, 0);
          }
        }
      }
    }
  }

  void H2::solve_backward_level(Matrix& x_level, int level) {
    int nblocks = pow(2, level);
    std::vector<Matrix> x_level_split = x_level.split(nblocks, 1);

    // backward substition using cc blocks
    for (int block = nblocks-1; block >= 0; --block) {
      int block_size = V(block, level).rows;
      int64_t row_split = block_size - U(block, level).cols,
        col_split = block_size - V(block, level).cols;
      auto block_splits = SPLIT_DENSE(D(block, block, level), row_split, col_split);
      // Apply co block.
      for (int left_col = block-1; left_col >= 0; --left_col) {
        if (is_admissible.exists(block, left_col, level) && !is_admissible(block, left_col, level)) {
          int64_t row_split = block_size - U(block, level).cols;
          int64_t col_split = block_size - V(left_col, level).cols;
          auto left_splits = SPLIT_DENSE(D(block, left_col, level), row_split, col_split);

          Matrix x_block(x_level_split[block]), x_left_col(x_level_split[left_col]);
          auto x_block_splits = x_block.split(std::vector<int64_t>(1, row_split), {});
          auto x_left_col_splits = x_left_col.split(std::vector<int64_t>(1, col_split), {});

          matmul(left_splits[1], x_left_col_splits[1], x_block_splits[0], false, false, -1.0, 1.0);
          for (int64_t i = 0; i < block_size; ++i) {
            x_level(block * block_size + i, 0) = x_block(i, 0);
          }
        }
      }

      // Apply c block present on the right of this diagonal block.
      for (int right_col = nblocks-1; right_col > block; --right_col) {
        if (is_admissible.exists(block, right_col, level) &&
            !is_admissible(block, right_col, level)) {
          int64_t row_split = block_size - U(block, level).cols;
          auto right_splits = D(block, right_col, level).
            split(std::vector<int64_t>(1, row_split), {});

          Matrix x_block(x_level_split[block]);
          auto x_block_splits = x_block.split(std::vector<int64_t>(1, row_split), {});

          matmul(right_splits[0], x_level_split[right_col],
                 x_block_splits[0], false, false, -1.0, 1.0);
          for (int64_t i = 0; i < block_size; ++i) {
            x_level(block * block_size + i, 0) = x_block(i, 0);
          }
        }
      }

      Matrix x_block(x_level_split[block]);
      auto x_block_splits = x_block.split(std::vector<int64_t>(1, row_split), {});
      matmul(block_splits[1], x_block_splits[1], x_block_splits[0], false, false, -1.0, 1.0);
      solve_triangular(block_splits[0], x_block_splits[0], Hatrix::Left, Hatrix::Upper, false);
      for (int64_t i = 0; i < block_size; ++i) {
        x_level(block * block_size + i, 0) = x_block(i, 0);
      }
    }

    for (int block = nblocks-1; block >= 0; --block) {
      int block_size = V(block, level).rows;
      auto V_F = make_complement(V(block, level));
      Matrix prod = matmul(V_F, x_level_split[block]);
      for (int i = 0; i < block_size; ++i) {
        x_level(block * block_size + i, 0) = prod(i, 0);
      }
    }
  }

  Matrix H2::solve(Matrix& b, int _level) {
    int level = _level;
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

      int n = 0;
      for (int i = 0; i < num_nodes; ++i) { n += U(i, level).rows; }
      Matrix x_level(n, 1);
      for (int i = 0; i < x_level.rows; ++i) {
        x_level(i, 0) = x(rhs_offset + i, 0);
      }

      solve_forward_level(x_level, level);

      for (int i = 0; i < x_level.rows; ++i) {
        x(rhs_offset + i, 0) = x_level(i, 0);
      }

      rhs_offset = permute_forward(x, level, rhs_offset);
    }

    // std::cout << "rhs offset: " << rhs_offset << std::endl;
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
    for (; level <= _level; ++level) {
      int64_t num_nodes = pow(2, level);

      bool lr_exists = false;
      for (int block = 0; block < num_nodes; ++block) {
        if (V.exists(block, level)) {
          lr_exists = true;
        }
      }
      if (!lr_exists) { break; }

      int n = 0;
      for (int i = 0; i < num_nodes; ++i) {
        n += V(i, level).rows;
      }
      Matrix x_level(n, 1);

      rhs_offset = permute_backward(x, level, rhs_offset);

      for (int i = 0; i < x_level.rows; ++i) {
        x_level(i, 0) = x(rhs_offset + i, 0);
      }

      solve_backward_level(x_level, level);

      for (int i = 0; i < x_level.rows; ++i) {
        x(rhs_offset + i, 0) = x_level(i, 0);
      }
    }

    return x;
  }

} // namespace Hatrix

// Generates UF chain for the A2 matrix.
std::vector<Hatrix::Matrix> generate_UF_chain(Hatrix::H2& A) {
  std::vector<Hatrix::Matrix> U_F;
  int level = 2;

  int num_nodes = pow(2, level);
  for (int block = 0; block < num_nodes; ++block) {
    Matrix UF_full = generate_identity_matrix(A.rank * 8, A.rank * 8);
    auto UF_full_splits = UF_full.split(8, 8);

    if (A.U.exists(block, level)) {
      int block_size = A.U(block, level).rows;
      Matrix UF_block = make_complement(A.U(block, level));

      auto UF_block_splits = SPLIT_DENSE(UF_block, block_size - A.U(block, level).cols,
                                         block_size - A.U(block, level).cols);

      int permuted_nblocks = 8;

      UF_full_splits[block * permuted_nblocks + block] = UF_block_splits[0];
      UF_full_splits[(block + num_nodes) * permuted_nblocks + block] = UF_block_splits[2];
      UF_full_splits[block * permuted_nblocks + (block + num_nodes)] = UF_block_splits[1];
      UF_full_splits[(block + num_nodes) * permuted_nblocks + block + num_nodes] =
        UF_block_splits[3];
    }

    U_F.push_back(UF_full);
  }

  return U_F;
}

std::vector<Hatrix::Matrix> generate_VF_chain(Hatrix::H2& A) {
  std::vector<Hatrix::Matrix> V_F;
  int level = 2;

  int num_nodes = pow(2, level);
  for (int block = 0; block < num_nodes; ++block) {
    Matrix VF_full = generate_identity_matrix(A.rank * 8, A.rank * 8);
    auto VF_full_splits = VF_full.split(8, 8);

    if (A.V.exists(block, level)) {
      int block_size = A.V(block, level).rows;
      Matrix VF_block = make_complement(A.V(block, level));

      auto VF_block_splits = SPLIT_DENSE(VF_block, block_size - A.V(block, level).cols,
                                         block_size - A.V(block, level).cols);

      int permuted_nblocks = 8;

      VF_full_splits[block * permuted_nblocks + block] = VF_block_splits[0];
      VF_full_splits[(block + num_nodes) * permuted_nblocks + block] = VF_block_splits[2];
      VF_full_splits[block * permuted_nblocks + (block + num_nodes)] = VF_block_splits[1];
      VF_full_splits[(block + num_nodes) * permuted_nblocks + block + num_nodes] =
        VF_block_splits[3];
    }

    V_F.push_back(VF_full);
  }

  return V_F;

}

std::vector<Hatrix::Matrix> generate_L2_chain(Hatrix::H2& A) {
  std::vector<Hatrix::Matrix> L2;
  int block_size = A.rank;
  int permuted_nblocks = 8;
  int level = 2;
  int nblocks = pow(2, level);

  for (int block = 0; block < nblocks; ++block) {
    Matrix L_block = generate_identity_matrix(A.rank * 8, A.rank * 8);
    auto L_block_splits = L_block.split(8, 8);

    for (int j = 0; j <= block; ++j) {
      if (A.is_admissible.exists(block, j, level) && !A.is_admissible(block, j, level)) {
        auto D_splits = A.D(block, j, level).split(2, 2);

        // Copy the cc parts
        if (block == j) {
          L_block_splits[block * permuted_nblocks + j] = lower(D_splits[0]);
        }
        else {
          L_block_splits[block * permuted_nblocks + j] = D_splits[0];
        }

        L_block_splits[(block + nblocks) * permuted_nblocks + j] = D_splits[2];
      }
    }

    // Copy oc parts belonging to the 'upper' parts of the matrix
    for (int i = 0; i < block; ++i) {
      if (A.is_admissible.exists(i, block, level) && !A.is_admissible(i, block, level)) {
        auto D_splits = A.D(i, block, level).split(2, 2);
        L_block_splits[(i + nblocks) * permuted_nblocks + block] = D_splits[2];
      }
    }

    L2.push_back(L_block);
  }

  return L2;
}

std::vector<Hatrix::Matrix> generate_U2_chain(Hatrix::H2& A) {
  std::vector<Hatrix::Matrix> U2;
  int block_size = A.rank;
  int permuted_nblocks = 8;
  int level = 2;
  int nblocks = pow(2, level);

  for (int block = 0; block < nblocks; ++block) {
    Matrix U_block = generate_identity_matrix(A.rank * 8, A.rank * 8);
    auto U_splits = U_block.split(8, 8);

    for (int i = 0; i <= block; ++i) {
      if (A.is_admissible.exists(i, block, level) && !A.is_admissible(i, block, level)) {
        auto D_splits = A.D(i, block, level).split(2, 2);

        // Copy the cc blocks
        if (block == i) {
          U_splits[i * permuted_nblocks + block] = upper(D_splits[0]);
        }
        else {
          U_splits[i * permuted_nblocks + block] = D_splits[0];
        }

        // Copy the co parts
        U_splits[i * permuted_nblocks + block + nblocks] = D_splits[1];
      }
    }

    for (int j = 0; j < block; ++j) {
      if (A.is_admissible.exists(block, j, level) && !A.is_admissible(block, j, level)) {
        auto D_splits = A.D(block, j, level).split(2, 2);
        U_splits[block * permuted_nblocks + (j + nblocks)] = D_splits[1];
      }
    }

    U2.push_back(U_block);
  }

  return U2;
}

Hatrix::Matrix generate_L1(Hatrix::H2& A) {
  Matrix L1 = generate_identity_matrix(A.rank * 8, A.rank * 8);
  auto L1_splits = L1.split(8, 8);
  int level = 1;
  int num_nodes = pow(2, level);

  for (int i = 0; i < num_nodes; ++i) {
    for (int j = 0; j <= i; ++j) {
      std::vector<int> row_children({i * 2 + 4, i * 2 + 1 + 4});
      std::vector<int> col_children({j * 2 + 4, j * 2 + 1 + 4});

      auto D_split = A.D(i, j, level).split(2, 2);

      if (i == j) {
        for (int c1 = 0; c1 < 2; ++c1) {
          for (int c2 = 0; c2 <= c1; ++c2) {
            if (c1 == c2) {
              L1_splits[row_children[c1] * 8 + col_children[c2]] = lower(D_split[c1 * 2 + c2]);
            }
            else {
              L1_splits[row_children[c1] * 8 + col_children[c2]] = D_split[c1 * 2 + c2];
            }
          }
        }
      }
      else {
        for (int c1 = 0; c1 < 2; ++c1) {
          for (int c2 = 0; c2 < 2; ++c2) {
            L1_splits[row_children[c1] * 8 + col_children[c2]] = D_split[c1 * 2 + c2];
          }
        }
      }
    }
  }

  return L1;
}

Hatrix::Matrix generate_U1(Hatrix::H2& A) {
  Matrix U1 = generate_identity_matrix(A.rank * 8, A.rank * 8);
  auto U1_splits = U1.split(8, 8);
  int level = 1;
  int num_nodes = pow(2, level);

  for (int i = 0; i < num_nodes; ++i) {
    for (int j = i; j < num_nodes; ++j) {
      std::vector<int> row_children({i * 2 + 4, i * 2 + 1 + 4});
      std::vector<int> col_children({j * 2 + 4, j * 2 + 1 + 4});

      auto D_split = A.D(i, j, level).split(2, 2);

      if (i == j) {
        for (int c1 = 0; c1 < 2; ++c1) {
          for (int c2 = c1; c2 < 2; ++c2) {
            if (c1 == c2) {
              U1_splits[row_children[c1] * 8 + col_children[c2]] = upper(D_split[c1 * 2 + c2]);
            }
            else {
              U1_splits[row_children[c1] * 8 + col_children[c2]] = D_split[c1 * 2 + c2];
            }
          }
        }
      }
      else {
        for (int c1 = 0; c1 < 2; ++c1) {
          for (int c2 = 0; c2 < 2; ++c2) {
            U1_splits[row_children[c1] * 8 + col_children[c2]] = D_split[c1 * 2 + c2];
          }
        }
      }
    }
  }

  return U1;
}

Matrix unpermute_matrix(Matrix PA, H2& A) {
  Matrix M(A.rank * 8, A.rank * 8);

  int level = 2;
  int64_t block_size = A.rank * 2;
  int64_t permuted_nblocks = 8;
  std::vector<int64_t> row_offsets, col_offsets;
  int num_nodes = pow(2, level);

  auto PA_splits = PA.split(8, 8);
  auto M_splits = M.split(4, 4);

  for (int i = 0; i < num_nodes; ++i) {
    for (int j = 0; j < num_nodes; ++j) {
      Matrix block(block_size, block_size);
      auto block_splits = SPLIT_DENSE(block,
                                      block_size - A.rank,
                                      block_size - A.rank);

      block_splits[0] = PA_splits[(i) * permuted_nblocks + j];
      block_splits[1] = PA_splits[i * permuted_nblocks + j + num_nodes];
      block_splits[2] = PA_splits[(i + num_nodes) * permuted_nblocks + j];
      block_splits[3] = PA_splits[(i + num_nodes) * permuted_nblocks + j + num_nodes];

      M_splits[i * num_nodes + j] = block;
    }
  }


  return M;
}

void verify_A1_solve(Matrix& A1, H2& A, const randvec_t& randpts) {
  Matrix b = generate_laplacend_matrix(randpts, A.rank * 4, 1, 0, 0, PV);
  auto A1_22 = Matrix(A1.split(2, 2)[3]);
  auto x1_solve = lu_solve(A1_22, b);

  auto x1_h2 = A.solve(b, 1);

  std::cout << "A1 solve error: " << norm(x1_h2 - x1_solve) / norm(x1_solve) << std::endl;
}

void verify_A1_factorization(Hatrix::H2& A, const randvec_t& randpts) {
  Matrix L1 = generate_L1(A);
  Matrix U1 = generate_U1(A);
  Matrix A1_actual = matmul(L1, U1);

  Matrix diff = A1_actual - A1_expected;
  int nblocks = 4;
  auto d_splits = diff.split(nblocks, nblocks);
  auto m_splits = A1_expected.split(nblocks, nblocks);

  std::cout << "A1 factorization rel error: " << norm(diff) / norm(A1_expected) << std::endl;

  // std::cout << "A1 block errors:\n";
  // for (int i = 2; i < nblocks; ++i) {
  //   for (int j = 2; j < nblocks; ++j) {
  //     int idx = i * nblocks + j;
  //     std::cout << "(" << i << "," << j << ") block rel error: " << (norm(d_splits[idx]) / norm(m_splits[idx])) << std::endl;
  //   }
  // }

  verify_A1_solve(A1_actual, A, randpts);
}

void verify_A2_solve(Matrix& A2, H2& A, const randvec_t& randpts) {
  Matrix b = generate_laplacend_matrix(randpts, A.rank * 8, 1, 0, 0, PV);
  auto x2_dense = lu_solve(A2, b);
  auto x2_h2 = A.solve(b, 2);

  std::cout << "A2 solve error: " << norm(x2_h2 - x2_dense) / norm(x2_dense) << std::endl;
}


void verify_A2_factorization(Hatrix::H2& A, const randvec_t& randpts) {
  auto UF = generate_UF_chain(A);
  auto VF = generate_VF_chain(A);
  auto L2 = generate_L2_chain(A);
  auto U2 = generate_U2_chain(A);
  Hatrix::Matrix L1 = generate_L1(A);
  Hatrix::Matrix U1 = generate_U1(A);

  auto product = generate_identity_matrix(A.rank * 8, A.rank * 8);

  for (int i = 0; i < 4; ++i) {
    product = matmul(product, UF[i]);
    product = matmul(product, L2[i]);
  }

  product = matmul(product, L1);
  product = matmul(product, U1);

  for (int i = 3; i >= 0; --i) {
    product = matmul(product, U2[i]);
    product = matmul(product, VF[i], false, true);
  }

  auto A2_actual = unpermute_matrix(product, A);

  auto diff = (A2_expected - A2_actual);

  auto diff_splits = diff.split(4, 4);
  auto A2_expected_splits = A2_expected.split(4, 4);

  (A2_expected - A2_actual).print();
  std::cout << "A2 factorization error: "
            <<  norm(A2_expected - A2_actual) / norm(A2_expected) << std::endl;

  std::cout << "A2 block wise factorization error: \n";
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      std::cout << "<i, j>: " << i << ", " << j
                << " -- " << std::setprecision(8) << norm(diff_splits[i * 4 + j]) / norm(A2_expected_splits[i * 4 + j])
                << "   ";
    }
    std::cout << std::endl;
  }

  verify_A2_solve(A2_actual, A, randpts);
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

  A.print_structure();
  A.factorize(randpts);

  std::cout << "-- H2 verification --\n";
  verify_A1_factorization(A, randpts);
  verify_A2_factorization(A, randpts);

  Hatrix::Matrix x = A.solve(b, A.height);
  Hatrix::Matrix Adense = Hatrix::generate_laplacend_matrix(randpts, N, N, 0, 0, PV);
  Hatrix::Matrix x_solve = lu_solve(Adense, b);

  int num_nodes = pow(2, A.height);

  auto x_splits = x.split(num_nodes, 1);
  auto x_solve_splits = x_solve.split(num_nodes, 1);

  std::cout << "BLOCK WISE SOLVE ACCURACY:\n";
  for (int i = 0; i < num_nodes; ++i) {
    std::cout << "i -> " << i << " "
              <<  norm(x_splits[i] - x_solve_splits[i]) / norm(x_solve_splits[i]) << std::endl;
  }

  // std::cout << "X - X_SOLVE\n";
  // (x - x_solve).print();

  double solve_error =  Hatrix::norm(x - x_solve) / Hatrix::norm(x_solve);

  Hatrix::Context::finalize();

  std::cout << "N= " << N << " rank= " << rank << " admis= " << admis << " leaf= "
            << int(N / pow(2, height))
            << " height=" << height
            << " const. error=" << construct_error
            << " solve error=" << solve_error << std::endl;

}
