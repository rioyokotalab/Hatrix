#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>

#include "Hatrix/Hatrix.hpp"
using namespace Hatrix;
Hatrix::greens_functions::kernel_function_t kernel;

using randvec_t = std::vector<std::vector<double> >;

double rel_error(const double A_norm, const double B_norm) {
  double diff = A_norm - B_norm;
  return std::sqrt((diff * diff) / (B_norm * B_norm));
}

namespace Hatrix {
  class HSS {
    ColLevelMap U;
    RowLevelMap V;
    RowColLevelMap<Matrix> D, S;
    int64_t N, rank, height;

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

    Matrix generate_coupling_matrix(const randvec_t& randvec, int row, int col, int leaf_size, int level) {
      Matrix D = generate_laplacend_matrix(randvec, leaf_size, leaf_size,
                                           row * leaf_size, col * leaf_size);
      Matrix S = Hatrix::matmul(Hatrix::matmul(U(row, level), D, true), V(col, level));
      return S;
    }

    std::tuple<RowLevelMap, ColLevelMap> generate_leaf_nodes(const randvec_t& randvec) {
      int nblocks = pow(height, 2);
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

      for (int block = 0; block < nblocks; ++block) {
        int s_col = block % 2 == 0 ? block + 1 : block - 1;
        S.insert(block, s_col, height, generate_coupling_matrix(randvec, block, s_col,
                                                                leaf_size, height));
      }

      return {Ugen, Vgen};
    }

    Matrix generate_U_actual_bases(int p) {
      int child1 = p * 2;
      int child2 = p * 2 + 1;
      int leaf_size = int(N / 2);
      Matrix Ubig(rank, leaf_size);

      std::vector<Matrix> Ubig_splits = Ubig.split(1, 2);
      std::vector<Matrix> U_splits = U(p, height-1).split(2, 1);

      matmul(U_splits[0], U(child1, height), Ubig_splits[0], true, true);
      matmul(U_splits[1], U(child2, height), Ubig_splits[1], true, true);

      return transpose(Ubig);
    }

    Matrix generate_V_actual_bases(int p) {
      int child1 = p * 2;
      int child2 = p * 2 + 1;
      int leaf_size = int(N / 2);
      Matrix Vbig(leaf_size, rank);

      std::vector<Matrix> Vbig_splits = Vbig.split(2, 1);
      std::vector<Matrix> V_splits = V(p, height - 1).split(2, 1);

      matmul(V(child1, height), V_splits[0], Vbig_splits[0]);
      matmul(V(child2, height), V_splits[1], Vbig_splits[1]);

      return Vbig;
    }

    Matrix generate_non_leaf_coupling_matrix(const randvec_t& randvec, int row, int col, int leaf_size,
                                             Matrix& Ubig, Matrix& Vbig) {
      Matrix D = generate_laplacend_matrix(randvec, leaf_size, leaf_size,
                                           row * leaf_size, col * leaf_size);
      return matmul(matmul(Ubig, D, true, false), Vbig);
    }

    void generate_transfer_matrices(const randvec_t& randvec, RowLevelMap& Ugen, ColLevelMap& Vgen) {
      Matrix Ui, Si, Vi; double error;

      for (int p = 0; p < 2; ++p) {
        int child1 = p * 2;
        int child2 = p * 2 + 1;

        // Generate U transfer matrix.
        Matrix& Ugen_upper = Ugen(child1, height);
        Matrix& Ugen_lower = Ugen(child2, height);
        Matrix Ugen_concat(Ugen_upper.rows + Ugen_lower.rows, Ugen_upper.cols);
        std::vector<Matrix> Ugen_slices = Ugen_concat.split(2, 1);
        Ugen_slices[0] = Ugen_upper;
        Ugen_slices[1] = Ugen_lower;

        std::tie(Ui, Si, Vi, error) = truncated_svd(Ugen_concat, rank);
        U.insert(p, height-1, std::move(Ui));

        // Generate V transfer matrix.
        Matrix& Vgen_upper = Vgen(child1, height);
        Matrix& Vgen_lower = Vgen(child2, height);
        Matrix Vgen_concat(Vgen_upper.rows + Vgen_lower.rows, Vgen_upper.cols);
        std::vector<Matrix> Vgen_slices = Vgen_concat.split(2, 1);
        Vgen_slices[0] = Vgen_upper;
        Vgen_slices[1] = Vgen_lower;

        std::tie(Ui, Si, Vi, error) = truncated_svd(Vgen_concat, rank);
        V.insert(p, height-1, std::move(Ui));
      }

      for (int row = 0; row < 2; ++row) {
        int col = row % 2 == 0 ? row + 1 : row - 1;
        int leaf_size = int(N / 2);

        Matrix Ubig = generate_U_actual_bases(row);
        Matrix Vbig = generate_V_actual_bases(col);

        S.insert(row, col, height - 1,
                 generate_non_leaf_coupling_matrix(randvec, row, col, leaf_size,
                                                   Ubig, Vbig));
      }
    }

  public:

    HSS(const randvec_t& randpts, int _N, int _rank, int _height) :
      N(_N), rank(_rank), height(_height) {
      RowLevelMap Ugen; ColLevelMap Vgen;

      std::tie(Ugen, Vgen) = generate_leaf_nodes(randpts);
      generate_transfer_matrices(randpts, Ugen, Vgen);
    }

    double construction_relative_error(const randvec_t& randvec) {
      int leaf_size = N / pow(height, 2);
      double error = 0;

      // Check leaf level blocks.
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
          Matrix A = generate_laplacend_matrix(randvec, leaf_size, leaf_size,
                                               i * leaf_size, j * leaf_size);
          if (i == j) {
            error += pow(rel_error(Hatrix::norm(A), Hatrix::norm(D(i, j, height)) ), 2);
          }
          else {
            Matrix Anew = Hatrix::matmul(Hatrix::matmul(U(i, height),
                                                        S(i, j, height)),
                                         V(j, height), false, true);
            error += pow(rel_error(Hatrix::norm(A),
                                   Hatrix::norm(Anew)), 2);
          }
        }
      }

      for (int i = 2; i < 4; ++i) {
        for (int j = 2; j < 4; ++j) {
          Matrix A = generate_laplacend_matrix(randvec, leaf_size, leaf_size,
                                               i * leaf_size, j * leaf_size);
          if (i == j) {
            error += pow(rel_error(Hatrix::norm(A), Hatrix::norm(D(i, j, height)) ), 2);
          }
          else {
            Matrix Anew = Hatrix::matmul(Hatrix::matmul(U(i, height),
                                                        S(i, j, height)),
                                         V(j, height), false, true);
            error += pow(rel_error(Hatrix::norm(A),
                                   Hatrix::norm(Anew)), 2);
          }
        }
      }

      // Check off-diagonal non-leaf blocks

      // Upper right
      int row = 0, col = 1;
      leaf_size = N / 2;

      Matrix A = generate_laplacend_matrix(randvec, leaf_size, leaf_size,
                                           row * leaf_size, col * leaf_size);
      Matrix Ubig = generate_U_actual_bases(row);
      Matrix Vbig = generate_V_actual_bases(col);
      Matrix Anew = matmul(matmul(Ubig, S(row, col, height-1)), Vbig, false, true);
      error += pow(rel_error(norm(A), norm(Anew)), 2);

      // Lower left
      row = 1, col = 0;
      A = generate_laplacend_matrix(randvec, leaf_size, leaf_size,
                                           row * leaf_size, col * leaf_size);
      Ubig = generate_U_actual_bases(row);
      Vbig = generate_V_actual_bases(col);
      Anew = matmul(matmul(Ubig, S(row, col, height-1)), Vbig, false, true);
      error += pow(rel_error(norm(A), norm(Anew)), 2);

      return std::sqrt(error);
    }
  };
}

std::vector<double> equally_spaced_vector(int N, double minVal, double maxVal) {
  std::vector<double> res(N, 0.0);
  double rnge = maxVal - minVal;
  for(int i=0; i<N; i++) {
    res[i] = minVal + ((double)i/(double)rnge);
  }
  return res;
}

void
construct_H2_weak_2level_leaf_nodes(Hatrix::SymmetricSharedBasisMatrix& A, Hatrix::Domain& domain,
                                    int64_t N, int64_t nleaf, int64_t rank) {
  Hatrix::Matrix Utemp, Stemp, Vtemp;
  double error;
  const int64_t nblocks = N / nleaf;
  // populate diagonal blocks.
  for (int64_t i = 0; i < nblocks; ++i) {
    Hatrix::Matrix Aij = generate_p2p_interactions(domain,
                                                   i * nleaf, nleaf,
                                                   i * nleaf, nleaf,
                                                   kernel);
    A.D.insert(i, i, A.max_level, std::move(Aij));
  }

  // populate leaf level shared bases.
  for (int64_t i = 0; i < nblocks; ++i) {
    Hatrix::Matrix AY(nleaf, nleaf);
    for (int64_t j = 0; j < nblocks; ++j) {
      if (i != j) {
        Hatrix::Matrix dense = generate_p2p_interactions(domain,
                                                         i * nleaf, nleaf,
                                                         j * nleaf, nleaf,
                                                         kernel);
        AY += dense;
      }
    }
    std::tie(Utemp, Stemp, Vtemp, error) = Hatrix::truncated_svd(AY, rank);
    A.U.insert(i, A.max_level, std::move(Utemp));
  }

  // Update S blocks.
  for (int i = 0; i < nblocks; ++i) {
    for (int j = 0; j < i; ++j) {
      if (A.is_admissible(i, j, A.max_level)) {
        Hatrix::Matrix dense = generate_p2p_interactions(domain,
                                                         i * nleaf, nleaf,
                                                         j * nleaf, nleaf,
                                                         kernel);
        A.S.insert(i, j, A.max_level,
                   Hatrix::matmul(Hatrix::matmul(A.U(i, A.max_level), dense, true), A.U(j, A.max_level)));
      }
    }
  }
}

Matrix
generate_actual_bases(Hatrix::SymmetricSharedBasisMatrix& A, const int64_t N,
                      const int64_t p, const int64_t rank) {
  int64_t child1 = p * 2;
  int64_t child2 = p * 2 + 1;
  int64_t leaf_size = N / 2;
  Matrix Ubig(leaf_size, rank);

  std::vector<Matrix> Ubig_splits = Ubig.split(2, 1);
  std::vector<Matrix> U_splits = A.U(p, A.max_level-1).split(2, 1);

  matmul(A.U(child1, A.max_level), U_splits[0], Ubig_splits[0]);
  matmul(A.U(child2, A.max_level), U_splits[1], Ubig_splits[1]);

  return Ubig;
}

void
generate_transfer_matrices(Hatrix::SymmetricSharedBasisMatrix& A, Hatrix::Domain& domain,
                           const int64_t N, const int64_t nleaf, const int64_t rank) {
  Matrix Ui, Si, _Vi; double error;
  int64_t block_size = N / 2;
  int64_t level = A.max_level - 1;
  int64_t nblocks = pow(2, level);
  Matrix AY(block_size, block_size);

  for (int64_t row = 0; row < nblocks; ++row) {
    for (int64_t col = 0; col < nblocks; ++col) {
      if (row != col) {             // admit only admissible blocks.
        Hatrix::Matrix dense = generate_p2p_interactions(domain,
                                                         row * block_size, block_size,
                                                         col * block_size, block_size,
                                                         kernel);
        AY += dense;
      }
    }

    int child1 = row * 2;
    int child2 = row * 2 + 1;

    // Generate U transfer matrix.
    Matrix& Ubig_child1 = A.U(child1, A.max_level);
    Matrix& Ubig_child2 = A.U(child2, A.max_level);
    Matrix temp(Ubig_child1.cols + Ubig_child2.cols, AY.cols);
    std::vector<Matrix> temp_splits = temp.split(2, 1);
    std::vector<Matrix> AY_splits = AY.split(2, 1);

    matmul(Ubig_child1, AY_splits[0], temp_splits[0], true, false, 1, 0);
    matmul(Ubig_child2, AY_splits[1], temp_splits[1], true, false, 1, 0);

    std::tie(Ui, Si, _Vi, error) = truncated_svd(temp, rank);
    A.U.insert(row, level, std::move(Ui));
  }

  for (int64_t row = 0; row < 2; ++row) {
    int64_t col = (row % 2 == 0) ? row + 1 : row - 1;
    Matrix Urow_actual = generate_actual_bases(A, N, row, rank);
    Matrix Ucol_actual = generate_actual_bases(A, N, col, rank);

    Matrix dense = generate_p2p_interactions(domain,
                                             row * block_size, block_size,
                                             col * block_size, block_size,
                                             kernel);

    Matrix S_block = matmul(matmul(Urow_actual, dense, true), Ucol_actual);
    A.S.insert(row, col, level, std::move(dense));
  }
}

void
construct_H2_weak_2level(Hatrix::SymmetricSharedBasisMatrix& A, Hatrix::Domain& domain,
                         int64_t N, int64_t nleaf, int64_t rank, double admis) {
  construct_H2_weak_2level_leaf_nodes(A, domain, N, nleaf, rank);
  generate_transfer_matrices(A, domain, N, nleaf, rank);
}

static void
multiply_S(const Hatrix::SymmetricSharedBasisMatrix& A,
           std::vector<Matrix>& x_hat, std::vector<Matrix>& b_hat,
           int64_t x_hat_offset, int64_t b_hat_offset, int64_t level) {
  int64_t nblocks = pow(2, level);

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j < i; ++j) {
      if (A.is_admissible.exists(i, j, level) && A.is_admissible(i, j, level)) {
        matmul(A.S(i, j, level), x_hat[x_hat_offset + j], b_hat[b_hat_offset + i]);
        matmul(A.S(i, j, level), x_hat[x_hat_offset + i], b_hat[b_hat_offset + j],
               true, false);
      }
    }
  }
}


static Matrix
matmul(SymmetricSharedBasisMatrix& A, Matrix& x, int64_t N, int64_t rank) {
  std::vector<Matrix> x_hat;
  const int64_t leaf_nblocks = pow(2, A.max_level);
  auto x_splits = x.split(leaf_nblocks, 1);

  for (int i = 0; i < leaf_nblocks; ++i) {
    x_hat.push_back(matmul(A.U(i, A.max_level), x_splits[i], true, false, 1.0));
  }

  int64_t x_hat_offset = 0;
  for (int64_t level = A.max_level - 1; level >= A.min_level; --level) {
    int64_t nblocks = pow(2, level);
    int64_t child_level = level + 1;

    for (int64_t i = 0; i < nblocks; ++i) {
      int64_t c1 = i * 2;
      int64_t c2 = i * 2 + 1;

      Matrix xtemp = Matrix(A.U(i, level).rows, 1);
      auto xtemp_splits = xtemp.split(std::vector<int64_t>(1, rank),
                                      {});
      xtemp_splits[0] = x_hat[x_hat_offset + c1];
      xtemp_splits[1] = x_hat[x_hat_offset + c2];

      x_hat.push_back(matmul(A.U(i, level), xtemp, true, false, 1.0));
    }

    x_hat_offset += pow(2, child_level);
  }

  std::vector<Matrix> b_hat;
  for (int64_t i = 0; i < pow(2, A.min_level); ++i) {
    b_hat.push_back(Matrix(rank, 1));
  }

  int64_t b_hat_offset = 0;
  multiply_S(A, x_hat, b_hat, x_hat_offset, b_hat_offset, A.min_level);

  for (int64_t level = A.min_level; level < A.max_level; ++level) {
    int64_t nblocks = pow(2, level);
    int64_t child_level = level+1;

    x_hat_offset -= pow(2, child_level);

    for (int64_t row = 0; row < nblocks; ++row) {
      int c_r1 = row * 2, c_r2 = row * 2 + 1;
      Matrix Ub = matmul(A.U(row, level),
                         b_hat[b_hat_offset + row]);
      auto Ub_splits = Ub.split(std::vector<int64_t>(1, A.U(c_r1, child_level).cols),
                                {});

      b_hat.push_back(Matrix(Ub_splits[0], true));
      b_hat.push_back(Matrix(Ub_splits[1], true));
    }
    multiply_S(A, x_hat, b_hat, x_hat_offset, b_hat_offset + nblocks, child_level);
    b_hat_offset += nblocks;
  }

  // Multiply with the leaf level transfer matrices to generate the product matrix.
  Matrix b(x.rows, 1);
  auto b_splits = b.split(leaf_nblocks, 1);
  for (int64_t i = 0; i < leaf_nblocks; ++i) {
    matmul(A.U(i, A.max_level), b_hat[b_hat_offset + i], b_splits[i]);
  }

  for (int64_t i = 0; i < leaf_nblocks; ++i) {
    for (int64_t j = 0; j <= i; ++j) {
      if (A.is_admissible.exists(i, j, A.max_level) &&
          !A.is_admissible(i, j, A.max_level)) {
        // TODO: make the diagonal tringular and remove this.
        if (i == j) {
          matmul(A.D(i, j, A.max_level), x_splits[j], b_splits[i]);
        }
        else {
          matmul(A.D(i, j, A.max_level), x_splits[j], b_splits[i]);
          matmul(A.D(i, j, A.max_level), x_splits[i], b_splits[j], true, false);
        }
      }
    }
  }


  return b;
}

int main(int argc, char *argv[]) {
  int64_t N = atoi(argv[1]);
  int64_t nleaf = atoi(argv[2]);
  int64_t rank = atoi(argv[3]);
  const double admis = 0;
  const int64_t height = 2;

  assert(N / nleaf == 4);

  if (N % nleaf != 0) {
    std::cout << "N % nleaf != 0. Aborting.\n";
    abort();
  }

  // Assign kernel function
  double add_diag = 1e-4;
  kernel = [&](const std::vector<double>& c_row,
               const std::vector<double>& c_col) {
    return Hatrix::greens_functions::laplace_1d_kernel(c_row, c_col, add_diag);
  };

  // Define a 1D grid geometry using the Domain class.
  const int64_t ndim = 1;
  Hatrix::Domain domain(N, ndim);
  domain.generate_grid_particles();
  domain.cardinal_sort_and_cell_generation(nleaf);

  // Initialize a Symmetric shared basis matrix container.
  Hatrix::SymmetricSharedBasisMatrix A;
  A.max_level = log2(N / nleaf);

  // Use a simple distance from diagonal based admissibility condition. admis is kept
  // at 0 since this is a weakly admissible code.
  A.generate_admissibility(domain, false, Hatrix::ADMIS_ALGORITHM::DIAGONAL_ADMIS, admis);

  // Construct a 2-level weak admissibility H2 matrix.
  construct_H2_weak_2level(A, domain, N, nleaf, rank, admis);

  // Verification of the construction with a matvec product.
  Hatrix::Matrix x = Hatrix::generate_random_matrix(N, 1);

  // Low rank matrix-vector product.
  Hatrix::Matrix b_lowrank = matmul(A, x, N, rank);

  // Generate a dense matrix.
  Matrix A_dense = Hatrix::generate_p2p_interactions(domain, kernel);
  Matrix b_dense = Hatrix::matmul(A_dense, x);
  Matrix diff = b_dense - b_lowrank;
  double rel_error = Hatrix::norm(diff) / Hatrix::norm(b_dense);

  std::cout << "Error : " << rel_error << std::endl;

  return 0;
}
