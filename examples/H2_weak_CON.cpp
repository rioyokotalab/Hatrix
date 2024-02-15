#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <chrono>
#include <fstream>

#include "Hatrix/Hatrix.hpp"
using namespace Hatrix;
Hatrix::greens_functions::kernel_function_t kernel;

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

double PV = 1e-3;

namespace Hatrix {
  class HSS {
  public:
    ColLevelMap U;
    RowLevelMap V;
    RowColLevelMap<Matrix> D, S;
    int64_t N, rank, height;

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


    std::tuple<RowLevelMap, ColLevelMap> generate_leaf_nodes(const randvec_t& randvec) {
      int nblocks = pow(2, height);
      ColLevelMap Ugen;
      RowLevelMap Vgen;

      for (int block = 0; block < nblocks; ++block) {
        int slice = N / nblocks;
        int leaf_size = (block == (nblocks-1)) ? (N - (slice * block)) :  slice;

        // Diagonal offset is used since the last block can have a different shape from
        // the rest.
        int diagonal_offset = slice * block;
        D.insert(block, block, height,
                 Hatrix::generate_laplacend_matrix(randvec, leaf_size, leaf_size,
                                                   diagonal_offset, diagonal_offset));

        Matrix U_temp, Ugen_temp;
        std::tie(U_temp, Ugen_temp) = generate_column_bases(block,
                                                            leaf_size,
                                                            diagonal_offset,
                                                            slice,
                                                            randvec);

        U.insert(block, height, std::move(U_temp));
        Ugen.insert(block, height, std::move(Ugen_temp));

        Matrix V_temp, Vgen_temp;
        std::tie(V_temp, Vgen_temp) = generate_row_bases(block,
                                                         leaf_size,
                                                         diagonal_offset,
                                                         slice,
                                                         randvec);
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
        int slice = N / num_nodes;
        int diagonal_offset = slice * block;
        int leaf_size = (block == (num_nodes-1)) ? (N - (slice * block)) :  slice;

        double diagonal_error =
          rel_error(D(block, block, height),
                    Hatrix::generate_laplacend_matrix(randvec, leaf_size, leaf_size,
                                                      diagonal_offset, diagonal_offset));
        error += pow(diagonal_error, 2);
      }

      // regenerate off-diagonal blocks and test for correctness.
      for (int level = height; level > 1; --level) {
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
          error += pow(offD_error, 2);
        }
      }

      return std::sqrt(error);
    }
  };
} // namespace Hatrix

static Matrix
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

static void
construct_H2_weak_leaf_nodes(Hatrix::SymmetricSharedBasisMatrix& A, Hatrix::Domain& domain,
                             int64_t N, int64_t nleaf, int64_t rank) {
  Hatrix::Matrix Utemp, Stemp, Vtemp;
  double error;
  const int64_t nblocks = pow(2, A.max_level);
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

static RowLevelMap
generate_transfer_matrices(Hatrix::SymmetricSharedBasisMatrix& A, Hatrix::Domain& domain,
                           const int64_t N, const int64_t nleaf, const int64_t rank, const int64_t level) {
  Matrix Ui, Si, _Vi; double error;
  const int64_t nblocks = pow(2, level);
  const int64_t block_size = N / nblocks;
  Matrix AY(block_size, block_size);
  RowLevelMap Ubig_parent;

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

    // Generate the full basis to pass to the next level.
    auto Utransfer_splits = Ui.split(std::vector<int64_t>{rank}, {});

    Matrix Ubig(block_size, rank);
    auto Ubig_splits = Ubig.split(2, 1);
    matmul(Ubig_child1, Utransfer_splits[0], Ubig_splits[0]);
    matmul(Ubig_child2, Utransfer_splits[1], Ubig_splits[1]);

    A.U.insert(row, level, std::move(Ui));
    Ubig_parent.insert(row, level, std::move(Ubig));
  }

  for (int64_t row = 0; row < nblocks; ++row) {
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

  return Ubig_parent;
}

static void
construct_H2_weak(Hatrix::SymmetricSharedBasisMatrix& A, Hatrix::Domain& domain,
                  const int64_t N, const int64_t nleaf, const int64_t rank) {
  construct_H2_weak_leaf_nodes(A, domain, N, nleaf, rank);

  RowLevelMap Uchild = A.U;
  for (int64_t level = A.max_level - 1; level >= A.min_level; --level) {
    Uchild = generate_transfer_matrices(A, domain, N, nleaf, rank, level);
  }
}

static Matrix
matmul(SymmetricSharedBasisMatrix& A, Matrix& x, int64_t N, int64_t rank) {
  std::vector<Matrix> x_hat;
  const int64_t leaf_nblocks = pow(2, A.max_level);
  auto x_splits = x.split(leaf_nblocks, 1);

  Matrix b(x.rows, 1);

  return b;
}

int main(int argc, char *argv[]) {
  const int64_t N = atoll(argv[1]);
  const int64_t rank = atoll(argv[2]);
  const int64_t height = atoll(argv[3]);
  const double admis = 0;
  const int64_t nleaf = N / pow(2, height);

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
  A.max_level = height;

  // Use a simple distance from diagonal based admissibility condition. admis is kept
  // at 0 since this is a weakly admissible code.
  A.generate_admissibility(domain, false, Hatrix::ADMIS_ALGORITHM::DIAGONAL_ADMIS, admis);

  // Construct a weak admissibility H2 matrix (HSS matrix).
  construct_H2_weak(A, domain, N, nleaf, rank);

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
