#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Hatrix/Hatrix.h"

using randvec_t = std::vector<std::vector<double> >;

namespace Hatrix {
  class H2 {
  private:
    ColLevelMap U;
    RowLevelMap V;
    RowColLevelMap<Matrix> D, S;
    RowColLevelMap<bool> is_admissible;
    int64_t N, rank, admis, height;

    Matrix get_Ubig(int p, int level) {
      if (level == height) {
        return U(p, level);
      }
      int child1 = p * 2;
      int child2 = p * 2 + 1;

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

    void generate_transfer_matrices(const randvec_t& randpts) {
      int nblocks = 2;          // level 1 so only 2 blocks on this level.
      int leaf_size = N / nblocks;
      constexpr int64_t oversampling = 5, child_level = 2;

      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          if (i == j) {
            is_admissible.insert(i, j, 1, false);
          }
          else {
            is_admissible.insert(i, j, 1, true);
          }
        }
      }

      std::vector<Matrix> Y;
      Matrix Utemp, Stemp, Vtemp;
      double error;
      RowLevelMap Ubig;
      ColLevelMap Vbig;

      // Generate random matrices.
      for (int64_t i = 0; i < nblocks; ++i) {
        Y.push_back(generate_random_matrix(leaf_size, rank + oversampling));
      }

      for (int64_t i = 0; i < nblocks; ++i) {
        Matrix AY(leaf_size, rank + oversampling);
        for (int64_t j = 0; j < nblocks; ++j) {
          if (is_admissible(i, j, 1)) {
            Matrix dense = generate_laplacend_matrix(randpts, leaf_size, leaf_size,
                                                     i * leaf_size, j * leaf_size);
            matmul(dense, Y[j], AY);
          }
        }

        // Generate U transfer matrix.
        Matrix& Ubig_child1 = U(i * 2, child_level);
        Matrix& Ubig_child2 = U(i * 2 + 1, child_level);
        Matrix temp(Ubig_child1.cols + Ubig_child2.cols, AY.cols);
        std::vector<Matrix> temp_splits = temp.split(2, 1);
        std::vector<Matrix> AY_splits = AY.split(2, 1);

        matmul(Ubig_child1, AY_splits[0], temp_splits[0], true, false, 1, 0);
        matmul(Ubig_child2, AY_splits[1], temp_splits[1], true, false, 1, 0);

        std::tie(Utemp, Stemp, Vtemp, error) = Hatrix::truncated_svd(temp, rank);
        U.insert(i, 1, std::move(Utemp));

        // Generate Ubig actual bases.
        Matrix Ubig_temp(leaf_size, rank);
        std::vector<Matrix> Utransfer_splits = U(i, 1).split(2, 1);
        std::vector<Matrix> Ubig_temp_splits = Ubig_temp.split(2, 1);

        matmul(Ubig_child1, Utransfer_splits[0], Ubig_temp_splits[0]);
        matmul(Ubig_child2, Utransfer_splits[1], Ubig_temp_splits[1]);
        Ubig.insert(i, 1, std::move(Ubig_temp));
      }

      for (int64_t j = 0; j < nblocks; ++j) {
        Matrix YtA(rank + oversampling, leaf_size);
        for (int64_t i = 0; i < nblocks; ++i) {
          if (is_admissible(i, j, 1)) {
            Matrix dense = generate_laplacend_matrix(randpts, leaf_size, leaf_size,
                                                     i * leaf_size, j * leaf_size);
            matmul(Y[i], dense, YtA, true);
          }
        }

        // Generate V transfer matrix.
        Matrix& Vbig_child1 = V(j * 2, child_level);
        Matrix& Vbig_child2 = V(j * 2 + 1, child_level);
        Matrix temp(YtA.rows, Vbig_child1.cols + Vbig_child2.cols);
        std::vector<Matrix> temp_splits = temp.split(1, 2);
        std::vector<Matrix> YtA_splits = YtA.split(1, 2);

        matmul(YtA_splits[0], Vbig_child1, temp_splits[0]);
        matmul(YtA_splits[1], Vbig_child2, temp_splits[1]);

        std::tie(Utemp, Stemp, Vtemp, error) = Hatrix::truncated_svd(temp, rank);
        V.insert(j, 1, std::move(transpose(Vtemp)));

        // Generate Vbig actual bases.
        Matrix Vbig_temp(rank, leaf_size);
        std::vector<Matrix> Vtransfer_splits = V(j, 1).split(2, 1);
        std::vector<Matrix> Vbig_temp_splits = Vbig_temp.split(1, 2);

        matmul(Vtransfer_splits[0], Vbig_child1, Vbig_temp_splits[0], true, true, 1, 0);
        matmul(Vtransfer_splits[1], Vbig_child2, Vbig_temp_splits[1], true, true, 1, 0);
        Vbig.insert(j, 1, transpose(Vbig_temp));
      }

      for (int64_t i = 0; i < nblocks; ++i) {
        for (int64_t j = 0; j < nblocks; ++j) {
          if (is_admissible(i, j, 1)) {
            Matrix dense = generate_laplacend_matrix(randpts, leaf_size, leaf_size,
                                                     i * leaf_size, j * leaf_size);
            S.insert(i, j, 1, matmul(matmul(Ubig(i, 1), dense, true, false), Vbig(j, 1)));
          }
        }
      }
    }

    void generate_leaf_nodes(const randvec_t& randpts) {
      int nblocks = pow(height, 2);
      int leaf_size = N / nblocks;
      ColLevelMap Ugen;
      RowLevelMap Vgen;

      for (int i = 0; i < nblocks; ++i) {
        int j = i % 2 == 0 ? i + 1 : i - 1;
        is_admissible.insert(i, j, height, std::abs(i - j) > admis);
        if (!is_admissible(i, j, height)) {
          D.insert(i, j, height, Hatrix::generate_laplacend_matrix(randpts, leaf_size, leaf_size,
                                                                   i * leaf_size, j * leaf_size));
        }

        is_admissible.insert(i, i, height, false);
        D.insert(i, i, height, Hatrix::generate_laplacend_matrix(randpts, leaf_size, leaf_size,
                                                                 i * leaf_size, i * leaf_size));

        for (int j = 0; j < nblocks; ++j) {
          if (!is_admissible.exists(i, j, 2)) {
            is_admissible.insert(i, j, 2, true);
          }
        }
      }

      // Use randomization and matrix multiplication to generate column bases U.
      constexpr int64_t oversampling = 5;
      Matrix Utemp, Stemp, Vtemp;
      double error;
      std::vector<Hatrix::Matrix> Y;

      // Generate a bunch of random matrices.
      for (int64_t i = 0; i < nblocks; ++i) {
        Y.push_back(
                    Hatrix::generate_random_matrix(leaf_size, rank + oversampling));
      }

      for (int64_t i = 0; i < nblocks; ++i) {
        Matrix AY(leaf_size, rank + oversampling);
        for (int64_t j = 0; j < nblocks; ++j) {
          if (is_admissible(i, j, height)) {
            Matrix dense = Hatrix::generate_laplacend_matrix(randpts,
                                                                     leaf_size, leaf_size,
                                                                     i*leaf_size, j*leaf_size);
            matmul(dense, Y[j], AY);
          }
        }
        std::tie(Utemp, Stemp, Vtemp, error) = Hatrix::truncated_svd(AY, rank);
        U.insert(i, height, std::move(Utemp));
      }

      for (int64_t j = 0; j < nblocks; ++j) {
        Matrix YtA(rank + oversampling, leaf_size);
        for (int64_t i = 0; i < nblocks; ++i) {
          if (is_admissible(i, j, height)) {
            Matrix dense = Hatrix::generate_laplacend_matrix(randpts,
                                                             leaf_size, leaf_size,
                                                             i*leaf_size, j*leaf_size);
            matmul(Y[i], dense, YtA, true);
          }
        }
        std::tie(Utemp, Stemp, Vtemp, error) = Hatrix::truncated_svd(YtA, rank);
        V.insert(j, height, std::move(transpose(Vtemp)));
      }

      // No S blocks at the leaf level since all the leaf nodes are dense.
    }

  public:

    H2(const randvec_t& randpts, int64_t N, int64_t rank, int64_t admis, int64_t height) :
      N(N), rank(rank), admis(admis), height(height) {
      RowLevelMap Ugen; ColLevelMap Vgen;

      generate_leaf_nodes(randpts);
      generate_transfer_matrices(randpts);
    }

    double construction_relative_error(const randvec_t& randpts) {
      double error = 0, fnorm = 0;

      // leaf level error checking for dense blocks
      int nblocks = pow(height, 2);
      int leaf_size = N / nblocks;
      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          if (D.exists(i, j, 2)) {
            Matrix dense = generate_laplacend_matrix(randpts, leaf_size, leaf_size,
                                                 i * leaf_size, j * leaf_size);
            fnorm += pow(norm(dense), 2);
            error += pow(norm(D(i, j, 2) - dense), 2);
          }
        }
      }

      // level 1
      nblocks = 2;
      leaf_size = N / nblocks;
      for (int i = 0; i < nblocks; ++i) {
        for (int j = 0; j < nblocks; ++j) {
          if (S.exists(i, j, 1)) {
            Matrix Ubig = get_Ubig(i, 1);
            Matrix Vbig = get_Vbig(j, 1);
            Matrix expected = matmul(matmul(Ubig, S(i, j, 1)), Vbig, false, true);
            Matrix actual = generate_laplacend_matrix(randpts, leaf_size, leaf_size,
                                                      i * leaf_size, j * leaf_size);
            fnorm += pow(norm(actual), 2);
            error += pow(norm(expected - actual), 2);
          }
        }
      }

      return std::sqrt(error/fnorm);
    }
  };
}

int main(int argc, char* argv[]) {
  int64_t N = atoi(argv[1]);
  int64_t rank = atoi(argv[2]);
  int64_t admis = atoi(argv[3]);

  if (admis > 1) {
    std::cout << "This program only supports admis with 0 or 1.\n";
    abort();
  }

  Hatrix::Context::init();
  randvec_t randpts;
  randpts.push_back(Hatrix::equally_spaced_vector(N, 0.0, 1.0 * N)); // 1D
  randpts.push_back(Hatrix::equally_spaced_vector(N, 0.0, 1.0 * N)); // 2D
  randpts.push_back(Hatrix::equally_spaced_vector(N, 0.0, 1.0 * N)); // 3D

  Hatrix::H2 A(randpts, N, rank, admis, 2);
  double error = A.construction_relative_error(randpts);

  Hatrix::Context::finalize();

  std::cout << "N=" << N << " rank=" << rank << " admis="
            << admis <<  " construction error=" << error << std::endl;
}
