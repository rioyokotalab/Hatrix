#include "Hatrix/util/matrix_generators.h"

#include <cmath>
#include <cstdint>
#include <random>

namespace Hatrix {

Matrix generate_random_matrix(int64_t rows, int64_t cols) {
  std::random_device rd;
  std::mt19937 gen(rd());
  // gen.seed(0);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  Matrix out(rows, cols);
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      out(i, j) = dist(gen);
    }
  }
  return out;
}

Matrix generate_low_rank_matrix(int64_t rows, int64_t cols) {
  // TODO: Might want more sophisticated method, specify rate of decay of
  // singular values etc...
  Matrix out(rows, cols);
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      out(i, j) = 1.0 / std::abs(i - j + out.max_dim());
    }
  }
  return out;
}

Matrix generate_identity_matrix(int64_t rows, int64_t cols) {
  Matrix out(rows, cols);
  for (int64_t i = 0; i < out.min_dim(); ++i) {
    out(i, i) = 1.;
  }
  return out;
}

Matrix generate_laplacend_matrix(std::vector<std::vector<double>>& x,
				 int64_t rows, int64_t cols,				   
				 int64_t row_start, int64_t col_start) {
  Matrix out(rows, cols);
  for(int64_t i = 0; i < rows; i++) {
    for(int64_t j = 0; j < cols; j++) {
      double rij = 0.0;
      for(int64_t k = 0; k < x.size(); k++) {
	rij += ((x[k][i+row_start] - x[k][j+col_start]) *
		(x[k][i+row_start] - x[k][j+col_start]));
      }
      out(i, j) = 1 / (std::sqrt(rij) + 1e-3);
    }
  }
  return out;
}


#include "build_tree.h"
#include "kernel.h"
#include "test_util.h"
#include "timer.h"

Hatrix::Matrix convert(const nbd::Matrix& m) {
  Hatrix::Matrix out(m.M, m.N);
  for (int64_t j = 0; j < m.N; j++)
  for (int64_t i = 0; i < m.M; i++) {
    out(i, j) = m.A[i + m.LDA * j];
  }
  return out;
}

BLR construct_BLR(int64_t block_size, int64_t n_blocks, int64_t rank) {

  BLR A;

  int dim = 2;
  int m = block_size * n_blocks;
  int leaf = block_size;
  int p = 20;
  double theta = 1.01; // weak
  auto fun = dim == 2 ? l2d() : l3d();

  nbd::start("bodies");
  nbd::Bodies b1(m);
  nbd::initRandom(b1, m, dim, 0., 1., 0);
  nbd::stop("bodies");

  nbd::start("build tree");
  nbd::Cells c1 = nbd::getLeaves(nbd::buildTree(b1, leaf, dim));
  nbd::stop("build tree");

  nbd::start("build H");
  nbd::Matrices d, bi;
  nbd::traverse(fun, c1, c1, dim, d, theta, rank);
  nbd::stop("build H");

  nbd::start("build H2");
  nbd::traverse_i(c1, c1, d, bi, p);
  nbd::shared_epilogue(d);
  nbd::stop("build H2");

  Hatrix::Matrix U, S, V;
  for (int64_t i = 0; i < n_blocks; ++i) {
    U = convert(bi[i + 1]);
    V = convert(bi[i + 1]);
    A.U.insert(i, std::move(U));
    A.V.insert(i, std::move(V));
  }

  for (int64_t i = 0; i < n_blocks; ++i) {
    for (int64_t j = 0; j < n_blocks; ++j) {
      nbd::Matrix& s = d[(i + 1) + (j + 1) * c1.size()];
      if (s.M > 0 && s.N > 0) {
        S = convert(s);
        A.S.insert(i, j, std::move(S));
      }
    }
  }

  auto i_begin = c1[0].BODY;
  Hatrix::Matrix D;
  for (int y = 0; y < c1.size(); y++) {
    auto i = c1[y];
    auto yi = i.BODY - i_begin;
    for (auto& j : i.listNear) {
      auto _x = j - &jcells[0];
      auto xi = j->BODY - j_begin;
      nbd::Matrix m;
      nbd::P2Pnear(ef, &icells[y], &jcells[_x], dim, m);
      
      D = convert(m);
      A.D.insert(i - 1, j - 1, std::move(D));
    }
  }

  return A;
}

}  // namespace Hatrix
