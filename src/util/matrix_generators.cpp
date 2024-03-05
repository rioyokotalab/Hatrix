#include "Hatrix/util/matrix_generators.hpp"
#include "Hatrix/functions/blas.hpp"

#include <cmath>
#include <cstdint>
#include <random>

namespace Hatrix {

Matrix generate_random_spd_matrix(int64_t rows, double diag_scale) {
  Matrix A = generate_random_matrix(rows, rows);
  Matrix SPD = matmul(A, A, true, false);
  for (int i = 0; i < rows; ++i) { A(i,i) *= diag_scale; }

  return SPD;
}

Matrix generate_random_matrix(int64_t rows, int64_t cols) {
  std::mt19937 gen(100);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  Matrix out(rows, cols);
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      out(i, j) = dist(gen);
    }
  }
  return out;
}

Matrix generate_range_matrix(int64_t rows, int64_t cols, int64_t start_range) {
  Matrix out(rows, cols);
  int num = start_range;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      out(i, j) = num;
      num++;
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

Matrix generate_p2p_interactions(const Domain& domain,
                                 const int64_t irow, const int64_t nrows,
                                 const int64_t jcol, const int64_t ncols,
                                 const greens_functions::kernel_function_t& kernel) {
  Matrix out(nrows, ncols);

#pragma omp parallel for
  for (int64_t i = 0; i < nrows; ++i) {
#pragma omp parallel for
    for (int64_t j = 0; j < ncols; ++j) {
      out(i, j) = kernel(domain.particles[i + irow].coords,
                         domain.particles[j + jcol].coords);
    }
  }

  return out;
}

Matrix generate_p2p_interactions(const Domain& domain,
                                 const greens_functions::kernel_function_t& kernel) {
  int64_t rows =  domain.particles.size();
  int64_t cols =  domain.particles.size();
  Matrix out(rows, cols);

#pragma omp parallel for
  for (int64_t i = 0; i < rows; ++i) {
#pragma omp parallel for
    for (int64_t j = 0; j < cols; ++j) {
      out(i, j) = kernel(domain.particles[i].coords,
                         domain.particles[j].coords);
    }
  }

  return out;
}

Matrix generate_laplacend_matrix(const std::vector<std::vector<double>>& x,
				 int64_t rows, int64_t cols,
				 int64_t row_start, int64_t col_start, double pv) {
  Matrix out(rows, cols);
  for(int64_t i = 0; i < rows; i++) {
    for(int64_t j = 0; j < cols; j++) {
      double rij = 0.0;
      for(unsigned k = 0; k < x.size(); k++) {
	rij += ((x[k][i+row_start] - x[k][j+col_start]) *
		(x[k][i+row_start] - x[k][j+col_start]));
      }
      out(i, j) = 1 / (std::sqrt(rij) + pv);
    }
  }
  return out;
}
}  // namespace Hatrix
