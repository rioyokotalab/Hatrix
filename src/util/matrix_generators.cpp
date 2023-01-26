#include "Hatrix/util/matrix_generators.h"
#include "Hatrix/functions/blas.h"

#include <cmath>
#include <cstdint>
#include <random>

namespace Hatrix {

template <typename DT>
Matrix<DT> generate_random_matrix(int64_t rows, int64_t cols) {
  //TODO pass seed?
  std::mt19937 gen(100);
  //TODO should this adapt to the template type
  std::uniform_real_distribution<DT> dist(0.0, 1.0);
  Matrix<DT> out(rows, cols);
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      out(i, j) = dist(gen);
    }
  }
  return out;
}

template <typename DT>
Matrix<DT> generate_random_spd_matrix(int64_t rows, DT diag_scale) {
  Matrix<DT> A = generate_random_matrix<DT>(rows, rows);
  Matrix<DT> SPD = matmul(A, A, true, false);
  for (int i = 0; i < rows; ++i) { A(i,i) *= diag_scale; }

  return SPD;
}

template <typename DT>
Matrix<DT> generate_range_matrix(int64_t rows, int64_t cols, int64_t start_range) {
  Matrix<DT> out(rows, cols);
  int num = start_range;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      out(i, j) = num;
      num++;
    }
  }

  return out;
}

template <typename DT>
Matrix<DT> generate_low_rank_matrix(int64_t rows, int64_t cols) {
  // TODO: Might want more sophisticated method, specify rate of decay of
  // singular values etc...
  Matrix<DT> out(rows, cols);
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      out(i, j) = 1.0 / std::abs(i - j + out.max_dim());
    }
  }
  return out;
}

template <typename DT>
Matrix<DT> generate_identity_matrix(int64_t rows, int64_t cols) {
  Matrix<DT> out(rows, cols);
  for (int64_t i = 0; i < out.min_dim(); ++i) {
    out(i, i) = 1.;
  }
  return out;
}

template <typename DT>
Matrix <DT> generate_laplacend_matrix(const std::vector<std::vector<double>>& x,
				 int64_t rows, int64_t cols,
				 int64_t row_start, int64_t col_start, double pv) {
  Matrix<DT> out(rows, cols);
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

template <typename DT>
Matrix<DT> generate_sqrexpnd_matrix(const std::vector<std::vector<double>>& x,
                                int64_t rows, int64_t cols,
                                int64_t row_start, int64_t col_start,
                                double beta, double nu, double noise,
                                double sigma) {
  Matrix<DT> out(rows, cols);
  double value;

  for(int64_t i = 0; i < rows; i++) {
    for(int64_t j = 0; j < cols; j++) {
      double dist = 0, temp;

      for (unsigned k = 0; k < x.size(); ++k) {
        temp = x[k][i + row_start] - x[k][j + col_start];
        dist += temp * temp;
      }
      dist = dist / beta;
      if (dist == 0) {
        value = sigma + noise;
      }
      else {
        value = sigma * exp(dist);
      }

      out(i, j) = value;
    }
  }

  return out;

}

// explicit instantiation (these are the only available data-types)
template Matrix<float> generate_random_matrix(int64_t rows, int64_t cols);
template Matrix<double> generate_random_matrix(int64_t rows, int64_t cols);

template Matrix<float> generate_random_spd_matrix(int64_t rows, float diag_scale);
template Matrix<double> generate_random_spd_matrix(int64_t rows, double diag_scale);

template Matrix<float> generate_low_rank_matrix(int64_t rows, int64_t cols);
template Matrix<double> generate_low_rank_matrix(int64_t rows, int64_t cols);

template Matrix<float> generate_identity_matrix(int64_t rows, int64_t cols);
template Matrix<double> generate_identity_matrix(int64_t rows, int64_t cols);

template Matrix<float> generate_range_matrix(int64_t rows, int64_t cols, int64_t start_range);
template Matrix<double> generate_range_matrix(int64_t rows, int64_t cols, int64_t start_range);

template Matrix<float> generate_laplacend_matrix(const std::vector<std::vector<double>>& x,
				 int64_t rows, int64_t cols,
				 int64_t row_start, int64_t col_start, double pv);
template Matrix<double> generate_laplacend_matrix(const std::vector<std::vector<double>>& x,
				 int64_t rows, int64_t cols,
				 int64_t row_start, int64_t col_start, double pv);

template Matrix<float> generate_sqrexpnd_matrix(const std::vector<std::vector<double>>& x,
                                int64_t rows, int64_t cols,
                                int64_t row_start, int64_t col_start,
                                double beta, double nu, double noise,
                                double sigma);
template Matrix<double> generate_sqrexpnd_matrix(const std::vector<std::vector<double>>& x,
                                int64_t rows, int64_t cols,
                                int64_t row_start, int64_t col_start,
                                double beta, double nu, double noise,
                                double sigma);

}  // namespace Hatrix
