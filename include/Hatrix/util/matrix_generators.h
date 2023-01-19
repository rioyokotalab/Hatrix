#pragma once
#include <cstdint>
#include <vector>

#include "Hatrix/classes/Matrix.h"

namespace Hatrix {

template <typename DT = double>
Matrix<DT> generate_random_spd_matrix(int64_t rows, double diag_scale=1.0);

template <typename DT = double>
Matrix<DT> generate_random_matrix(int64_t rows, int64_t cols);

template <typename DT = double>
Matrix<DT> generate_low_rank_matrix(int64_t rows, int64_t cols);

template <typename DT = double>
Matrix<DT> generate_identity_matrix(int64_t rows, int64_t cols);

template <typename DT = double>
Matrix<DT> generate_range_matrix(int64_t rows, int64_t cols, int64_t start_range);

template <typename DT = double>
Matrix<DT> generate_laplacend_matrix(const std::vector<std::vector<double>>& x,
				 int64_t rows, int64_t cols,
				 int64_t row_start, int64_t col_start, double pv=1e-3);

// Sqr. Exp. function copied from stars-H.
template <typename DT = double>
Matrix<DT> generate_sqrexpnd_matrix(const std::vector<std::vector<double>>& x,
                                int64_t rows, int64_t cols,
                                int64_t row_start, int64_t col_start,
                                double beta=0.1, double nu=0.5, double noise=1.e-1,
                                double sigma=1.0);
}  // namespace Hatrix
