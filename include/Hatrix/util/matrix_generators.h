#pragma once
#include <cstdint>
#include <vector>

#include "Hatrix/classes/Matrix.h"

namespace Hatrix {

Matrix generate_random_matrix(int64_t rows, int64_t cols);

Matrix generate_low_rank_matrix(int64_t rows, int64_t cols);

Matrix generate_identity_matrix(int64_t rows, int64_t cols);

Matrix generate_range_matrix(int64_t rows, int64_t cols, int64_t start_range);

Matrix generate_laplacend_matrix(const std::vector<std::vector<double>>& x,
				 int64_t rows, int64_t cols,
				 int64_t row_start, int64_t col_start, double pv=1e-3);

// Sqr. Exp. function copied from stars-H.
Matrix generate_sqrexpnd_matrix(const std::vector<std::vector<double>>& x,
                                int64_t rows, int64_t cols,
                                int64_t row_start, int64_t col_start,
                                double beta=0.1, double nu=0.5, double noise=1.e-1,
                                double sigma=1.0);
}  // namespace Hatrix
