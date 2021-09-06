#pragma once
#include <cstdint>
#include <vector>

#include "Hatrix/classes/Matrix.h"
#include "Hatrix/classes/BLR.h"

namespace Hatrix {

Matrix generate_random_matrix(int64_t rows, int64_t cols);

Matrix generate_low_rank_matrix(int64_t rows, int64_t cols);

Matrix generate_identity_matrix(int64_t rows, int64_t cols);

Matrix generate_laplacend_matrix(std::vector<std::vector<double>>& x,
				 int64_t rows, int64_t cols,
				 int64_t row_start, int64_t col_start);

BLR construct_BLR(int64_t block_size, int64_t n_blocks, int64_t rank);

}  // namespace Hatrix
