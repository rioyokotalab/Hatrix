#pragma once
#include "Hatrix/classes/Matrix.h"

#include <cstdint>
using std::int64_t;


namespace Hatrix {

Matrix generate_random_matrix(int64_t rows, int64_t cols);

Matrix generate_low_rank_matrix(int64_t rows, int64_t cols);

Matrix generate_identity_matrix(int64_t rows, int64_t cols);

} // namespace Hatrix
