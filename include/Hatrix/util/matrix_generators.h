#pragma once
#include "Hatrix/classes/Matrix.h"

#include <string>
#include <cstdint>
using std::int64_t;


namespace Hatrix {

Matrix generate_random_matrix(int64_t rows, int64_t cols);

Matrix generate_low_rank_matrix(int64_t rows, int64_t cols);

Matrix generate_from_csv(std::string filename, char delimiter=',', int64_t rows=0, int64_t cols=0);

} // namespace Hatrix
