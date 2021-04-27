#pragma once
#include <cstdint>
using std::int64_t;


namespace Hatrix {

class Matrix;

void gesv_IR(Matrix &A, Matrix &b, int64_t max_iter);

} // namespace Hatrix