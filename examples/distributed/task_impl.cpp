#include "Hatrix/Hatrix.h"
#include "distributed/distributed.hpp"

using namespace Hatrix;


// Really dumb wrapper over Matrix.
class MatrixWrapper : public Hatrix::Matrix {
public:
  MatrixWrapper(double* data, int64_t _rows, int64_t _cols, int64_t _stride) {
    data_ptr = data;
    rows = _rows;
    cols = _cols;
    stride = _stride;
  }

  void copy_mem(const Matrix& A) {
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        (*this)(i, j) = A(i, j);
      }
    }
  }

  ~MatrixWrapper() {
    data_ptr = nullptr;
    rows = -1;
    cols = -1;
    stride = -1;
  }
};

void CORE_multiply_complement(int64_t D_nrows, int64_t D_ncols, int64_t D_row_rank, int64_t D_col_rank,
                              int64_t U_nrows, int64_t U_ncols, double* _D, double* _U, char which) {

}
