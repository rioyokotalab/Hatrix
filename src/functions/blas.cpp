#include "Hatrix/classes/Matrix.h"

#include "cblas.h"

#include <cassert>
#include <cstdlib>


namespace Hatrix {

void gemm(const Matrix& A, const Matrix& B, Matrix& C) {
  cblas_dgemm(
    CblasRowMajor,
    CblasNoTrans, CblasNoTrans,
    A.rows, B.cols, A.cols, 1.0, &A, A.cols, &B, B.cols, 1.0, &C, C.cols
  );
};

void trsm(const Matrix& A, Matrix& B, const char& uplo, const char& lr) {
  switch(uplo) {
  case 'l':
    switch(lr) {
    case 'l':
      cblas_dtrsm(
        CblasRowMajor,
        CblasLeft, CblasLower,
        CblasNoTrans, CblasUnit,
        B.rows, B.cols, 1, &A, A.cols, &B, B.cols
      );
      break;
    case 'r':
      std::abort();
      break;
    }
    break;
  case 'u':
    switch(lr) {
    case 'l':
      assert(B.cols == 1);
      cblas_dtrsm(
        CblasRowMajor,
        CblasLeft, CblasUpper,
        CblasNoTrans, CblasNonUnit,
        B.rows, B.cols, 1, &A, A.cols, &B, B.cols
      );
      break;
    case 'r':
      cblas_dtrsm(
        CblasRowMajor,
        CblasRight, CblasUpper,
        CblasNoTrans, CblasNonUnit,
        B.rows, B.cols, 1, &A, A.cols, &B, B.cols
      );
      break;
    }
    break;
  }
}

} // namespace Hatrix
