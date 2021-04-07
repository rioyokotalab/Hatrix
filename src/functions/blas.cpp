#include "Hatrix/classes/Matrix.h"

#include "cblas.h"


namespace Hatrix {

void gemm(const Matrix& A, const Matrix& B, Matrix& C) {
  cblas_dgemm(
    CblasRowMajor,
    CblasNoTrans, CblasNoTrans,
    A.rows, B.cols, A.cols, 1.0, &A, A.cols, &B, B.cols, 1.0, &C, C.cols
  );
};

void trsm(const Matrix& A, Matrix& B, const char& uplo) {
  switch(uplo) {
    case 'l':
      cblas_dtrsm(
        CblasRowMajor,
        CblasLeft, CblasLower,
        CblasNoTrans, CblasUnit,
        B.rows, B.cols, 1, &A, A.cols, &B, B.cols
      );
      break;
    case 'u':
      cblas_dtrsm(
        CblasRowMajor,
        CblasRight, CblasUpper,
        CblasNoTrans, CblasNonUnit,
        B.rows, B.cols, 1, &A, A.cols, &B, B.cols
      );
      break;
  }
}

} // namespace Hatrix
