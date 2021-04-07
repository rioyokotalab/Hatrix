#include "Hatrix/classes/Matrix.h"

#include "mkl.h"

#include <cassert>
#include <cstdlib>


namespace Hatrix {

void gemm(const Matrix& A, const Matrix& B, Matrix& C) {
  char N = 'N';
  double alpha = 1., beta = 1.;
  dgemm(&N, &N, &C.rows, &C.cols, &A.cols, &alpha, A.data_, &A.rows, B.data_, &B.rows, &beta, C.data_, &C.rows);
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
