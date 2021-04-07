namespace Hatrix {

class Matrix;

void gemm(
  const Matrix& A, const Matrix& B, Matrix& C,
  char transa, char transb, double alpha, double beta
);

void trsm(
  const Matrix& A, Matrix& B,
  char side, char uplo, char transa, char diag, double alpha
);

} // namespace Hatrix
