namespace Hatrix {

enum {
  TRSMLeft,
  TRSMRight,
  TRSMUpper,
  TRSMLower
};

class Matrix;

void gemm(
  const Matrix& A, const Matrix& B, Matrix& C,
  bool transA, bool transB, double alpha, double beta
);

void trsm(
  const Matrix& A, Matrix& B,
  int side, int uplo, bool transA, bool diag, double alpha
);

} // namespace Hatrix
