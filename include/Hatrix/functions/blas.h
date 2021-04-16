namespace Hatrix {

enum {
  TRSMLeft,
  TRSMRight,
  TRSMUpper,
  TRSMLower
};

class Matrix;

void matmul(
  const Matrix& A, const Matrix& B, Matrix& C,
  bool transA=false, bool transB=false, double alpha=1.0, double beta=1.0
);

void solve_triangular(
  const Matrix& A, Matrix& B,
  int side, int uplo, bool diag, bool transA=false, double alpha=1.0
);

} // namespace Hatrix
