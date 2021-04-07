namespace Hatrix {

class Matrix;

void gemm(const Matrix& A, const Matrix& B, Matrix& C);

void trsm(const Matrix& A, Matrix& B, const char& uplo, const char& lr);

} // namespace Hatrix
