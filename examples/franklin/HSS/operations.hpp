#include "SymmetricSharedBasisMatrix.hpp"

using namespace Hatrix;

void factorize(SymmetricSharedBasisMatrix& A);
Matrix solve(const SymmetricSharedBasisMatrix& A, const Matrix& x);
Matrix matmul(const SymmetricSharedBasisMatrix& A, const Matrix& x);
