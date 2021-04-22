namespace Hatrix {

class Matrix {
 public:
  double* data_;
  int rows, cols;

  Matrix() = delete;

  ~Matrix();

  Matrix(int rows, int cols);

  Matrix(const Matrix& A);

  Matrix& operator=(const Matrix& A);

  Matrix(Matrix&& other);

  Matrix& operator=(Matrix&& A);

  const Matrix& operator=(const double a);

  double* operator&();
  const double* operator&() const;

  double& operator()(int i, int j);
  const double& operator()(int i, int j) const;

  void shrink(int rows, int cols);
  int min_dim();
};

} // namespace Hatrix
