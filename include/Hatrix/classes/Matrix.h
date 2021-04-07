namespace Hatrix {

class Matrix {
 public:
  double* data_;
  int rows, cols;

  Matrix() = delete;

  ~Matrix();

  Matrix(int rows, int cols);

  Matrix(const Matrix& A);

  const Matrix& operator=(const double a);

  double* operator&();
  const double* operator&() const;

  double& operator()(int i, int j);
  const double& operator()(int i, int j) const;

  void shrink(int rows, int cols);
};

} // namespace Hatrix
