namespace Hatrix {

// Note: attempt to implement without stride
// All the storage is column major
template <typename DT = double>
class Dense {
 public:
   // TODO maybe change this datatype
  int rows = 0;
  int cols = 0;

 protected:
  DT* data = nullptr;
  // Shows whether a Matrix is a view of another matrix or actually owns the data
  bool view = false;

 public:
  Dense();

  ~Dense();

  // Copy constructor for Matrix. Behavior depends on whether A is a view or not
  // If A is a view, it creates another view of the same data
  // If A is not a view, the data is copied explicitly
  // This is done to accomodate std::vector#push_back or #emplace_back
  // which call the default copy constructor when they need to grow the storage (and thus copy all the old elements).
  // This assures that the data remains identical in such a case.
  // https://stackoverflow.com/questions/40457302/c-vector-emplace-back-calls-copy-constructor
  Dense(const Dense& A);

  // Copy constructor for Matrix. 
  // If copy is true, a deep copy off all matrix entries is performed
  // A view is created otherwise
  Dense(const Dense& A, const bool copy);

  Dense& operator=(const Dense& A);

  Dense& operator=(Dense&& A);

  Dense(Dense&& A);

  Dense(const int rows, const int cols, const bool init=true);

  const Dense& operator=(const DT a);

  DT* operator&();
  const DT* operator&() const;

  DT& operator()(unsigned int i, unsigned int j);
  const DT& operator()(unsigned int i, unsigned int j) const;

  inline bool is_view() const {return this->view;};
  /*
  // WARNING: does not deallocate the extra data!
  void shrink(int64_t rows, int64_t cols);

  // Split the matrix into n_row_splits * n_col_splits blocks.
  // n_row_splits is the number of blocks in the row dimension and
  // n_col_splits is the number of blocks in the column dimension.
  //
  // All blocks except the last block will be subdivided into equal parts
  // (last block will be different if split dimension is not an exact dimension
  // of the total dimension of the matrix).
  //
  // The vector returns the slices stored in row major order.
  //
  // Example:
  //
  // Matrix A(100, 100);
  // std::vector<Matrix> splits = A.split(4, 4);
  // // splits is now a vector of size 16 containing 16 Matrix objects
  // // of size 25x25 each.
  //
  // Results in:
  //       ____25____50______75_____100
  //      |     |     |      |      |
  //  25  |_____|_____|______|______|
  //      |     |     |      |      |
  //  50  |_____L_____L______|______|
  //      |     |     |      |      |
  //  75  |_____L_____L______|______|
  //      |     |     |      |      |
  //  100 |_____L_____L______|______|
  std::vector<Dense> split(int64_t n_row_splits, int64_t n_col_splits,
                            bool copy = false) const;


  // Split the matrix along the dimensions specifed in each vector
  // in row_split_indices and col_split_indices. This function will
  // return as many tiles that can be generated from the indices supplied.
  // The vector returns the slices stored in row major order.
  //
  // Example:
  // Matrix A(100, 100);
  // std::vector<Matrix> split = A.split({75, 90}, {60, 80});
  //
  // Results in:
  // Example:
  // Matrix A(100, 100);
  // std::vector<Matrix> split = A.split({30, 90}, {60, 80});
  //
  // Results in:
  //       __________________60___80____100
  //      |                  |    |    |
  //      |                  |    |    |
  //      |__________________|____|____|
  // 30   |                  |    |    |
  //      |                  |    |    |
  //      |                  |    |    |
  //      |                  |    |    |
  //      |                  |    |    |
  //      |__________________L____L____|
  // 90   |__________________L____L____|
  // 100
  std::vector<Dense> split(const std::vector<int64_t>& row_split_indices,
                            const std::vector<int64_t>& col_split_indices,
                            bool copy = false) const;

  int64_t min_dim() const;
  int64_t max_dim() const;
  int64_t numel() const;

  void print() const;

  void print_meta() const;

  void read_file(std::string in_file);

  void out_file(std::string out_file) const;

  // Get the size of the memory used by this matrix. If this is a view,
  // this function returns only the memory consumed by the view.
  size_t memory_used() const;

  Dense block_ranks(int64_t nblocks, double accuracy) const;

  Dense swap_rows(const std::vector<int64_t>& row_indices);
  Dense swap_cols(const std::vector<int64_t>& col_indices);

  // Destructively resize.
  void destructive_resize(const int64_t nrows, const int64_t ncols);

  Dense tril(const int64_t diag);*/
 private:
  // does not deallocate memory
  void empty();
  // allocates a new data_ptr of size rows * cols
  void allocate(const bool init);
  // deallocates memory if matrix is not a view and it owns memory
  void deallocate();
  // copies all the values assuming equal sizes
  void copy_values(const Dense& A);
};

}  // namespace Hatrix
