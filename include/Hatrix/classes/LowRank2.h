#pragma once
#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>

#include "Hatrix/classes/Matrix.h"

namespace Hatrix {

enum class Approx {SVD, RSVD};

template <typename DT = double>
class LowRank2 {
 public:
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t rank = 0;
  DT error = 0.0;
  Matrix<DT> U, S, V;

  // Shows whether a Matrix is a view of an object or the actual copy.
  // TODO can a LowRank matrix be a view?
  //bool is_view = false;

 

 public:
  LowRank2() = default;

  ~LowRank2() = default;

  LowRank2(const LowRank2& A) = default;
  //TODO why is this not done with overloading?
  // Copy constructor for Matrix. Create a view object by default. The reason
  // why this is done is mainly to accomodate std::vector#push_back or #emplace_back
  // style functions which call the default copy constructor after they call the
  // move constructor.
  // https://stackoverflow.com/questions/40457302/c-vector-emplace-back-calls-copy-constructor
  //Matrix(const Matrix& A, bool copy);

  template <typename OT>
  explicit LowRank2(const LowRank2<OT>& A);

  LowRank2(const Matrix<DT>& U, const Matrix<DT>& S, const Matrix<DT>& V, bool copy=false);

  LowRank2(Matrix<DT>&& U, Matrix<DT>&& S, Matrix<DT>&& V);

  LowRank2& operator=(const LowRank2& A) = default;

  LowRank2& operator=(LowRank2&& A) = default;

  LowRank2(LowRank2&& A) = default;

  LowRank2(const Matrix<DT>& A, int64_t rank, Approx scheme=Approx::RSVD);

  const DT& operator()(int64_t i, int64_t j) const;

  Matrix<DT> make_dense() const;

  void print() const;

  void print_approx() const;

  DT get_error(const Matrix<DT>& A) const;

  int64_t get_rank(DT error) const;
};

}  // namespace Hatrix
