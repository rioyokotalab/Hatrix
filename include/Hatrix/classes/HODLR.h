#pragma once
#include "Hatrix/classes/IndexedMap.h"
#include "Hatrix/classes/Matrix.h"
#include "Hatrix/classes/LowRank.h"
#include "Hatrix/functions/math_common.h"

#include <omp.h>

namespace Hatrix {

template <typename DT>
class HODLR {
 public:
  int leaf_size;
  int max_level;
  int rank;
  RowColLevelMap<bool> is_admissible;
  RowColLevelMap<Matrix<DT>> dense;
  RowColLevelMap<LowRank<DT>> low_rank;

  HODLR(const Matrix<DT>& A, int leaf_size, int rank);
  ~HODLR() = default;

  void lu();
  void solve(Matrix<DT>& B) const;
  Matrix<DT> make_dense() const;
  void matmul(const Matrix<DT>& B, Matrix<DT>& C, double alpha=1, double beta=0) const;

 private:
  void add_admissibility(int row=0, int col=0, int level=0);
  bool insert_block(int row, int col, int level);
  void add_children(int row, int col, int level);
  void add_dense_blocks(const Matrix<DT>& A, omp_lock_t& lock);
  void add_lr_block(const Matrix<DT>& A, omp_lock_t& lock, int row=0, int col=0, int level=0);
  void spawn_lr_children(int row, int col, int level);
  void empty_task();
  void update_children(int row, int col, int level);
  void update_parent(int row, int col, int level);
  void trsm(int row, int col, int level, Side side, Mode uplo);
  void add(int row, int col, int level, LowRank<DT>& temp, int block_start, int block_size);
  void matmul(int row, int col, int level, LowRank<DT>& temp);
  void getrf(int row, int col, int level, LowRank<DT>& temp);
  void trsm_solve(int row, int col, int level, Matrix<DT>& B, Side side, Mode uplo) const;
  void materialize(Matrix<DT>& A, int row, int col, int level) const;
  void materialize_low_rank(Matrix<DT>& A, int row, int col, int level) const;
  void matmul(int row, int col, int level, const Matrix<DT>& B, Matrix<DT>& C, double alpha, double beta) const;

};

}  // namespace Hatrix
