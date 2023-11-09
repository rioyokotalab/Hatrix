#include "Hatrix/classes/Hmatrix.h"
#include "Hatrix/classes/Matrix.h"
#include "Hatrix/classes/LowRank2.h"
#include "Hatrix/functions/lapack.h"
#include "Hatrix/functions/blas.h"
#include "Hatrix/functions/arithmetics.h"
#include "Hatrix/functions/math_common.h"

#include <omp.h>
#include <vector>
#include <iostream>

namespace Hatrix {

template<typename DT> template<typename OT>
Hmatrix<DT>::Hmatrix(const Hmatrix<OT>& A): leaf_size(A.leaf_size), max_level(A.max_level), rank(A.rank) {
  //TODO is this a good solution?
  omp_lock_t lr_lock;
  omp_init_lock(&lr_lock);
  omp_lock_t dense_lock;
  omp_init_lock(&dense_lock);

  #pragma omp parallel
  {
    #pragma omp single 
    {
      #pragma omp task shared(is_admissible, A)
      {
        is_admissible.deep_copy(A.is_admissible);
      }
      add_dense_blocks(A, dense_lock);
      add_lr_block(A, lr_lock);     
    }
  }

}

template <typename DT>
Hmatrix<DT>::Hmatrix(const Matrix<DT>& A, int leaf_size, int rank) : leaf_size(leaf_size), rank(rank), max_level(0) {
  int size = A.rows;
  while (leaf_size < size){
    size = size>>1;
    max_level++;
  }
  add_admissibility();
  //TODO is this a good solution?
  omp_lock_t lr_lock;
  omp_init_lock(&lr_lock);
  omp_lock_t dense_lock;
  omp_init_lock(&dense_lock);
      
  #pragma omp parallel
  {
    #pragma omp single 
    {
      add_dense_blocks(A, dense_lock);
      add_lr_block(A, lr_lock);     
    }
  }
  //add_lr_block(A, lr_lock); 
}

template<typename DT>
void Hmatrix<DT>::spawn_lr_children(int row, int col, int level) {
  if (level >= max_level)
    return;

  int child_row = row * 2;
  int child_col = col * 2;
  // S and non-split U/V are currently not used, but might be useful in the future for GEMM split
  if (col > row) {
    std::vector<Matrix<DT>> children = low_rank(row, col, level).U.split(2,1);
    low_rank.insert(child_row, child_col, level+1, LowRank2<DT>(children[0], low_rank(row, col, level).S, low_rank(row, col, level).V, false));
    spawn_lr_children(child_row, child_col, level+1);
    low_rank.insert(child_row+1, child_col, level+1, LowRank2<DT>(children[1], low_rank(row, col, level).S, low_rank(row, col, level).V, false));
    spawn_lr_children(child_row+1, child_col, level+1);
  } else {
    std::vector<Matrix<DT>> children = low_rank(row, col, level).V.split(1,2);
    low_rank.insert(child_row, child_col, level+1, LowRank2<DT>(low_rank(row, col, level).U, low_rank(row, col, level).S, children[0], false));
    spawn_lr_children(child_row, child_col, level+1);
    low_rank.insert(child_row, child_col+1, level+1, LowRank2<DT>(low_rank(row, col, level).U, low_rank(row, col, level).S, children[1], false));
    spawn_lr_children(child_row, child_col+1, level+1);
  }
}
  
template <typename DT>
void Hmatrix<DT>::add_lr_block(const Matrix<DT>& A, omp_lock_t& lock, int row, int col, int level) {
  if (A.rows < leaf_size)
    return;

  if (is_admissible(row, col, level)) {
    #pragma omp task shared(low_rank, lock)
    {
      LowRank2<DT> LR(A, rank);
      omp_set_lock(&lock);
      low_rank.insert(row, col, level, std::move(LR));
      spawn_lr_children(row, col, level);
      omp_unset_lock(&lock);
    }
  } else {
    int start = row * 2;
    std::vector<Hatrix::Matrix<DT>> A_split = A.split(2, 2);
    add_lr_block(A_split[0], lock, start, start, level+1);
    add_lr_block(A_split[1], lock, start, start+1, level+1);
    add_lr_block(A_split[2], lock, start+1, start, level+1);
    add_lr_block(A_split[3], lock, start+1, start+1, level+1);
  }
}

template <typename DT>
void Hmatrix<DT>::add_dense_blocks(const Matrix<DT>& A, omp_lock_t& lock) {
  int num_blocks = A.rows / leaf_size;

  for (int i=0; i<num_blocks; ++i) {
    #pragma omp task shared(dense, A, lock)
    {
      Matrix<DT> block_copy(leaf_size, leaf_size);
      for (int k=0; k<leaf_size; ++k) {
        for (int l=0; l<leaf_size; ++l) {
          block_copy(k,l) = A(i*leaf_size+k, i*leaf_size+l);
        }
      }
      omp_set_lock(&lock);
      dense.insert(i, i, max_level, std::move(block_copy));
      omp_unset_lock(&lock);
    }
  }
}

template <typename DT> template <typename OT>
void Hmatrix<DT>::add_dense_blocks(const Hmatrix<OT>& A, omp_lock_t& lock) {
  size_t num_blocks = A.dense.size();

  for (size_t i=0; i<num_blocks; ++i) {
    #pragma omp task shared(dense, A, lock)
    {
      Matrix<DT> block_copy(A.dense(i, i, max_level));
      omp_set_lock(&lock);
      dense.insert(i, i, max_level, std::move(block_copy));
      omp_unset_lock(&lock);
    }
  }
}

template <typename DT> template <typename OT>
void Hmatrix<DT>::add_lr_block(const Hmatrix<OT>& A, omp_lock_t& lock, int row, int col, int level) {
  if (level > max_level)
    return;

  if (A.is_admissible(row, col, level)) {
    #pragma omp task shared(low_rank, A, lock)
    {
      LowRank2<DT> LR(A.low_rank(row, col, level));
      omp_set_lock(&lock);
      low_rank.insert(row, col, level, std::move(LR));
      spawn_lr_children(row, col, level);
      omp_unset_lock(&lock);
    }
  } else {
    int start = row * 2;
    add_lr_block(A, lock, start, start, level+1);
    add_lr_block(A, lock, start, start+1, level+1);
    add_lr_block(A, lock, start+1, start, level+1);
    add_lr_block(A, lock, start+1, start+1, level+1);
  }
}

template <typename DT>
bool Hmatrix<DT>::insert_block(int row, int col, int level) {
  is_admissible.insert(row, col, level, row!=col);
  return row!=col;
}

template <typename DT>
void Hmatrix<DT>::add_admissibility(int row, int col, int level) {
  if (level > max_level)
    return;
  // add block
  if (!insert_block(row, col, level)) {
    // call recursively for children
    int start = row * 2;
    add_admissibility(start, start, level+1);
    add_admissibility(start, start+1, level+1);
    add_admissibility(start+1, start, level+1);
    add_admissibility(start+1, start+1, level+1);
  } else {
    add_children(row, col, level+1);
  }
}

template <typename DT>
void Hmatrix<DT>::add_children(int row, int col, int level) {
  if (level > max_level)
    return;

  row *= 2;
  col *= 2;
  if (col > row) {
    is_admissible.insert(row, col, level, true);
    add_children(row, col, level+1);
    is_admissible.insert(row+1, col, level, true);
    add_children(row+1, col, level+1);
  } else {
    //no diagonal blocks are addmissible
    is_admissible.insert(row, col, level, true);
    add_children(row, col, level+1);
    is_admissible.insert(row, col+1, level, true);
    add_children(row, col+1, level+1);
  }
}

template <typename DT>
void Hmatrix<DT>::empty_task() {
    return;
}

template <typename DT>
void Hmatrix<DT>::update_children(int row, int col, int level) {
  // has no children
  if (level >= max_level)
    return;

  // creates a dependency with the parent as input and the children as output
  // i.e. the parent updates all his children
  int child_row = row * 2;
  int child_col = col * 2;
  if (row > col) {
    #pragma omp task depend(in: is_admissible(row, col, level)) depend(out: is_admissible(child_row, child_col, level+1), is_admissible(child_row, child_col+1, level+1))
    {
      empty_task();
    }
    update_children(child_row, child_col, level+1);
    update_children(child_row, child_col+1, level+1);
  } else {
    #pragma omp task depend(in: is_admissible(row, col, level)) depend(out: is_admissible(child_row, child_col, level+1), is_admissible(child_row+1, child_col, level+1))
    {
      empty_task();
    }
    update_children(child_row, child_col, level+1);
    update_children(child_row+1, child_col, level+1);
  }
}

template <typename DT>
void Hmatrix<DT>::update_parent(int row, int col, int level) {
  // has no children
  if (level >= max_level)
    return;

  // creates a dependency with the children as input and the parent as output
  // i.e. the children update their parent
  int child_row = row * 2;
  int child_col = col * 2;
  if (row > col) {
    // Must go from bottom to top
    update_parent(child_row, child_col, level+1);
    update_parent(child_row, child_col+1, level+1);
    #pragma omp task depend(in: is_admissible(child_row, child_col, level+1), is_admissible(child_row, child_col+1, level+1)) depend(out: is_admissible(row, col, level))
    {
      empty_task();
    }
  } else {
    // Must go from bottom to top
    update_parent(child_row, child_col, level+1);
    update_parent(child_row+1, child_col, level+1);
    #pragma omp task depend(in: is_admissible(child_row, child_col, level+1), is_admissible(child_row+1, child_col, level+1)) depend(out: is_admissible(row, col, level))
    {
      empty_task();
    }
  }
}

template <typename DT>
void Hmatrix<DT>::trsm(int row, int col, int level, Side side, Mode uplo) {
  if (level < max_level) {
    if (uplo == Hatrix::Upper) {
      int start = col * 2;
      //std::cout<<"TRSM1"<<std::endl;
      trsm(row*2, start, level+1, side, uplo);
      update_parent(row*2, start, level+1);
      update_parent(start, start+1, level+1);
      #pragma omp task shared(low_rank) depend(in: is_admissible(row*2, start, level+1), is_admissible(start, start+1, level+1)) depend(inout: is_admissible(row*2, start+1, level+1))
      {  //std::cout<<"GEMM (t-task)"<<row*2<<","<<start+1<<","<<level+1<<std::endl;
        Hatrix::matmul(low_rank(row*2, start, level+1).V, low_rank(start, start+1, level+1), low_rank(row*2, start+1, level+1).V, false, false, -1, 1);
      }
      update_children(row*2, start+1, level+1);
      trsm(row*2, start+1, level+1, side, uplo);
    } else {
      int start = row * 2;
      //std::cout<<"TRSM1"<<std::endl;
      trsm(start, col*2, level+1, side, uplo);
      update_parent(start+1, start, level+1);
      update_parent(start, col*2, level+1);
      #pragma omp task shared(low_rank) depend(in: is_admissible(start, col*2, level+1), is_admissible(start+1, start, level+1)) depend(inout: is_admissible(start+1, col*2, level+1))
      { //std::cout<<"GEMM (t-task)"<<start+1<<","<<col*2<<","<<level+1<<std::endl;
        Hatrix::matmul(low_rank(start+1, start, level+1), low_rank(start, col*2, level+1).U, low_rank(start+1, col*2, level+1).U, false, false, -1, 1);
      }
      update_children(start+1, col*2, level+1);
      trsm(start+1, col*2, level+1, side, uplo);
    }
  } else {
    bool diag = uplo == Hatrix::Lower;
    int src_idx = diag?row:col;
    //std::cout<<"TRSM "<<row<<","<<col<<","<<level<<std::endl;
    #pragma omp task shared(dense, low_rank) depend(in: is_admissible(src_idx, src_idx, level)) depend(inout: is_admissible(row, col, level))
    { //std::cout<<"TRSM (task) "<<row<<","<<col<<","<<level<<std::endl;
      Hatrix::solve_triangular(dense(src_idx, src_idx, level), diag?low_rank(row, col, level).U:low_rank(row, col, level).V, side, uplo, diag, false);
    }
    //std::cout<<"Finished"<<std::endl;
  }
}

template <typename DT>
void Hmatrix<DT>::add(int row, int col, int level, LowRank2<DT>& temp, int block_start, int block_size) {
    if (level < max_level) {
      int start = row * 2;
      block_size = block_size/2;
      add(start, start, level+1, temp, block_start, block_size);
      add(start+1, start+1, level+1, temp, block_start+block_size, block_size);

      // off-diagonal blocks
      #pragma omp task shared(low_rank, temp) depend(inout: is_admissible(start, start+1, level+1)) depend(in: temp)
      {
        Matrix<DT> U = temp.U.get_row_block(block_start, block_size);
        Matrix<DT> V = temp.V.get_col_block(block_start+block_size, block_size);
        LowRank2<DT> T(U, temp.S, V, false);
        low_rank(start, start+1, level+1) += T;
      }
      update_children(start, start+1, level+1);
      #pragma omp task shared(low_rank, temp) depend(inout: is_admissible(start+1, start, level+1)) depend(in: temp)
      {
        Matrix<DT> U = temp.U.get_row_block(block_start+block_size, block_size);
        Matrix<DT> V = temp.V.get_col_block(block_start, block_size);
        LowRank2<DT> T(U, temp.S, V, false);
        low_rank(start+1, start, level+1) += T;
      }
      update_children(start+1, start, level+1);
    } else { //this must be a dense block
      #pragma omp task shared(dense, temp) depend(inout: is_admissible(row, col, level)) depend(in: temp)
      {
        Matrix<DT> U = temp.U.get_row_block(block_start, block_size);
        Matrix<DT> V = temp.V.get_col_block(block_start, block_size);
        LowRank2<DT> T(U, temp.S, V, false);
        dense(row, col, level) += T;
      }
    }
  }

template <typename DT>
void Hmatrix<DT>::matmul(int row, int col, int level, Hatrix::LowRank2<DT>& temp) {
  if (level < max_level) {
    // wait for all tasks on children of the input low-rank blocks
    update_parent(row, col-1, level);
    update_parent(row-1, col, level);
    #pragma omp task shared(dense, low_rank, temp) depend(in: is_admissible(row, col-1, level), is_admissible(row-1, col, level)) depend(out: temp)
    {  
      Hatrix::matmul(low_rank(row, col-1, level), low_rank(row-1, col, level), temp, false, false, -1, 1, false);
    }
    // can't infer the row number from temp, since processing might not be finished yet
    add(row, col, level, temp, 0, low_rank(row, col-1, level).rows);
  } else {
    #pragma omp task shared(dense, low_rank) depend(in: is_admissible(row, col-1, level), is_admissible(row-1, col, level)) depend(inout: is_admissible(row, col, level))
    {//std::cout<<"GEMM (matmul task) on ("<<row<<","<<col<<","<<level<<")"<<std::endl;
      Hatrix::matmul(low_rank(row, col-1, level), low_rank(row-1, col, level), dense(row, col, level), false, false, -1, 1);
    }
  }
}

template <typename DT>
void Hmatrix<DT>::getrf(int row, int col, int level, Hatrix::LowRank2<DT>& temp) {
  int start = row * 2;
  if (level < max_level) {
    getrf(start, start, level+1, temp);
    trsm(start, start+1, level+1, Hatrix::Left, Hatrix::Lower);
    trsm(start+1, start, level+1, Hatrix::Right, Hatrix::Upper);
    matmul(start+1, start+1, level+1, temp);
    getrf(start+1, start+1, level+1, temp);
  } else {
    #pragma omp task shared(dense) depend(inout: is_admissible(row, col, level))
      {//std::cout<<"GETRF(task) on ("<<row<<","<<col<<","<<level<<")"<<std::endl;
        Hatrix::lu_nopiv(dense(row, col, level));
      }
  }
}

template <typename DT>
void Hmatrix<DT>::lu() {
  Hatrix::LowRank2<DT> temp;
  #pragma omp parallel
  {
    #pragma omp single 
    {
      getrf(0, 0, 0, temp);
    }
  }
}

template <typename DT>
void Hmatrix<DT>::solve(Matrix<DT>& B) const {
  trsm_solve(0, 0, 0, B, Hatrix::Left, Hatrix::Lower);
  trsm_solve(0, 0, 0, B, Hatrix::Left, Hatrix::Upper);
}

template <typename DT>
void Hmatrix<DT>::trsm_solve(int row, int col, int level, Matrix<DT>& B, Side side, Mode uplo) const {
  int start = row * 2;
  if (level < max_level) {
    std::vector<Hatrix::Matrix<DT>> B_split;
    B_split = B.split(2, 1);
    if (uplo == Hatrix::Upper) {
      trsm_solve(start+1, start+1, level+1, B_split[1], side, uplo);
      Hatrix::matmul(low_rank(start, start+1, level+1), B_split[1], B_split[0], false, false, -1, 1);
      trsm_solve(start, start, level+1, B_split[0], side, uplo);
    } else {
      trsm_solve(start, start, level+1, B_split[0], side, uplo);
      Hatrix::matmul(low_rank(start+1, start, level+1), B_split[0], B_split[1], false, false, -1, 1);
      trsm_solve(start+1, start+1, level+1, B_split[1], side, uplo);
    }
  } else {
    bool diag = uplo == Hatrix::Lower;
    Hatrix::solve_triangular(dense(row, col, level), B, side, uplo, diag, false);
  }
}

template <typename DT>
void Hmatrix<DT>::materialize_low_rank(Matrix<DT>& A, int row, int col, int level) const {
  #pragma omp task shared(low_rank)
  {
    Hatrix::Matrix<DT> B = low_rank(row, col, level).make_dense();
    for (int i=0; i<A.rows; ++i)
      for (int j=0; j<A.cols; ++j)
        A(i,j) = B(i,j);
  }
}

template <typename DT>
void Hmatrix<DT>::materialize(Matrix<DT>& A, int row, int col, int level) const {
  int start = row * 2;
  if (level < max_level) {
    std::vector<Matrix<DT>> A_split = A.split(2,2);
    materialize_low_rank(A_split[1], start, start+1, level+1);
    materialize_low_rank(A_split[2], start+1, start, level+1);
    materialize(A_split[0], start, start, level+1);
    materialize(A_split[3], start+1, start+1, level+1);
  } else {
    #pragma omp task shared(dense)
    {
      for (int i=0; i<A.rows; ++i)
        for (int j=0; j<A.cols; ++j)
          A(i,j) = dense(row, col, level)(i,j);
    }
  }

}

template <typename DT>
Matrix<DT> Hmatrix<DT>::make_dense() const {
  int rows = dense.size() * leaf_size;
  Matrix<DT> A(rows, rows);
  #pragma omp parallel
  {
    #pragma omp single
    {
      materialize(A, 0, 0, 0);
    }
  }
  return A;
}

// explicit instantiation (these are the only available data-types)
template class Hmatrix<float>;
template class Hmatrix<double>;
template Hmatrix<float>::Hmatrix(const Hmatrix<double>&);
template Hmatrix<double>::Hmatrix(const Hmatrix<float>&);

}  // namespace Hatrix
