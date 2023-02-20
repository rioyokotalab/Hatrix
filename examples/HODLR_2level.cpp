#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <omp.h>
#include <sys/time.h>

#include "Hatrix/Hatrix.h"

double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return (double)(tv.tv_sec+tv.tv_usec*1e-6);
}

namespace Hatrix {

  void is_admissible(RowColLevelMap<bool>& map, int row, int col, int level) {
    map.insert(row, col, level, row!=col);
  }

  void add_admissibility(RowColLevelMap<bool>& map, int row, int col, int level, int max_level) {
    is_admissible(map, row, col, level);
    is_admissible(map, row+1, col, level);
    is_admissible(map, row, col+1, level);
    is_admissible(map, row+1, col+1, level);
    std::cout<<row<<","<<col<<","<<level<<" is "<<map(row,col,level)<<std::endl;
    std::cout<<row<<","<<col+1<<","<<level<<" is "<<map(row,col+1,level)<<std::endl;
    std::cout<<row+1<<","<<col<<","<<level<<" is "<<map(row+1,col,level)<<std::endl;
    std::cout<<row+1<<","<<col+1<<","<<level<<" is "<<map(row+1,col+1,level)<<std::endl;


    if (level < max_level) {
      add_admissibility(map, row, col, level+1, max_level);
      add_admissibility(map, row+2*level, col+2*level, level+1, max_level);
    }
  }

  RowColLevelMap<bool> create_admissibility(int max_level) {
    RowColLevelMap<bool> map;
    
    map.insert(0, 0, 0, false);
    add_admissibility(map, 0, 0, 1, max_level);
    return map;
  }

  template <typename DT>
  void add_low_rank(const Matrix<DT>& A, RowColLevelMap<LowRank<DT>>& map, int row, int col, int level, int leaf_size, int rank, omp_lock_t& lock) {
    std::vector<Hatrix::Matrix<DT>> A_split = A.split(2, 2);
    #pragma omp task shared(map)
    {
      LowRank<DT> LR(A_split[1], rank);
      omp_set_lock(&lock);
      map.insert(row, col+1, level, std::move(LR));
      omp_unset_lock(&lock);
    }
    #pragma omp task shared(map)
    {
      LowRank<DT> LR(A_split[2], rank);
      omp_set_lock(&lock);
      map.insert(row+1, col, level, std::move(LR));
      omp_unset_lock(&lock);
    }
    if (A_split[0].rows > leaf_size) {
      add_low_rank(A_split[0], map, row, col, level+1, leaf_size, rank, lock);
      add_low_rank(A_split[3], map, row+2*level, col+2*level, level+1, leaf_size, rank, lock);
    }
  }

  template <typename DT>
  RowColLevelMap<LowRank<DT>> create_lr_map(const Matrix<DT>& A, RowColLevelMap<bool>& is_admissible, int leaf_size, int rank) {
    RowColLevelMap<LowRank<DT>> map;

    //TODO is this a good solution?
    omp_lock_t writelock;
    omp_init_lock(&writelock);
    
    #pragma omp parallel
    {
      #pragma omp single 
      {
        add_low_rank(A, map, 0, 0, 1, leaf_size, rank, writelock);
      }
    }
    return map;
  }

  template <typename DT>
  RowColLevelMap<Matrix<DT>> create_dense_map(const Matrix<DT>& A, int leaf_size, int max_level) {
    RowColLevelMap<Matrix<DT>> map;

    //TODO is this a good solution?
    omp_lock_t writelock;
    omp_init_lock(&writelock);

    int num_blocks = A.rows / leaf_size;

    #pragma omp parallel
    {
      #pragma omp single 
      {
        for (int i=0; i<num_blocks; ++i) {
          #pragma omp task
          {
            Matrix<DT> block_copy(leaf_size, leaf_size);
            for (int k=0; k<leaf_size; ++k) {
              for (int l=0; l<leaf_size; ++l) {
                block_copy(k,l) = A(i*leaf_size+k, i*leaf_size+l);
              }
            }
            omp_set_lock(&writelock);
            map.insert(i, i, max_level, std::move(block_copy));
            omp_unset_lock(&writelock);
          }
        }
      }
    }
    return map;
  }

  template <typename DT>
  void getrf(Hatrix::RowColLevelMap<Hatrix::Matrix<DT>>& dense_map, Hatrix::RowColLevelMap<Hatrix::LowRank<DT>>& lr_map, int row, int col, int level, int max_level) {
    int start = row * 2;
    if (level < max_level) {
      //std::cout<<"GETRF on ("<<start<<","<<start<<","<<level<<")"<<std::endl;
      getrf<DT>(dense_map, lr_map, start, start, level+1, max_level);
      //std::cout<<"TRSM on ("<<start<<","<<start+1<<","<<level+1<<")"<<std::endl;
      Hatrix::solve_triangular(dense_map(start, start, level+1), lr_map(start, start+1, level+1).U, Hatrix::Left, Hatrix::Lower, true, false);
      //std::cout<<"TRSM on ("<<start+1<<","<<start<<","<<level+1<<")"<<std::endl;
      Hatrix::solve_triangular(dense_map(start, start, level+1), lr_map(start+1, start, level+1).V, Hatrix::Right, Hatrix::Upper, false, false);
      //std::cout<<"GEMM on ("<<start+1<<","<<start+1<<","<<level+1<<")"<<std::endl;
      Hatrix::matmul(lr_map(start+1, start, level+1), lr_map(start, start+1, level+1), dense_map(start+1, start+1, level+1), false, false, -1, 1);
      //std::cout<<"GETRF on ("<<start+1<<","<<start+1<<","<<level<<")"<<std::endl;
      getrf<DT>(dense_map, lr_map, start+1, start+1, level+1, max_level);
    } else {
      //std::cout<<"GETRF on ("<<row<<","<<col<<","<<level<<")"<<std::endl;
      Hatrix::lu_nopiv(dense_map(row,col,level));
    }
  }

  template <typename DT>
  void trsm(Hatrix::RowColLevelMap<Hatrix::Matrix<DT>>& dense_map, Hatrix::RowColLevelMap<Hatrix::LowRank<DT>>& lr_map, int row, int col, int level, int max_level, Hatrix::Matrix<DT>& B, Hatrix::Side side, Hatrix::Mode uplo) {
    if (level < max_level) {
      std::vector<Hatrix::Matrix<DT>> B_split = B.split(2, 1);
      if (uplo == Hatrix::Upper) {
        trsm<DT>(dense_map, lr_map, row+1, col+1, level+1, max_level, B_split[1], side, uplo);
        matmul(lr_map(row, col+1, level+1), B_split[1], B_split[0], false, false, -1, 1);
        trsm<DT>(dense_map, lr_map, row, col, level+1, max_level, B_split[0], side, uplo);
      } else {
        trsm<DT>(dense_map, lr_map, row, col, level+1, max_level, B_split[0], side, uplo);
        matmul(lr_map(row+1, col, level+1), B_split[0], B_split[1], false, false, -1, 1);
        trsm<DT>(dense_map, lr_map, row+1, col+1, level+1, max_level, B_split[1], side, uplo);
      }
    } else {
      bool diag = uplo == Hatrix::Lower;
      Hatrix::solve_triangular(dense_map(row, col, level), B, side, uplo, diag, false);
    }
  }
}

int main() {
  int n = 16;
  int leaf_size = 8;
  int num_blocks = n / leaf_size;
  int rank = 3;

  int max_level = 0;
  int size = n;
  while (leaf_size < size){
    size = size>>1;
    max_level++;
  }

  std::vector<std::vector<double>> randpts;
  randpts.push_back(Hatrix::equally_spaced_vector(2*n, 0.0, 1.0));

  Hatrix::Matrix D = Hatrix::generate_laplacend_matrix(randpts, n, n, 0, 0);
  Hatrix::Matrix D_copy(D, true);
  Hatrix::RowColLevelMap<bool> is_admissible = Hatrix::create_admissibility(max_level);
  //D.print();
  double tic = get_time();
  Hatrix::RowColLevelMap<Hatrix::Matrix<double>> dense_map = Hatrix::create_dense_map(D, leaf_size, max_level);
  Hatrix::RowColLevelMap<Hatrix::LowRank<double>> lr_map = Hatrix::create_lr_map(D, is_admissible, leaf_size, rank);
  double toc = get_time();
  std::cout<<"Time (Construction): "<<toc-tic<<std::endl;
  dense_map(0,0,1).print();
  dense_map(1,1,1).print();
  lr_map(0,1,1).print();
  lr_map(1,0,1).print();
  
  /*for (int i=0; i<num_blocks; i++) {
    for (int j=0; j<num_blocks; j++) {
      map(i,j,0).print();
    }
  }*/
  
  omp_set_num_threads(4);
  tic = get_time();
  #pragma omp parallel
  {
    #pragma omp single 
    {
      Hatrix::getrf<double>(dense_map, lr_map, 0, 0, 0, max_level);
    }
  }
  
  toc = get_time();
  std::cout<<"Time: "<<toc-tic<<std::endl;
  
  Hatrix::Matrix b = Hatrix::generate_random_matrix(n,1);
  Hatrix::Matrix b_copy(b, true);

  #pragma omp parallel
  {
    #pragma omp single 
    {
      Hatrix::trsm<double>(dense_map, lr_map, 0, 0, 0, max_level, b, Hatrix::Left, Hatrix::Upper);
      Hatrix::trsm<double>(dense_map, lr_map, 0, 0, 0, max_level, b, Hatrix::Left, Hatrix::Lower);
    }
  }
  b.print();
  //Hatrix::solve_triangular(map(1,1,0), b_split[1], Hatrix::Left, Hatrix::Lower, true, false);

  lu_nopiv(D_copy);
  //D_copy.print();
  Hatrix::solve_triangular(D_copy, b_copy, Hatrix::Left, Hatrix::Upper, false, false);
  //b_copy.print();
  Hatrix::solve_triangular(D_copy, b_copy, Hatrix::Left, Hatrix::Lower, true, false);

  double error = Hatrix::norm(b_copy - b);
  std::cout<<"Error: "<<error<<std::endl;

  //b.print();
  //std::cout<<std::endl;
  //b_copy.print();
  return 0;
}