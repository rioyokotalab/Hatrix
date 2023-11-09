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

  void add_children(RowColLevelMap<bool>& map, int row, int col, int level, int max_level) {
    if (level > max_level)
      return;

    row *= 2;
    col *= 2;
    if (col > row) {
      map.insert(row, col, level, true);
      //std::cout<<"Insert "<<row<<","<<col<<","<<level<<"(child)"<<std::endl;
      add_children(map, row, col, level+1, max_level);
      map.insert(row+1, col, level, true);
      //std::cout<<"Insert "<<row+1<<","<<col<<","<<level<<"(child)"<<std::endl;
      add_children(map, row+1, col, level+1, max_level);
    } else {
      //no diagonal blocks are addmissible
      map.insert(row, col, level, true);
      //std::cout<<"Insert "<<row<<","<<col<<","<<level<<"(child)"<<std::endl;
      add_children(map, row, col, level+1, max_level);
      map.insert(row, col+1, level, true);
      //std::cout<<"Insert "<<row<<","<<col+1<<","<<level<<"(child)"<<std::endl;
      add_children(map, row, col+1, level+1, max_level);
    }
  }

  bool is_admissible(RowColLevelMap<bool>& map, int row, int col, int level) {
    map.insert(row, col, level, row!=col);
    //std::cout<<"Insert "<<row<<","<<col<<","<<level<<"("<<(row!=col)<<")"<<std::endl;
    return row!=col;
  }

  void add_admissibility(RowColLevelMap<bool>& map, int row, int col, int level, int max_level) {
    if (level > max_level)
      return;
    // add yourself
    if (!is_admissible(map, row, col, level)) {
      // call recursively for children
      int start = row * 2;
      add_admissibility(map, start, start, level+1, max_level);
      add_admissibility(map, start, start+1, level+1, max_level);
      add_admissibility(map, start+1, start, level+1, max_level);
      add_admissibility(map, start+1, start+1, level+1, max_level);
    } else {
      add_children(map, row, col, level+1, max_level);
    }
  }

  RowColLevelMap<bool> create_admissibility(int max_level) {
    RowColLevelMap<bool> map;
    
    //map.insert(0, 0, 0, false);
    add_admissibility(map, 0, 0, 0, max_level);
    return map;
  }

  template<typename DT>
  void spawn_children(RowColLevelMap<LowRank<DT>>& map, int row, int col, int level, int max_level) {
    if (level >= max_level)
      return;

    int child_row = row * 2;
    int child_col = col * 2;
    // S and non-split U/V are currently not used, but might be useful in the future for GEMM split
    if (col > row) {
      std::vector<Matrix<DT>> children = map(row, col, level).U.split(2,1);
      map.insert(child_row, child_col, level+1, LowRank<DT>(children[0], map(row, col, level).S, map(row, col, level).V, false));
      //std::cout<<"Add child "<<child_row<<","<<child_col<<","<<level+1<<std::endl;
      spawn_children(map, child_row, child_col, level+1, max_level);
      map.insert(child_row+1, child_col, level+1, LowRank<DT>(children[1], map(row, col, level).S, map(row, col, level).V, false));
      //std::cout<<"Add child "<<child_row+1<<","<<child_col<<","<<level+1<<std::endl;
      spawn_children(map, child_row+1, child_col, level+1, max_level);
    } else {
      std::vector<Matrix<DT>> children = map(row, col, level).V.split(1,2);
      map.insert(child_row, child_col, level+1, LowRank<DT>(map(row, col, level).U, map(row, col, level).S, children[0], false));
      //std::cout<<"Add child "<<child_row<<","<<child_col<<","<<level+1<<std::endl;
      spawn_children(map, child_row, child_col, level+1, max_level);
      map.insert(child_row, child_col+1, level+1, LowRank<DT>(map(row, col, level).U, map(row, col, level).S, children[1], false));
      //std::cout<<"Add child "<<child_row<<","<<child_col+1<<","<<level+1<<std::endl;
      spawn_children(map, child_row, child_col+1, level+1, max_level);
    }
  }
  
  template <typename DT>
  void add_low_rank(const Matrix<DT>& A, RowColLevelMap<LowRank<DT>>& map, RowColLevelMap<bool>& is_admissible, int row, int col, int level, int leaf_size, int rank, omp_lock_t& lock, int max_level) {
    if (A.rows < leaf_size)
      return;

    if (is_admissible(row, col, level)) {
      #pragma omp task shared(map)
      {
        LowRank<DT> LR(A, rank);
        std::cout<<"Row: "<<row<<", Col: "<<col<<" Lvl: "<<level<<std::endl;
        LR.print_approx();
        std::cout<<"rank for e-8: "<<LR.get_rank(1e-8)<<std::endl;
        std::cout<<std::endl;
        omp_set_lock(&lock);
        map.insert(row, col, level, std::move(LR));
        spawn_children(map, row, col, level, max_level);
        //std::cout<<"LowRank "<<row<<","<<col<<","<<level<<std::endl;
        omp_unset_lock(&lock);
      }
    } else {
      int start = row * 2;
      std::vector<Hatrix::Matrix<DT>> A_split = A.split(2, 2);
      add_low_rank<DT>(A_split[0], map, is_admissible, start, start, level+1, leaf_size, rank, lock, max_level);
      add_low_rank<DT>(A_split[1], map, is_admissible, start, start+1, level+1, leaf_size, rank, lock, max_level);
      add_low_rank<DT>(A_split[2], map, is_admissible, start+1, start, level+1, leaf_size, rank, lock, max_level);
      add_low_rank<DT>(A_split[3], map, is_admissible, start+1, start+1, level+1, leaf_size, rank, lock, max_level);
    }
  }

  template <typename DT>
  RowColLevelMap<LowRank<DT>> create_lr_map(const Matrix<DT>& A, RowColLevelMap<bool>& is_admissible, int leaf_size, int rank, int max_level) {
    RowColLevelMap<LowRank<DT>> map;

    //TODO is this a good solution?
    omp_lock_t writelock;
    omp_init_lock(&writelock);
    
    #pragma omp parallel
    {
      #pragma omp single 
      {
        add_low_rank(A, map, is_admissible, 0, 0, 0, leaf_size, rank, writelock, max_level);
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

  void empty_task() {
    return;
  }

  void update_children(RowColLevelMap<bool>& is_admissible, int row, int col, int level, int max_level) {
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
      update_children(is_admissible, child_row, child_col, level+1, max_level);
      update_children(is_admissible, child_row, child_col+1, level+1, max_level);
    } else {
      #pragma omp task depend(in: is_admissible(row, col, level)) depend(out: is_admissible(child_row, child_col, level+1), is_admissible(child_row+1, child_col, level+1))
      {
        empty_task();
      }
      update_children(is_admissible, child_row, child_col, level+1, max_level);
      update_children(is_admissible, child_row+1, child_col, level+1, max_level);
    }
  }

  void update_parent(RowColLevelMap<bool>& is_admissible, int row, int col, int level, int max_level) {
    // has no children
    if (level >= max_level)
      return;

    // creates a dependency with the children as input and the parent as output
    // i.e. the children update their parent
    int child_row = row * 2;
    int child_col = col * 2;
    if (row > col) {
      // Must go from bottom to top
      update_parent(is_admissible, child_row, child_col, level+1, max_level);
      update_parent(is_admissible, child_row, child_col+1, level+1, max_level);
      //std::cout<<"Update parent of ("<<child_row<<","<<child_col<<","<<level+1<<") ("<<row<<","<<col<<","<<level<<std::endl;
      #pragma omp task depend(in: is_admissible(child_row, child_col, level+1), is_admissible(child_row, child_col+1, level+1)) depend(out: is_admissible(row, col, level))
      {
        empty_task();
      }
      
    } else {
      // Must go from bottom to top
      update_parent(is_admissible, child_row, child_col, level+1, max_level);
      update_parent(is_admissible, child_row+1, child_col, level+1, max_level);
      //std::cout<<"Update parent of ("<<child_row<<","<<child_col<<","<<level+1<<") ("<<row<<","<<col<<","<<level<<std::endl;
      #pragma omp task depend(in: is_admissible(child_row, child_col, level+1), is_admissible(child_row+1, child_col, level+1)) depend(out: is_admissible(row, col, level))
      {
        empty_task();
      }
      
    }
  }

  void wait_for_children(RowColLevelMap<bool>& is_admissible, int row, int col, int level, int max_level) {
    if (level > max_level)
      return;
    row *= 2;
    col *= 2;
    if (level < max_level) {
      wait_for_children(is_admissible, row, col, level+1, max_level);
      if (row > col) {
        wait_for_children(is_admissible, row, col+1, level+1, max_level);
      } else {
        wait_for_children(is_admissible, row+1, col, level+1, max_level);
      }
    } else {
      if (row > col) {
        //std::cout<<"Wait for "<<row<<","<<col<<","<<level<<std::endl;
        //std::cout<<"Wait for "<<row<<","<<col+1<<","<<level<<std::endl;
        #pragma omp taskwait depend(in: is_admissible(row, col, level), is_admissible(row, col+1, level))
      } else {
        //std::cout<<"Wait for "<<row<<","<<col<<","<<level<<std::endl;
        //std::cout<<"Wait for "<<row+1<<","<<col<<","<<level<<std::endl;
        #pragma omp taskwait depend(in: is_admissible(row, col, level), is_admissible(row+1, col, level))
      }
    }

  }

  template <typename DT>
  void trsm(Hatrix::RowColLevelMap<Hatrix::Matrix<DT>>& dense_map, Hatrix::RowColLevelMap<Hatrix::LowRank<DT>>& lr_map, Hatrix::RowColLevelMap<bool>& is_admissible, int row, int col, int level, int max_level, Hatrix::Side side, Hatrix::Mode uplo) {
    if (level < max_level) {
      // TODO can os should this matmuls be written recursively?
      if (uplo == Hatrix::Upper) {
        int start = col * 2;
        //std::cout<<"TRSM1"<<std::endl;
        trsm<DT>(dense_map, lr_map, is_admissible, row*2, start, level+1, max_level, side, uplo);
        //#pragma omp taskwait
        //std::cout<<"GEMM1"<<std::endl;
        //wait_for_children(is_admissible, start, start+1, level+2, max_level);
        update_parent(is_admissible, row*2, start, level+1, max_level);
        update_parent(is_admissible, start, start+1, level+1, max_level);
        #pragma omp task shared(lr_map) depend(in: is_admissible(row*2, start, level+1), is_admissible(start, start+1, level+1)) depend(inout: is_admissible(row*2, start+1, level+1))
        {  //std::cout<<"GEMM (t-task)"<<row*2<<","<<start+1<<","<<level+1<<std::endl;
          matmul(lr_map(row*2, start, level+1).V, lr_map(start, start+1, level+1), lr_map(row*2, start+1, level+1).V, false, false, -1, 1);
        }
        //#pragma omp taskwait
        update_children(is_admissible, row*2, start+1, level+1, max_level);
        //#pragma omp taskwait
        //#pragma omp task shared(dense_map, lr_map, is_admissible) depend(in: is_admissible(row*2, start+1, level+1))
        trsm<DT>(dense_map, lr_map, is_admissible, row*2, start+1, level+1, max_level, side, uplo);
      } else {
        int start = row * 2;
        //std::cout<<"TRSM1"<<std::endl;
        trsm<DT>(dense_map, lr_map, is_admissible, start, col*2, level+1, max_level, side, uplo);
        //#pragma omp taskwait
        
        //std::cout<<"GEMM"<<std::endl;
        //wait_for_children(is_admissible, start+1, start, level+2, max_level);
        update_parent(is_admissible, start+1, start, level+1, max_level);
        update_parent(is_admissible, start, col*2, level+1, max_level);
        #pragma omp task shared(lr_map) depend(in: is_admissible(start, col*2, level+1), is_admissible(start+1, start, level+1)) depend(inout: is_admissible(start+1, col*2, level+1))
        { //std::cout<<"GEMM (t-task)"<<start+1<<","<<col*2<<","<<level+1<<std::endl;
          matmul(lr_map(start+1, start, level+1), lr_map(start, col*2, level+1).U, lr_map(start+1, col*2, level+1).U, false, false, -1, 1);
        }
        //#pragma omp taskwait
        update_children(is_admissible, start+1, col*2, level+1, max_level);
        //#pragma omp taskwait
        //std::cout<<"TRSM2"<<std::endl;
        //#pragma omp task shared(dense_map, lr_map, is_admissible) depend(in: is_admissible(start+1, col*2, level+1))
        trsm<DT>(dense_map, lr_map, is_admissible, start+1, col*2, level+1, max_level, side, uplo);
      }
    } else {
      bool diag = uplo == Hatrix::Lower;
      int src_idx = diag?row:col;
      //std::cout<<"TRSM "<<row<<","<<col<<","<<level<<std::endl;
      #pragma omp task shared(dense_map, lr_map) depend(in: is_admissible(src_idx, src_idx, level)) depend(inout: is_admissible(row, col, level))
      { //std::cout<<"TRSM (task) "<<row<<","<<col<<","<<level<<std::endl;
        Hatrix::solve_triangular(dense_map(src_idx, src_idx, level), diag?lr_map(row, col, level).U:lr_map(row, col, level).V, side, uplo, diag, false);
      }
      //#pragma omp taskwait
      //std::cout<<"Finished"<<std::endl;
    }
  }

  template <typename DT>
  void add(Hatrix::RowColLevelMap<Hatrix::Matrix<DT>>& dense_map, Hatrix::RowColLevelMap<Hatrix::LowRank<DT>>& lr_map, Hatrix::RowColLevelMap<bool>& is_admissible, int row, int col, int level, int max_level, LowRank<DT>& temp, int block_start, int block_size) {
    /*if (is_admissible(row, col, level)) {
      //#pragma omp task shared(lr_map, T) depend(inout: is_admissible(row, col, level)) depend(in: T)
      {
        lr_map(row, col, level) += T;
      }
      update_children(is_admissible, row, col, level, max_level);
    } else {
    */
    if (level < max_level) {
      //std::cout<<"Add on ("<<row<<","<<col<<","<<level<<")"<<std::endl;
      int start = row * 2;
      //std::cout<<"Try to split"<<std::endl;
      //std::vector<Matrix<DT>> U_split = T.U.split(2, 1);
      //std::vector<Matrix<DT>> V_split = T.V.split(1, 2); 
      block_size = block_size/2;
      /*yMatrix<DT> U0 = test.U.get_row_block(block_start, block_size);
      Matrix<DT> U1 = test.U.get_row_block(block_start+block_size, block_size);
      Matrix<DT> V0 = test.V.get_col_block(block_start, block_size);
      Matrix<DT> V1 = test.V.get_col_block(block_start+block_size, block_size);
      LowRank<DT> T0(U_split[0], T.S, V_split[0], false);
      LowRank<DT> T1(U_split[0], T.S, V_split[1], false);
      LowRank<DT> T2(U_split[1], T.S, V_split[0], false);
      LowRank<DT> T3(U_split[1], T.S, V_split[1], false);*/
      //#pragma omp task shared(dense_map, lr_map, is_admissible, T0) depend(out: T0)
      add(dense_map, lr_map, is_admissible, start, start, level+1, max_level, temp, block_start, block_size);
      //#pragma omp task shared(dense_map, lr_map, is_admissible, T3) depend(out: T3)
      add(dense_map, lr_map, is_admissible, start+1, start+1, level+1, max_level, temp, block_start+block_size, block_size);
      // The order is important here since one of the following operations seems to manimulate T
      // Is this a problem if I increase the level?
      // Changed the low rank addition instead to make a copy when needed and now the issue seems to be resolved
      //#pragma omp task shared(dense_map, lr_map, is_admissible, T1) depend(out: T1)
        
      // off-diagonal blocks
      #pragma omp task shared(lr_map, temp) depend(inout: is_admissible(start, start+1, level+1)) depend(in: temp)
      {//std::cout<<"Add on ("<<start<<","<<start+1<<","<<level+1<<")"<<std::endl;
        // upper
        //std::cout<<block_start<<" "<<block_size<<std::endl;
        Matrix<DT> U = temp.U.get_row_block(block_start, block_size);
        Matrix<DT> V = temp.V.get_col_block(block_start+block_size, block_size);
        LowRank<DT> T(U, temp.S, V, false);
        /*std::cout<<U(0,0)<<","<<U(1,0)<<U(0,1)<<std::endl;
        std::cout<<T.U(0,0)<<","<<T.U(1,0)<<T.U(0,1)<<std::endl;
        std::cout<<V(0,0)<<","<<V(1,0)<<V(0,1)<<std::endl;
        std::cout<<T.V(0,0)<<","<<T.V(1,0)<<T.V(0,1)<<std::endl;*/
        lr_map(start, start+1, level+1) += T;
      }
      update_children(is_admissible, start, start+1, level+1, max_level);
      #pragma omp task shared(lr_map, temp) depend(inout: is_admissible(start+1, start, level+1)) depend(in: temp)
      {//std::cout<<"Add on ("<<start+1<<","<<start<<","<<level+1<<")"<<std::endl;
        // lower
        Matrix<DT> U = temp.U.get_row_block(block_start+block_size, block_size);
        Matrix<DT> V = temp.V.get_col_block(block_start, block_size);
        LowRank<DT> T(U, temp.S, V, false);
        lr_map(start+1, start, level+1) += T;
      }
      update_children(is_admissible, start+1, start, level+1, max_level);

        
      //add(dense_map, lr_map, is_admissible, start, start+1, level+1, max_level, T1, block_start, block_size);
      //#pragma omp task shared(dense_map, lr_map, is_admissible, T2) depend(out: T2)
      //add(dense_map, lr_map, is_admissible, start+1, start, level+1, max_level, T2, block_start, block_size);
    } else { //this must be a dense block
      //std::cout<<"Add on dense ("<<row<<","<<col<<","<<level<<")"<<std::endl;
      #pragma omp task shared(dense_map, temp) depend(inout: is_admissible(row, col, level)) depend(in: temp)
      {//std::cout<<"Add on ("<<row<<","<<col<<","<<level<<")"<<std::endl;
        Matrix<DT> U = temp.U.get_row_block(block_start, block_size);
        Matrix<DT> V = temp.V.get_col_block(block_start, block_size);
        //std::cout<<U(0,0)<<","<<U(1,0)<<U(0,1)<<std::endl;
        //std::cout<<T.U(0,0)<<","<<T.U(1,0)<<T.U(0,1)<<std::endl;
        //std::cout<<V(0,0)<<","<<V(1,0)<<V(0,1)<<std::endl;
        //std::cout<<T.V(0,0)<<","<<T.V(1,0)<<T.V(0,1)<<std::endl;
        LowRank<DT> T(U, temp.S, V, false);
        dense_map(row, col, level) += T;
      }
    }
  }

  template <typename DT>
  void matmul(Hatrix::RowColLevelMap<Hatrix::Matrix<DT>>& dense_map, Hatrix::RowColLevelMap<Hatrix::LowRank<DT>>& lr_map, Hatrix::RowColLevelMap<bool>& is_admissible, int row, int col, int level, int max_level, Hatrix::LowRank<DT>& temp) {
    if (level < max_level) {
      // wait for all tasks on children of the input low-rank blocks
      //std::cout<<"GEMM on ("<<row<<","<<col<<","<<level<<")"<<std::endl;
      //wait_for_children(is_admissible, row, col-1, level+1, max_level);
      //wait_for_children(is_admissible, row-1, col, level+1, max_level);
      //update_parent(is_admissible, row, col-1, level, max_level);
      //update_parent(is_admissible, row-1, col, level, max_level);
      update_parent(is_admissible, row, col-1, level, max_level);
      update_parent(is_admissible, row-1, col, level, max_level);
      #pragma omp task shared(dense_map, lr_map, temp) depend(in: is_admissible(row, col-1, level), is_admissible(row-1, col, level)) depend(out: temp)
      {  
        matmul(lr_map(row, col-1, level), lr_map(row-1, col, level), temp, false, false, -1, 1, false);
      }
      //#pragma omp taskwait
        //std::cout<<"Add on ("<<row<<","<<col<<","<<level<<")"<<std::endl
      // can't infer the row number from temp, since processing might not be finished yet
      add(dense_map, lr_map, is_admissible, row, col, level, max_level, temp, 0, lr_map(row, col-1, level).rows);

    } else {
      #pragma omp task shared(dense_map, lr_map) depend(in: is_admissible(row, col-1, level), is_admissible(row-1, col, level)) depend(inout: is_admissible(row, col, level))
      {//std::cout<<"GEMM (matmul task) on ("<<row<<","<<col<<","<<level<<")"<<std::endl;
        matmul(lr_map(row, col-1, level), lr_map(row-1, col, level), dense_map(row, col, level), false, false, -1, 1);
      }
     }
  }

  template <typename DT>
  void getrf(Hatrix::RowColLevelMap<Hatrix::Matrix<DT>>& dense_map, Hatrix::RowColLevelMap<Hatrix::LowRank<DT>>& lr_map, Hatrix::RowColLevelMap<bool>& is_admissible, int row, int col, int level, int max_level, Hatrix::LowRank<DT>& temp) {
    int start = row * 2;
    if (level < max_level) {
      //std::cout<<"GETRF on ("<<start<<","<<start<<","<<level<<")"<<std::endl;
      //#pragma omp task shared(dense_map, lr_map, is_admissible) depend(inout: is_admissible(start, start, level+1))
      {
        getrf<DT>(dense_map, lr_map, is_admissible, start, start, level+1, max_level, temp);
      }
      //#pragma omp taskwait
      //#pragma omp task shared(dense_map, lr_map, is_admissible) depend(in: is_admissible(start, start, level+1)) depend(inout: is_admissible(start, start+1, level+1))
      {
        //std::cout<<"TRSM on ("<<start<<","<<start+1<<","<<level+1<<")"<<std::endl;
        Hatrix::trsm<DT>(dense_map, lr_map, is_admissible, start, start+1, level+1, max_level, Hatrix::Left, Hatrix::Lower);
      }
      //#pragma omp taskwait
      //#pragma omp task shared(dense_map, lr_map, is_admissible) depend(in: is_admissible(start, start, level+1)) depend(inout: is_admissible(start+1, start, level+1))
      {//std::cout<<"TRSM on ("<<start+1<<","<<start<<","<<level+1<<")"<<std::endl;
        Hatrix::trsm<DT>(dense_map, lr_map, is_admissible, start+1, start, level+1, max_level, Hatrix::Right, Hatrix::Upper);
      }
      //#pragma omp taskwait
      //#pragma omp task shared(dense_map, lr_map, is_admissible) depend(in: is_admissible(start+1, start, level+1), is_admissible(start, start+1, level+1)) depend(inout: is_admissible(start+1, start+1, level+1))
      {//std::cout<<"GEMM on ("<<start+1<<","<<start+1<<","<<level+1<<")"<<std::endl;
        Hatrix::matmul(dense_map, lr_map, is_admissible, start+1, start+1, level+1, max_level, temp);
      }
      //#pragma omp taskwait
      //#pragma omp task shared(dense_map, lr_map, is_admissible) depend(inout: is_admissible(start+1, start+1, level+1))
      {//std::cout<<"GETRF(dense) on ("<<row<<","<<col<<","<<level<<")"<<std::endl;
        getrf<DT>(dense_map, lr_map, is_admissible, start+1, start+1, level+1, max_level, temp);
      }
      //#pragma omp taskwait
    } else {
      #pragma omp task shared(dense_map) depend(inout: is_admissible(row, col, level))
        {//std::cout<<"GETRF(task) on ("<<row<<","<<col<<","<<level<<")"<<std::endl;
          Hatrix::lu_nopiv(dense_map(row,col,level));
        }
    }
  }

  template <typename DT>
  void trsm_solve(Hatrix::RowColLevelMap<Hatrix::Matrix<DT>>& dense_map, Hatrix::RowColLevelMap<Hatrix::LowRank<DT>>& lr_map, int row, int col, int level, int max_level, Hatrix::Matrix<DT>& B, Hatrix::Side side, Hatrix::Mode uplo) {
    int start = row * 2;
    if (level < max_level) {
      std::vector<Hatrix::Matrix<DT>> B_split;
      if (side == Hatrix::Left) {
        B_split = B.split(2, 1);
      } else {
        assert(false);
      }
      if (uplo == Hatrix::Upper) {
        trsm_solve<DT>(dense_map, lr_map, start+1, start+1, level+1, max_level, B_split[1], side, uplo);
        matmul(lr_map(start, start+1, level+1), B_split[1], B_split[0], false, false, -1, 1);
        trsm_solve<DT>(dense_map, lr_map, start, start, level+1, max_level, B_split[0], side, uplo);
      } else {
        trsm_solve<DT>(dense_map, lr_map, start, start, level+1, max_level, B_split[0], side, uplo);
        matmul(lr_map(start+1, start, level+1), B_split[0], B_split[1], false, false, -1, 1);
        trsm_solve<DT>(dense_map, lr_map, start+1, start+1, level+1, max_level, B_split[1], side, uplo);
      }
    } else {
      bool diag = uplo == Hatrix::Lower;
      Hatrix::solve_triangular(dense_map(row, col, level), B, side, uplo, diag, false);
    }
  }

  template <typename DT>
  void matmul(Hatrix::RowColLevelMap<Hatrix::Matrix<DT>>& dense_map, Hatrix::RowColLevelMap<Hatrix::LowRank<DT>>& lr_map, int row, int col, int level, int max_level, Hatrix::Matrix<DT>& B, Hatrix::Matrix<DT>& C, double alpha=1, double beta=0) {
    //std::cout<<"Called matmul for row="<<row<<" col="<<col<<" level="<<level<<std::endl;
    if (level < max_level) {
      int start = row * 2;
      std::vector<Hatrix::Matrix<DT>> B_split = B.split(2, 1);
      std::vector<Hatrix::Matrix<DT>> C_split = C.split(2, 1);
      
      
      #pragma omp task shared(dense_map, lr_map, B, C)
      {
        matmul(dense_map, lr_map, start, start, level+1, max_level, B_split[0], C_split[0]);
        //std::cout<<"Calc matmul for row="<<start<<" col="<<start+1<<" level="<<level+1<<std::endl;
        #pragma omp taskwait
        matmul(lr_map(start, start+1, level+1), B_split[1], C_split[0], false, false, 1, 1);
      }
      #pragma omp task shared(dense_map, lr_map, B, C)
      {
        matmul(dense_map, lr_map, start+1, start+1, level+1, max_level, B_split[1], C_split[1]);
        //std::cout<<"Calc matmul for row="<<start+1<<" col="<<start<<" level="<<level+1<<std::endl;
        #pragma omp taskwait
        matmul(lr_map(start+1, start, level+1), B_split[0], C_split[1], false, false, 1, 1);
      }
    } else {
      matmul(dense_map(row, col, level), B, C, false, false, alpha, beta);
    }
  }
}

int main() {
  int n = 512;
  int leaf_size = 256;
  int num_blocks = n / leaf_size;
  int rank = 16;

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
  omp_set_num_threads(1);
  double tic = get_time();
  Hatrix::RowColLevelMap<Hatrix::Matrix<double>> dense_map = Hatrix::create_dense_map(D, leaf_size, max_level);
  Hatrix::RowColLevelMap<Hatrix::LowRank<double>> lr_map = Hatrix::create_lr_map(D, is_admissible, leaf_size, rank, max_level);
  double toc = get_time();
  std::cout<<"Time (Construction): "<<toc-tic<<std::endl;
  /*
  Hatrix::Matrix b = Hatrix::generate_random_matrix(n,1);
  Hatrix::Matrix c(n, 1);
  Hatrix:matmul(D_copy, b, c, false, false, 1, 0);
  Hatrix::Matrix c1(n, 1);
  #pragma omp parallel
  {
    #pragma omp single 
    {
      Hatrix::matmul(dense_map, lr_map, 0, 0, 0, max_level, b, c1);
    }
  }*/
  

  //Hatrix::Matrix q(leaf_size, 1);
  //std::vector<Hatrix::Matrix<double>> B_split = b.split(2, 1);
  //b.print();
  //B_split[0].print();
  //Hatrix::matmul(dense_map(0, 0, 1), B_split[0], q, false, false, 1, 0);
  //Hatrix::matmul(lr_map(0, 1, 1), B_split[1], q, false, false, 1, 1);
  //q.print();
  
  /*
  double error = Hatrix::norm(c - c1);
  std::cout<<"Error: "<<error<<std::endl;
  Hatrix::HODLR<double> test(D, leaf_size, rank);
  //Hatrix::Matrix test_dense = test.make_dense();
  Hatrix::Matrix c2(n, 1);
  test.matmul(b, c2);
  //Hatrix::matmul(test_dense, b, c2, false, false, 1, 0);
  error = Hatrix::norm(c - c2);
  std::cout<<"Error: "<<error<<std::endl;

  omp_set_num_threads(2);
  Hatrix::HODLR<float> test2(test);
  test.lu();
  test2.lu();
  Hatrix::Matrix x = Hatrix::generate_random_matrix(n,1);
  Hatrix::Matrix x_copy(b, true);
  test.solve(x_copy);
  Hatrix::Matrix<float> x_copy2(b, true);
  test2.solve(x_copy2);
  Hatrix::Matrix<double> x_conv(x_copy2);

  error = Hatrix::norm(x_copy - x_conv);
  std::cout<<"Error: "<<error<<std::endl;
  */

  Hatrix::HODLR<double> test(D, leaf_size, rank);
  Hatrix::Matrix check = test.make_dense();
  double error = Hatrix::norm(D_copy - check);
  std::cout<<"Error: "<<error<<std::endl;

  tic = get_time();    
  test.lu();
  toc = get_time();
  std::cout<<"Time (Construction & Factorization): "<<toc-tic<<std::endl;
  /*for (int i=0; i<num_blocks; i++) {
    for (int j=0; j<num_blocks; j++) {
      map(i,j,0).print();
    }
  }*/
  
  omp_set_num_threads(7);
  tic = get_time();
  Hatrix::LowRank<double> temp;
  #pragma omp parallel
  {
    #pragma omp single 
    {
      Hatrix::getrf<double>(dense_map, lr_map, is_admissible, 0, 0, 0, max_level, temp);
    }
  }
  
  toc = get_time();
  std::cout<<"Time: "<<toc-tic<<std::endl;
  
  Hatrix::Matrix b = Hatrix::generate_random_matrix(n,1);
  std::cout<<"Norm: "<<norm(b)<<std::endl;
  Hatrix::Matrix b_copy(b, true);
  Hatrix::Matrix b_copy2(b, true);
  test.solve(b_copy2);

  #pragma omp parallel
  {
    #pragma omp single 
    {
      Hatrix::trsm_solve<double>(dense_map, lr_map, 0, 0, 0, max_level, b, Hatrix::Left, Hatrix::Upper);
      Hatrix::trsm_solve<double>(dense_map, lr_map, 0, 0, 0, max_level, b, Hatrix::Left, Hatrix::Lower);
    }
  }
  //b.print();
  //Hatrix::solve_triangular(map(1,1,0), b_split[1], Hatrix::Left, Hatrix::Lower, true, false);

  lu_nopiv(D_copy);
  //D_copy.print();
  Hatrix::solve_triangular(D_copy, b_copy, Hatrix::Left, Hatrix::Upper, false, false);
  //b_copy.print();
  Hatrix::solve_triangular(D_copy, b_copy, Hatrix::Left, Hatrix::Lower, true, false);

  error = Hatrix::norm(b_copy - b);
  std::cout<<"Error: "<<error<<std::endl;
  error = Hatrix::norm(b_copy2 - b_copy);
  std::cout<<"Error: "<<error<<std::endl;
  
  // TODO the LU error seems to be good, but there is still a problem in the solve
  /*std::vector<Hatrix::Matrix<double>> lu_blocks1 = D_copy.split(2,2);
  std::vector<Hatrix::Matrix<double>> lu_blocks2 = lu_blocks1[0].split(2,2);
  error = Hatrix::norm(lu_blocks2[0] - dense_map(0,0,2));
  std::cout<<"Error (0,0): "<<error<<std::endl;
  error = Hatrix::norm(lu_blocks2[1] - lr_map(0,1,2).make_dense());
  std::cout<<"Error (0,1): "<<error<<std::endl;
  error = Hatrix::norm(lu_blocks2[2] - lr_map(1,0,2).make_dense());
  std::cout<<"Error (1,0): "<<error<<std::endl;
  error = Hatrix::norm(lu_blocks2[3] - dense_map(1,1,2));
  std::cout<<"Error (1,1): "<<error<<std::endl;

  error = Hatrix::norm(lu_blocks1[1] - lr_map(0,1,1).make_dense());
  std::cout<<"Error (0,1): "<<error<<std::endl;
  error = Hatrix::norm(lu_blocks1[2] - lr_map(1,0,1).make_dense());
  std::cout<<"Error (1,0): "<<error<<std::endl;

  lu_blocks2 = lu_blocks1[3].split(2,2);
  error = Hatrix::norm(lu_blocks2[0] - dense_map(2,2,2));
  std::cout<<"Error (0,0): "<<error<<std::endl;
  error = Hatrix::norm(lu_blocks2[1] - lr_map(2,3,2).make_dense());
  std::cout<<"Error (0,1): "<<error<<std::endl;
  error = Hatrix::norm(lu_blocks2[2] - lr_map(3,2,2).make_dense());
  std::cout<<"Error (1,0): "<<error<<std::endl;
  error = Hatrix::norm(lu_blocks2[3] - dense_map(3,3,2));
  std::cout<<"Error (1,1): "<<error<<std::endl;*/



  //b.print();
  //std::cout<<std::endl;
  //b_copy.print();
  return 0;
}