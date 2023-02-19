#include <vector>
#include <iostream>
#include <cmath>
#include <omp.h>
#include <sys/time.h>

#include "Hatrix/Hatrix.h"

double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return (double)(tv.tv_sec+tv.tv_usec*1e-6);
}

namespace Hatrix {

  template <typename DT>
  RowColLevelMap<Matrix<DT>> create_map(const Matrix<DT>& A, int leaf_size, int num_blocks) {
    RowColLevelMap<Matrix<DT>> map;

    //TODO is this a good solution?
    omp_lock_t writelock;
    omp_init_lock(&writelock);

    #pragma omp parallel
    {
      #pragma omp single 
      {
        for (int i=0; i<num_blocks; ++i) {
          for (int j=0; j<num_blocks; ++j) {
            #pragma omp task
            {
              Matrix<DT> block_copy(leaf_size, leaf_size);
              for (int k=0; k<leaf_size; ++k) {
                for (int l=0; l<leaf_size; ++l) {
                  block_copy(k,l) = A(i*leaf_size+k, j*leaf_size+l);
                }
              }
              omp_set_lock(&writelock);
              map.insert(i, j, 0, std::move(block_copy));
              omp_unset_lock(&writelock);
            }
          }
        }
      }
    }
    return map;
  }
}

int main() {
  int n = 1024;
  int leaf_size = 16;
  int num_blocks = n / leaf_size;

  std::vector<std::vector<double>> randpts;
  randpts.push_back(Hatrix::equally_spaced_vector(2*n, 0.0, 1.0));

  Hatrix::Matrix D = Hatrix::generate_laplacend_matrix(randpts, n, n, 0, 0);
  Hatrix::Matrix D_copy(D, true);
  //D.print();
  double tic = get_time();
  Hatrix::RowColLevelMap<Hatrix::Matrix<double>> map = Hatrix::create_map(D, leaf_size, num_blocks);
  double toc = get_time();
  std::cout<<"Time (Construction): "<<toc-tic<<std::endl;
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
      for (int i=0; i<num_blocks; ++i) {
        //std::cout<<i<<std::endl;
        #pragma omp task depend(inout: map(i,i,0))
          Hatrix::lu_nopiv(map(i,i,0));
        for (int j=i+1; j<num_blocks; ++j) {
          #pragma omp task depend(in: map(i,i,0)) depend(inout: map(i,j,0))
          {//std::cout<<"TRSM "<<i<<","<<j<<std::endl;
            Hatrix::solve_triangular(map(i,i,0), map(i,j,0), Hatrix::Left, Hatrix::Lower, true, false);
          }
        }
        for (int j=i+1; j<num_blocks; ++j) {
          #pragma omp task depend(in: map(i,i,0)) depend(inout: map(j,i,0))
          {//std::cout<<"TRSM "<<j<<","<<i<<std::endl;
            Hatrix::solve_triangular(map(i,i,0), map(j,i,0), Hatrix::Right, Hatrix::Upper, false, false);
          }
        }
        //std::cout<<"here"<<std::endl;
        for (int j=i+1; j<num_blocks; ++j) {
          for (int k=i+1; k<num_blocks; ++k) {
            //std::cout<<j<<","<<k<<std::endl;
            #pragma omp task depend(in: map(j,i,0), map(i,k,0)) depend(inout: map(j,k,0))
            {
              Hatrix::matmul(map(j,i,0), map(i,k,0), map(j,k,0), false, false, -1, 1);
            }
          }
        }
      }
    }
  }
  toc = get_time();
  std::cout<<"Time: "<<toc-tic<<std::endl;

  Hatrix::Matrix b = Hatrix::generate_random_matrix(n,1);
  Hatrix::Matrix b_copy(b, true);

  std::vector<Hatrix::Matrix<double>> b_split = b.split(num_blocks, 1);
  Hatrix::Matrix<double>* b_split_ptr = b_split.data();

  #pragma omp parallel
  {
    #pragma omp single 
    {
      for (int i=num_blocks-1; i>=0; --i) {
        #pragma omp task depend(inout: b_split_ptr[i])
        {
          Hatrix::solve_triangular(map(i,i,0), b_split_ptr[i], Hatrix::Left, Hatrix::Upper, false, false);
        }
        for (int j=i-1; j>=0; --j) {//std::cout<<j<<","<<i<<std::endl;
          #pragma omp task depend(in: b_split_ptr[i]) depend(inout: b_split_ptr[j])
          {
            Hatrix::matmul(map(j,i,0), b_split[i], b_split[j], false, false, -1, 1);
          }
        }
      }
      //b.print();
      for (int i=0; i<num_blocks; ++i) {
        #pragma omp task depend(inout: b_split_ptr[i])
        {
          Hatrix::solve_triangular(map(i,i,0), b_split_ptr[i], Hatrix::Left, Hatrix::Lower, true, false);
        }
        for (int j=i+1; j<num_blocks; ++j) {//std::cout<<j<<","<<i<<std::endl;
          #pragma omp task depend(in: b_split_ptr[i]) depend(inout: b_split_ptr[j])
          {
            Hatrix::matmul(map(j,i,0), b_split[i], b_split[j], false, false, -1, 1);
          }
        }
      }
    }
  }
  //b.print();
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