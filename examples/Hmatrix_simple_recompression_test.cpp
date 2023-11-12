#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <omp.h>
#include <sys/time.h>
#include <iomanip>

#include "Hatrix/Hatrix.h"

double get_time() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return (double)(tv.tv_sec+tv.tv_usec*1e-6);
}

int main() {
  int n = 1024;
  int leaf_size = 32;
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
  Hatrix::Matrix D_dense(D, true);
  Hatrix::Matrix D_hodlr(D, true);
  Hatrix::Matrix D_hmatrix(D, true);

  Hatrix::Matrix b = Hatrix::generate_random_matrix(n,1);
  Hatrix::Matrix b_dense(b, true);
  Hatrix::Matrix b_hodlr(b, true);
  Hatrix::Matrix b_hmatrix(b, true);
  Hatrix::Matrix b_hmatrix2(b, true);

  omp_set_num_threads(1);
  double tic = get_time();
  lu_nopiv(D_dense);
  double toc = get_time();
  std::cout<<"Dense: "<<std::endl;
  std::cout<<"Factorization Time: "<<toc-tic<<std::endl;

  tic = get_time();   
  Hatrix::HODLR<double> hodlr(D, leaf_size, rank);
  toc = get_time();
  std::cout<<"HODLR: "<<std::endl;
  std::cout<<"Construction Time: "<<toc-tic<<std::endl;
  Hatrix::Matrix<double> hodlr_dense = hodlr.make_dense();
  std::cout<<"Construction Error: "<<Hatrix::norm(D - hodlr_dense)<<std::endl;
  /*
  tic = get_time();   
  Hatrix::Hmatrix<double> hmatrix(D_hmatrix, leaf_size, rank);
  toc = get_time();
  std::cout<<"H-matrix (RSVD): "<<std::endl;
  std::cout<<"Construction Time: "<<toc-tic<<std::endl;
  Hatrix::Matrix<double> hmatrix_dense = hmatrix.make_dense();
  std::cout<<"Construction Error: "<<Hatrix::norm(D - hmatrix_dense)<<std::endl;
  hmatrix.print_error();
  std::cout<<std::endl;
  hmatrix.print_error(D);
  std::cout<<std::endl;
  */

  tic = get_time();   
  Hatrix::Hmatrix<double> hmatrix2(D_hmatrix, leaf_size, rank, Hatrix::Approx::SVD);
  // need to copy before factorization
  //Hatrix::Hmatrix<float> hmatrix_f(hmatrix);
  toc = get_time();
  std::cout<<"H-matrix (SVD): "<<std::endl;
  std::cout<<"Construction Time: "<<toc-tic<<std::endl;
  Hatrix::Matrix<double> hmatrix2_dense = hmatrix2.make_dense();
  std::cout<<"Construction Error: "<<Hatrix::norm(D - hmatrix2_dense)<<std::endl;
  //hmatrix2.print_error();
  //std::cout<<std::endl;
  //hmatrix2.print_error(D);
  //std::cout<<std::endl;
  hmatrix2.lu();

  return 0;
}