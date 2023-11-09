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

  omp_set_num_threads(1);
  double tic = get_time();
  lu_nopiv(D_dense);
  double toc = get_time();
  std::cout<<"Dense: "<<std::endl;
  std::cout<<"Factorization Time: "<<toc-tic<<std::endl;
  tic = get_time();
  Hatrix::solve_triangular(D_dense, b_dense, Hatrix::Left, Hatrix::Lower, true, false);
  Hatrix::solve_triangular(D_dense, b_dense, Hatrix::Left, Hatrix::Upper, false, false);
  toc = get_time();
  std::cout<<"Solution Time: "<<toc-tic<<std::endl;
  Hatrix::Matrix residual(n, 1);
  Hatrix::matmul(D, b_dense, residual, false, false, 1, 0);
  residual -= b;
  std::cout<<std::setprecision(16)<<"Solve Error: "<<Hatrix::norm(residual)<<std::endl<<std::endl;

  tic = get_time();   
  Hatrix::HODLR<double> hodlr(D, leaf_size, rank);
  toc = get_time();
  std::cout<<"HODLR: "<<std::endl;
  std::cout<<"Construction Time: "<<toc-tic<<std::endl;
  Hatrix::Matrix<double> hodlr_dense = hodlr.make_dense();
  std::cout<<"Construction Error: "<<Hatrix::norm(D - hodlr_dense)<<std::endl;
  tic = get_time();    
  hodlr.lu();
  toc = get_time();
  std::cout<<"Factorization Time: "<<toc-tic<<std::endl;
  Hatrix::Matrix<double> hodlr_lu_dense = hodlr.make_dense();
  std::cout<<"Factorization Error: "<<Hatrix::norm(D_dense - hodlr_lu_dense)<<std::endl;
  tic = get_time();
  hodlr.solve(b_hodlr);
  toc = get_time();
  std::cout<<"Solution Time: "<<toc-tic<<std::endl;
  Hatrix::matmul(D, b_hodlr, residual, false, false, 1, 0);
  residual -= b;
  std::cout<<"Solve Error: "<<Hatrix::norm(residual)<<std::endl<<std::endl;

  tic = get_time();   
  Hatrix::Hmatrix<double> hmatrix(D_hmatrix, leaf_size, rank);
  // need to copy before factorization
  Hatrix::Hmatrix<float> hmatrix_f(hmatrix);
  toc = get_time();
  std::cout<<"H-matrix: "<<std::endl;
  std::cout<<"Construction Time: "<<toc-tic<<std::endl;
  Hatrix::Matrix<double> hmatrix_dense = hmatrix.make_dense();
  std::cout<<"Construction Error: "<<Hatrix::norm(D - hmatrix_dense)<<std::endl;
  tic = get_time();    
  hmatrix.lu();
  toc = get_time();
  std::cout<<"Factorization Time: "<<toc-tic<<std::endl;
  Hatrix::Matrix<double> hmatrix_lu_dense = hmatrix.make_dense();
  std::cout<<"Factorization Error: "<<Hatrix::norm(D_dense - hmatrix_lu_dense)<<std::endl;
  tic = get_time();
  hmatrix.solve(b_hmatrix);
  toc = get_time();
  std::cout<<"Solution Time: "<<toc-tic<<std::endl;
  Hatrix::matmul(D, b_hmatrix, residual, false, false, 1, 0);
  residual -= b;
  std::cout<<"SolveError: "<<Hatrix::norm(residual)<<std::endl<<std::endl;
  
  // single precision
  Hatrix::Matrix<float> D_dense_f(D);
  Hatrix::Matrix<float> b_dense_f(b);
  Hatrix::Matrix<float> b_hmatrix_f(b);

  std::cout<<"Dense (single precision): "<<std::endl;
  Hatrix::Matrix<double> D_dense_d(D_dense_f);
  std::cout<<"Construction Error: "<<Hatrix::norm(D - D_dense_d)<<std::endl;
  tic = get_time();
  lu_nopiv(D_dense_f);
  toc = get_time();
  std::cout<<"Factorization Time: "<<toc-tic<<std::endl;
  Hatrix::Matrix<double> D_dense_lu_d(D_dense_f);
  std::cout<<"Factorization Error: "<<Hatrix::norm(D_dense - D_dense_lu_d)<<std::endl;
  tic = get_time();
  Hatrix::solve_triangular(D_dense_f, b_dense_f, Hatrix::Left, Hatrix::Lower, true, false);
  Hatrix::solve_triangular(D_dense_f, b_dense_f, Hatrix::Left, Hatrix::Upper, false, false);
  toc = get_time();
  std::cout<<"Solution Time: "<<toc-tic<<std::endl;
  Hatrix::Matrix<double> x_dense(b_dense_f);
  Hatrix::matmul(D, x_dense, residual, false, false, 1, 0);
  residual -= b;
  std::cout<<"Solve Error: "<<Hatrix::norm(residual)<<std::endl<<std::endl;

  std::cout<<"H-matrix (single precision): "<<std::endl;
  Hatrix::Matrix<float> hmatrix_dense_f = hmatrix_f.make_dense();
  Hatrix::Matrix<double> hmatrix_dense_f_d(hmatrix_dense_f);
  std::cout<<"Construction Error: "<<Hatrix::norm(D - hmatrix_dense_f_d)<<std::endl;
  tic = get_time();    
  hmatrix_f.lu();
  toc = get_time();
  std::cout<<"Factorization Time: "<<toc-tic<<std::endl;
  Hatrix::Matrix<float> hmatrix_dense_lu_f = hmatrix_f.make_dense();
  Hatrix::Matrix<double> hmatrix_dense_lu_f_d(hmatrix_dense_lu_f);
  std::cout<<"Factorization Error: "<<Hatrix::norm(D_dense - hmatrix_dense_lu_f_d)<<std::endl;
  tic = get_time();
  hmatrix_f.solve(b_hmatrix_f);
  toc = get_time();
  std::cout<<"Solution Time: "<<toc-tic<<std::endl;
  Hatrix::Matrix<double> x_hmatrix(b_hmatrix_f);
  Hatrix::matmul(D, x_hmatrix, residual, false, false, 1, 0);
  residual -= b;
  std::cout<<"Solve Error: "<<Hatrix::norm(residual)<<std::endl<<std::endl;
  
  return 0;
}