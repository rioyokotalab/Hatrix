
#include "build_tree.h"
#include "kernel.h"
#include "aca.h"
#include "h2mv.h"
#include "test_util.h"
#include "timer.h"

#include <cstdio>
#include <random>
#include <chrono>


int main(int argc, char* argv[]) {

  using namespace nbd;

  int dim = 2;
  int m = 1900;
  int n = 1800;
  int leaf = 150;
  int rank = 80;
  int p = 20;
  double theta = 0.9;

  Bodies b1(m), b2(n);
  initRandom(b1, m, dim, 0, 1., 100);
  initRandom(b2, n, dim, 0, 1., 0);

  Cells c1 = buildTree(b1, leaf, dim);
  Cells c2 = buildTree(b2, leaf, dim);

  start("build H");
  auto fun = dim == 2 ? l2d() : l3d();
  Matrices d, bi, bj;
  traverse(fun, c1, c2, dim, d, theta, rank);
  stop("build H");

  {
    start("verify H");
    Matrix a_ref(m, n, m), a_rebuilt(m, n, m);
    convertHmat2Dense(fun, dim, c1, c2, d, a_rebuilt, a_rebuilt.LDA);
    P2Pnear(fun, &c1[0], &c2[0], dim, a_ref);
    printf("H-mat compress err %e\n", rel2err(&a_rebuilt[0], &a_ref[0], m, n, m, m));
    stop("verify H");
  }

  start("build H2");
  traverse_i(c1, c2, d, bi, p);
  traverse_j(c1, c2, d, bj, p);
  shared_epilogue(d);
  stop("build H2");

  {
    start("verify H2");
    Matrix a_ref(m, n, m), a_rebuilt(m, n, m);
    convertH2mat2Dense(fun, dim, c1, c2, bi, bj, d, a_rebuilt, a_rebuilt.LDA);
    P2Pnear(fun, &c1[0], &c2[0], dim, a_ref);
    printf("H2-mat compress err %e\n", rel2err(&a_rebuilt[0], &a_ref[0], m, n, m, m));
    stop("verify H2");
  }


  std::vector<double> x(n), b(m);
  vecRandom(&x[0], n, 1, 0, 1);

  start("H2-vec");
  h2mv_complete(fun, c1, c2, dim, bi, bj, d, &x[0], &b[0]);
  stop("H2-vec");

  std::vector<double> b_ref(m);

  mvec_kernel(fun, &c1[0], &c2[0], dim, 1., &x[0], 1, 0., &b_ref[0], 1);

  printf("H2-vec vs direct m-vec err %e\n", rel2err(&b[0], &b_ref[0], m, 1, m, m));

  //printTree(&c1[0], dim);
  //printTree(&c2[0], dim);

  return 0;
}
