
#pragma once

#include <vector>
#include <algorithm>
#include <cstddef>

namespace nbd {

  typedef double real_t;
  constexpr int dim = 4;

  typedef void (*eval_func_t) (real_t&, real_t, real_t);

  struct EvalFunc {
    eval_func_t r2f;
    real_t singularity;
    real_t alpha;
  };

  struct Body {
    real_t X[dim];
  };

  typedef std::vector<Body> Bodies;

  struct Cell {
    int NCHILD = 0;
    int NBODY = 0;
    Cell* CHILD = NULL;
    Body* BODY = NULL;
    real_t C[dim];
    real_t R[dim];

    std::vector<Cell*> listFar;
    std::vector<Cell*> listNear;
  };

  typedef std::vector<Cell> Cells;

  struct Matrix {
    std::vector<real_t> A;
    int M;
    int N;
    int LDA;

    std::vector<real_t> B;
    int R;
    int LDB;

    Matrix() : M(0), N(0), LDA(0), R(0), LDB(0)
      { }

    Matrix(int m, int n, int lda) : M(m), N(n), LDA(std::max(lda, m)), R(0), LDB(0)
      { A.resize((size_t)LDA * N); }

    Matrix(int m, int n, int r, int lda, int ldb) : M(m), N(n), LDA(std::max(lda, m)), R(r), LDB(std::max(ldb, n))
      { A.resize((size_t)LDA * R); B.resize((size_t)LDB * R); }

    operator real_t*()
      { return A.data(); }

    operator const real_t*() const
      { return A.data(); }
  };

  typedef std::vector<Matrix> Matrices;

}
