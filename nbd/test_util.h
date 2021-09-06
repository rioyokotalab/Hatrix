
#pragma once

#include "nbd.h"

namespace nbd {

  void printVec(const real_t* a, int n, int inc);

  void printMat(const real_t* a, int m, int n, int lda);

  void printBodies(const Cell& c, int dim);

  void initRandom(Bodies& b, int m, int dim, real_t min, real_t max, unsigned int seed = 0);

  void vecRandom(real_t* a, int n, int inc, real_t min, real_t max, unsigned int seed = 0);

  void convertHmat2Dense(EvalFunc ef, int dim, const Cells& icells, const Cells& jcells, const Matrices& d, real_t* a, int lda);

  void convertFullBase(const Cell* cell, const Matrix* base, Matrix* base_full);

  void mulUSV(const Matrix& u, const Matrix& v, const Matrix& s, real_t* a, int lda);

  void convertH2mat2Dense(EvalFunc ef, int dim, const Cells& icells, const Cells& jcells, const Matrices& ibase, const Matrices& jbase, const Matrices& d, real_t* a, int lda);

  void printMatrixDim(const Matrix& a);

  void printTree(const Cell* cell, int dim, int level = 0, int offset_c = 0, int offset_b = 0);

  real_t rel2err(const real_t* a, const real_t* ref, int m, int n, int lda, int ldref);


}
