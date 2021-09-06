
#include "test_util.h"
#include "kernel.h"

#include <iostream>
#include <random>
#include <cmath>

using namespace nbd;

void nbd::printVec(const real_t* a, int n, int inc) {
  for (int i = 0; i < n; i++)
    std::cout << a[i * inc] << " ";
  std::cout << std::endl;
}

void nbd::printMat(const real_t* a, int m, int n, int lda) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++)
      std::cout << a[i + j * lda] << " ";
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void nbd::printBodies(const Cell& c, int dim) {
  for (int i = 0; i < c.NBODY; i++) {
    std::cout << i << ": <";
    for (int j = 0; j < dim; j++)
      std::cout << c.BODY[i].X[j] << " ";
    std::cout << ">" << std::endl;
  }
}


void nbd::initRandom(Bodies& b, int m, int dim, real_t min, real_t max, unsigned int seed) {
  if (seed)
    std::srand(seed);
  for (auto& i : b) {
    for (int x = 0; x < dim; x++)
      i.X[x] = (max - min) * ((real_t)std::rand() / RAND_MAX) + min;
  }
}

void nbd::vecRandom(real_t* a, int n, int inc, real_t min, real_t max, unsigned int seed) {
  if (seed)
    std::srand(seed);
  for (int i = 0; i < n; i++) {
    a[i * inc] = (max - min) * ((real_t)std::rand() / RAND_MAX) + min;
  }
}


void nbd::convertHmat2Dense(EvalFunc ef, int dim, const Cells& icells, const Cells& jcells, const Matrices& d, real_t* a, int lda) {
  auto j_begin = jcells[0].BODY;
  auto i_begin = icells[0].BODY;
  int ld = (int)icells.size();

#pragma omp parallel for
  for (int y = 0; y < icells.size(); y++) {
    auto i = icells[y];
    auto yi = i.BODY - i_begin;
    for (auto& j : i.listNear) {
      auto _x = j - &jcells[0];
      auto xi = j->BODY - j_begin;
      Matrix m;
      P2Pnear(ef, &icells[y], &jcells[_x], dim, m);
      for (int jj = 0; jj < m.N; jj++)
        for (int ii = 0; ii < m.M; ii++)
          a[ii + yi + (jj + xi) * lda] = m.A[ii + (size_t)jj * m.LDA];
    }
    for (auto& j : i.listFar) {
      auto _x = j - &jcells[0];
      auto xi = j->BODY - j_begin;
      const Matrix& m = d[y + _x * ld];
      for (int jj = 0; jj < m.N; jj++)
        for (int ii = 0; ii < m.M; ii++) {
          double e = 0.;
          for (int k = 0; k < m.R; k++)
            e += m.A[ii + (size_t)k * m.LDA] * m.B[jj + (size_t)k * m.LDB];
          a[ii + yi + (jj + xi) * lda] = e;
        }
    }
  }
}


void nbd::convertFullBase(const Cell* cell, const Matrix* base, Matrix* base_full) {
  if (cell->NCHILD == 0) {
    *base_full = Matrix(base->M, base->N, base->M);
    std::copy(base->A.begin(), base->A.end(), base_full->A.begin());
    return;
  }

  for (auto c = cell->CHILD; c != cell->CHILD + cell->NCHILD; c++) {
    auto i = c - cell;
    convertFullBase(c, base + i, base_full + i);
  }

  *base_full = Matrix(cell->NBODY, base->N, cell->NBODY);
  int y_off = 0, k_off = 0;
  for (auto c = cell->CHILD; c != cell->CHILD + cell->NCHILD; c++) {
    auto i = c - cell;
    for (int jj = 0; jj < base->N; jj++)
      for (int ii = 0; ii < c->NBODY; ii++) {
        double e = 0.;
        for (int k = 0; k < base_full[i].N; k++)
          e += base_full[i].A[ii + (size_t)k * base_full[i].LDA] * base->A[k_off + k + (size_t)jj * base->LDA];
        real_t* bf = &(base_full->A)[y_off];
        bf[ii + (size_t)jj * base_full->LDA] = e;
      }
    k_off += base_full[i].N;
    y_off += c->NBODY;
  }
}

void nbd::mulUSV(const Matrix& u, const Matrix& v, const Matrix& s, real_t* a, int lda) {
  Matrix us(u.M, s.N, u.M);

  for (int jj = 0; jj < s.N; jj++)
    for (int ii = 0; ii < u.M; ii++) {
      double e = 0.;
      for (int k = 0; k < u.N; k++)
        e += u.A[ii + (size_t)k * u.LDA] * s.A[k + (size_t)jj * s.LDA];
      us.A[ii + (size_t)jj * us.LDA] = e;
    }

  for (int jj = 0; jj < v.M; jj++)
    for (int ii = 0; ii < u.M; ii++) {
      double e = 0.;
      for (int k = 0; k < s.N; k++)
        e += us.A[ii + (size_t)k * us.LDA] * v.A[jj + (size_t)k * v.LDA];
      a[ii + (size_t)jj * lda] = e;
    }
}


void nbd::convertH2mat2Dense(EvalFunc ef, int dim, const Cells& icells, const Cells& jcells, const Matrices& ibase, const Matrices& jbase, const Matrices& d, real_t* a, int lda) {
  auto j_begin = jcells[0].BODY;
  auto i_begin = icells[0].BODY;
  int ld = (int)icells.size();

  Matrices ibase_full(icells.size()), jbase_full(jcells.size());
  convertFullBase(&icells[0], &ibase[0], &ibase_full[0]);
  convertFullBase(&jcells[0], &jbase[0], &jbase_full[0]);

#pragma omp parallel for
  for (int y = 0; y < icells.size(); y++) {
    auto i = icells[y];
    auto yi = i.BODY - i_begin;
    for (auto& j : i.listNear) {
      auto _x = j - &jcells[0];
      auto xi = j->BODY - j_begin;
      Matrix m;
      P2Pnear(ef, &icells[y], &jcells[_x], dim, m);
      for (int jj = 0; jj < m.N; jj++)
        for (int ii = 0; ii < m.M; ii++)
          a[ii + yi + (jj + xi) * lda] = m.A[ii + (size_t)jj * m.LDA];
    }
    for (auto& j : i.listFar) {
      auto _x = j - &jcells[0];
      auto xi = j->BODY - j_begin;
      const Matrix& m = d[y + _x * ld];
      mulUSV(ibase_full[y], jbase_full[_x], m, a + yi + xi * lda, lda);
    }
  }
}


void nbd::printTree(const Cell* cell, int dim, int level, int offset_c, int offset_b) {
  for (int i = 0; i < level; i++)
    printf("  ");
  printf("%d: <%d, %d> C:<", offset_c, offset_b, offset_b + cell->NBODY);
  for (int d = 0; d < dim; d++)
    printf("%.3f ", cell->C[d]);
  printf("> R:<");
  for (int d = 0; d < dim; d++)
    printf("%.3f ", cell->R[d]);
  printf("> <Far: ");
  for (auto& c : cell->listFar)
    printf("%d ", offset_c + (int)(c - cell));
  printf("> <Near: ");
  for (auto& c : cell->listNear)
    printf("%d ", offset_c + (int)(c - cell));
  printf(">\n");
  for (auto c = cell->CHILD; c != cell->CHILD + cell->NCHILD; c++)
    printTree(c, dim, level + 1, offset_c + (int)(c - cell), offset_b + (int)(c->BODY - cell->BODY));
}

void nbd::printMatrixDim(const Matrix& a) {
  if (a.R > 0)
    printf("LR %d x %d by %d, %d x %d by %d\n", a.M, a.R, a.LDA, a.N, a.R, a.LDB);
  else
    printf("D %d x %d by %d\n", a.M, a.N, a.LDA);
}

real_t nbd::rel2err(const real_t* a, const real_t* ref, int m, int n, int lda, int ldref) {
  real_t err = 0., nrm = 0.;
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      real_t e = ref[i + j * ldref] - a[i + j * lda];
      err += e * e;
      nrm += a[i + j * lda] * a[i + j * lda];
    }
  }

  return std::sqrt(err / nrm);
}

