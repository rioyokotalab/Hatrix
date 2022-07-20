
#include "mpi.h"
#include "stdint.h"
#include "stddef.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "string.h"
#include "inttypes.h"

#ifdef USE_MKL
#include "mkl.h"
#else
#include "cblas.h"
#include "lapacke.h"
#endif

struct Body { double X[3], B; };
struct Matrix { double* A; int64_t M, N; };
struct Cell { int64_t Child, Body[2], Level, Procs[2]; double R[3], C[3]; };
struct CSC { int64_t M, N, *ColIndex, *RowIndex; };
struct CellComm { struct CSC Comms; int64_t Proc[3], *ProcRootI, *ProcBoxes, *ProcBoxesEnd; MPI_Comm Comm_share, Comm_merge, *Comm_box; };
struct Base { int64_t Ulen, *Lchild, *Dims, *DimsLr, *Offsets, *Multipoles; struct Matrix *Uo, *Uc, *R; };
struct Node { int64_t lenA, lenS; struct Matrix *A, *S, *A_cc, *A_oc, *A_oo; };
struct RightHandSides { int64_t Xlen; struct Matrix *X, *XcM, *XoL, *B; };

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void matrix_mem(int64_t* bytes, const struct Matrix* A, int64_t lenA) {
  int64_t count = sizeof(struct Matrix) * lenA;
  for (int64_t i = 0; i < lenA; i++)
    count = count + sizeof(double) * A[i].M * A[i].N;
  *bytes = count;
}

void basis_mem(int64_t* bytes, const struct Base* basis, int64_t levels) {
  int64_t count = sizeof(struct Base) * levels;
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nodes = basis[i].Ulen;
    int64_t bytes_o, bytes_c, bytes_r;
    matrix_mem(&bytes_o, &basis[i].Uo[0], nodes);
    matrix_mem(&bytes_c, &basis[i].Uc[0], nodes);
    matrix_mem(&bytes_r, &basis[i].R[0], nodes);

    count = count + bytes_o + bytes_c + bytes_r;
  }
  *bytes = count;
}

void node_mem(int64_t* bytes, const struct Node* node, int64_t levels) {
  int64_t count = sizeof(struct Node) * levels;
  for (int64_t i = 0; i <= levels; i++) {
    int64_t nnz = node[i].lenA;
    int64_t nnz_f = node[i].lenS;
    int64_t bytes_a, bytes_s;
    matrix_mem(&bytes_a, &node[i].A[0], nnz);
    matrix_mem(&bytes_s, &node[i].S[0], nnz_f);
    count = count + bytes_a + bytes_s;
  }
  *bytes = count;
}

void rightHandSides_mem(int64_t* bytes, const struct RightHandSides* rhs, int64_t levels) {
  int64_t count = sizeof(struct RightHandSides) * levels;
  for (int64_t i = 0; i <= levels; i++) {
    int64_t len = rhs[i].Xlen;
    int64_t bytes_x, bytes_o, bytes_c;
    matrix_mem(&bytes_x, &rhs[i].X[0], len);
    matrix_mem(&bytes_o, &rhs[i].XoL[0], len);
    matrix_mem(&bytes_c, &rhs[i].XcM[0], len);
    count = count + bytes_x * 2 + bytes_o + bytes_c;
  }
  *bytes = count;
}

double tot_cm_time = 0.;

void startTimer(double* wtime, double* cmtime) {
  MPI_Barrier(MPI_COMM_WORLD);
  *wtime = MPI_Wtime();
  *cmtime = tot_cm_time;
}

void stopTimer(double* wtime, double* cmtime) {
  MPI_Barrier(MPI_COMM_WORLD);
  double etime = MPI_Wtime();
  double time[2] = { etime - *wtime, tot_cm_time - *cmtime };
  MPI_Allreduce(MPI_IN_PLACE, time, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  int mpi_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  *wtime = time[0] / mpi_size;
  *cmtime = time[1] / mpi_size;
}

void recordCommTime(double cmtime) {
  tot_cm_time = tot_cm_time + cmtime;
}

void getCommTime(double* cmtime) {
#ifndef _PROF
  printf("Communication time not recorded: Profiling macro not defined, compile lib with -D_PROF.\n");
  *cmtime = -1;
#else
  *cmtime = tot_cm_time;
#endif
}

double _singularity = 1.e-8;
double _alpha = 1.;

void laplace3d(double* r2) {
  double _r2 = *r2;
  double r = sqrt(_r2) + _singularity;
  *r2 = 1. / r;
}

void yukawa3d(double* r2) {
  double _r2 = *r2;
  double r = sqrt(_r2) + _singularity;
  *r2 = exp(_alpha * -r) / r;
}

void set_kernel_constants(double singularity, double alpha) {
  _singularity = singularity;
  _alpha = alpha;
}

void gen_matrix(void(*ef)(double*), int64_t m, int64_t n, const struct Body* bi, const struct Body* bj, double Aij[], const int64_t sel_i[], const int64_t sel_j[]) {
  for (int64_t i = 0; i < m * n; i++) {
    int64_t x = i / m;
    int64_t bx = sel_j == NULL ? x : sel_j[x];
    int64_t y = i - x * m;
    int64_t by = sel_i == NULL ? y : sel_i[y];

    const struct Body* bii = bi + by;
    const struct Body* bjj = bj + bx;

    double dX = bii->X[0] - bjj->X[0];
    double dY = bii->X[1] - bjj->X[1];
    double dZ = bii->X[2] - bjj->X[2];

    double r2 = dX * dX + dY * dY + dZ * dZ;
    ef(&r2);
    Aij[x * m + y] = r2;
  }
}

void uniform_unit_cube(struct Body* bodies, int64_t nbodies, int64_t dim, unsigned int seed) {
  if (seed > 0)
    srand(seed);

  for (int64_t i = 0; i < nbodies; i++) {
    double r0 = dim > 0 ? ((double)rand() / RAND_MAX) : 0.;
    double r1 = dim > 1 ? ((double)rand() / RAND_MAX) : 0.;
    double r2 = dim > 2 ? ((double)rand() / RAND_MAX) : 0.;
    bodies[i].X[0] = r0;
    bodies[i].X[1] = r1;
    bodies[i].X[2] = r2;
  }
}

void mesh_unit_sphere(struct Body* bodies, int64_t nbodies) {
  int64_t mlen = nbodies - 2;
  if (mlen < 0) {
    fprintf(stderr, "Error spherical mesh size (GT/EQ. 2 required): %" PRId64 ".\n", nbodies);
    return;
  }

  double alen = sqrt(mlen);
  int64_t m = (int64_t)ceil(alen);
  int64_t n = (int64_t)ceil((double)mlen / m);

  double pi = M_PI;
  double seg_theta = pi / (m + 1);
  double seg_phi = 2. * pi / n;

  for (int64_t i = 0; i < mlen; i++) {
    int64_t x = i / m;
    int64_t y = 1 + i - x * m;
    int64_t x2 = !(y & 1);

    double theta = y * seg_theta;
    double phi = (0.5 * x2 + x) * seg_phi;

    double cost = cos(theta);
    double sint = sin(theta);
    double cosp = cos(phi);
    double sinp = sin(phi);

    double* x_bi = bodies[i + 1].X;
    x_bi[0] = sint * cosp;
    x_bi[1] = sint * sinp;
    x_bi[2] = cost;
  }

  bodies[0].X[0] = 0.;
  bodies[0].X[1] = 0.;
  bodies[0].X[2] = 1.;

  bodies[nbodies - 1].X[0] = 0.;
  bodies[nbodies - 1].X[1] = 0.;
  bodies[nbodies - 1].X[2] = -1.;
}

void mesh_unit_cube(struct Body* bodies, int64_t nbodies) {
  if (nbodies < 0) {
    fprintf(stderr, "Error cubic mesh size (GT/EQ. 0 required): %" PRId64 ".\n", nbodies);
    return;
  }

  int64_t mlen = (int64_t)ceil((double)nbodies / 6.);
  double alen = sqrt(mlen);
  int64_t m = (int64_t)ceil(alen);
  int64_t n = (int64_t)ceil((double)mlen / m);

  double seg_fv = 1. / (m - 1);
  double seg_fu = 1. / n;
  double seg_sv = 1. / (m + 1);
  double seg_su = 1. / (n + 1);

  for (int64_t i = 0; i < nbodies; i++) {
    int64_t face = i / mlen;
    int64_t ii = i - face * mlen;
    int64_t x = ii / m;
    int64_t y = ii - x * m;
    int64_t x2 = y & 1;

    double u, v;
    double* x_bi = bodies[i].X;

    switch (face) {
      case 0: // POSITIVE X
        v = y * seg_fv;
        u = (0.5 * x2 + x) * seg_fu;
        x_bi[0] = 1.;
        x_bi[1] = 2. * v - 1.;
        x_bi[2] = -2. * u + 1.;
        break;
      case 1: // NEGATIVE X
        v = y * seg_fv;
        u = (0.5 * x2 + x) * seg_fu;
        x_bi[0] = -1.;
        x_bi[1] = 2. * v - 1.;
        x_bi[2] = 2. * u - 1.;
        break;
      case 2: // POSITIVE Y
        v = (y + 1) * seg_sv;
        u = (0.5 * x2 + x + 1) * seg_su;
        x_bi[0] = 2. * u - 1.;
        x_bi[1] = 1.;
        x_bi[2] = -2. * v + 1.;
        break;
      case 3: // NEGATIVE Y
        v = (y + 1) * seg_sv;
        u = (0.5 * x2 + x + 1) * seg_su;
        x_bi[0] = 2. * u - 1.;
        x_bi[1] = -1.;
        x_bi[2] = 2. * v - 1.;
        break;
      case 4: // POSITIVE Z
        v = y * seg_fv;
        u = (0.5 * x2 + x) * seg_fu;
        x_bi[0] = 2. * u - 1.;
        x_bi[1] = 2. * v - 1.;
        x_bi[2] = 1.;
        break;
      case 5: // NEGATIVE Z
        v = y * seg_fv;
        u = (0.5 * x2 + x) * seg_fu;
        x_bi[0] = -2. * u + 1.;
        x_bi[1] = 2. * v - 1.;
        x_bi[2] = -1.;
        break;
    }
  }
}

void magnify_reloc(struct Body* bodies, int64_t nbodies, const double Ccur[], const double Cnew[], const double R[]) {
  for (int64_t i = 0; i < nbodies; i++) {
    double* x_bi = bodies[i].X;
    double v0 = x_bi[0] - Ccur[0];
    double v1 = x_bi[1] - Ccur[1];
    double v2 = x_bi[2] - Ccur[2];
    x_bi[0] = Cnew[0] + R[0] * v0;
    x_bi[1] = Cnew[1] + R[1] * v1;
    x_bi[2] = Cnew[2] + R[2] * v2;
  }
}

void body_neutral_charge(struct Body* bodies, int64_t nbodies, double cmax, unsigned int seed) {
  if (seed > 0)
    srand(seed);

  double avg = 0.;
  double cmax2 = cmax * 2;
  for (int64_t i = 0; i < nbodies; i++) {
    double c = ((double)rand() / RAND_MAX) * cmax2 - cmax;
    bodies[i].B = c;
    avg = avg + c;
  }
  avg = avg / nbodies;

  if (avg != 0.)
    for (int64_t i = 0; i < nbodies; i++)
      bodies[i].B = bodies[i].B - avg;
}

void get_bounds(const struct Body* bodies, int64_t nbodies, double R[], double C[]) {
  double Xmin[3];
  double Xmax[3];
  Xmin[0] = Xmax[0] = bodies[0].X[0];
  Xmin[1] = Xmax[1] = bodies[0].X[1];
  Xmin[2] = Xmax[2] = bodies[0].X[2];

  for (int64_t i = 1; i < nbodies; i++) {
    const double* x_bi = bodies[i].X;
    Xmin[0] = fmin(x_bi[0], Xmin[0]);
    Xmin[1] = fmin(x_bi[1], Xmin[1]);
    Xmin[2] = fmin(x_bi[2], Xmin[2]);

    Xmax[0] = fmax(x_bi[0], Xmax[0]);
    Xmax[1] = fmax(x_bi[1], Xmax[1]);
    Xmax[2] = fmax(x_bi[2], Xmax[2]);
  }

  C[0] = (Xmin[0] + Xmax[0]) / 2.;
  C[1] = (Xmin[1] + Xmax[1]) / 2.;
  C[2] = (Xmin[2] + Xmax[2]) / 2.;

  double d0 = Xmax[0] - Xmin[0];
  double d1 = Xmax[1] - Xmin[1];
  double d2 = Xmax[2] - Xmin[2];

  R[0] = (d0 == 0. && Xmin[0] == 0.) ? 0. : (1.e-8 + d0 / 2.);
  R[1] = (d1 == 0. && Xmin[1] == 0.) ? 0. : (1.e-8 + d1 / 2.);
  R[2] = (d2 == 0. && Xmin[2] == 0.) ? 0. : (1.e-8 + d2 / 2.);
}

int comp_bodies_s0(const void *a, const void *b) {
  struct Body* body_a = (struct Body*)a;
  struct Body* body_b = (struct Body*)b;
  double diff = (body_a->X)[0] - (body_b->X)[0];
  return diff < 0. ? -1 : (int)(diff > 0.);
}

int comp_bodies_s1(const void *a, const void *b) {
  struct Body* body_a = (struct Body*)a;
  struct Body* body_b = (struct Body*)b;
  double diff = (body_a->X)[1] - (body_b->X)[1];
  return diff < 0. ? -1 : (int)(diff > 0.);
}

int comp_bodies_s2(const void *a, const void *b) {
  struct Body* body_a = (struct Body*)a;
  struct Body* body_b = (struct Body*)b;
  double diff = (body_a->X)[2] - (body_b->X)[2];
  return diff < 0. ? -1 : (int)(diff > 0.);
}

void sort_bodies(struct Body* bodies, int64_t nbodies, int64_t sdim) {
  size_t size = sizeof(struct Body);
  if (sdim == 0)
    qsort(bodies, nbodies, size, comp_bodies_s0);
  else if (sdim == 1)
    qsort(bodies, nbodies, size, comp_bodies_s1);
  else if (sdim == 2)
    qsort(bodies, nbodies, size, comp_bodies_s2);
}

void read_sorted_bodies(int64_t nbodies, int64_t lbuckets, struct Body* bodies, int64_t buckets[], const char* fname) {
  FILE* file = fopen(fname, "r");
  int64_t curr = 1;
  int64_t cbegin = 0;
  for (int64_t i = 0; i < nbodies; i++) {
    int b; double x, y, z;
    int ret = fscanf(file, "%lf %lf %lf %d", &x, &y, &z, &b);
    bodies[i].X[0] = x;
    bodies[i].X[1] = y;
    bodies[i].X[2] = z;
    while (curr < b && curr <= lbuckets) {
      buckets[curr - 1] = i - cbegin;
      cbegin = i;
      curr++;
    }
    if (ret == EOF)
      break;
  }
  while (curr <= lbuckets) {
    buckets[curr - 1] = nbodies - cbegin;
    cbegin = nbodies;
    curr++;
  }
}

void mat_vec_reference(void(*ef)(double*), int64_t begin, int64_t end, double B[], int64_t nbodies, const struct Body* bodies) {
  int64_t m = end - begin;
  int64_t n = nbodies;
  for (int64_t i = 0; i < m; i++) {
    int64_t y = begin + i;
    const struct Body* bi = bodies + y;
    double s = 0.;
    for (int64_t j = 0; j < n; j++) {
      const struct Body* bj = bodies + j;
      double dX = bi->X[0] - bj->X[0];
      double dY = bi->X[1] - bj->X[1];
      double dZ = bi->X[2] - bj->X[2];

      double r2 = dX * dX + dY * dY + dZ * dZ;
      ef(&r2);
      s = s + r2 * bj->B;
    }
    B[i] = s;
  }
}

void cpyMatToMat(int64_t m, int64_t n, const struct Matrix* m1, struct Matrix* m2, int64_t y1, int64_t x1, int64_t y2, int64_t x2) {
#ifdef USE_MKL
  mkl_domatcopy('C', 'N', m, n, 1., &(m1->A)[y1 + x1 * m1->M], m1->M, &(m2->A)[y2 + x2 * m2->M], m2->M);
#else
  for (int64_t j = 0; j < n; j++) {
    int64_t j1 = y1 + (x1 + j) * m1->M;
    int64_t j2 = y2 + (x2 + j) * m2->M;
    memcpy(&(m2->A)[j2], &(m1->A)[j1], sizeof(double) * m);
  }
#endif
}

void qr_full(struct Matrix* Q, struct Matrix* R, double* tau) {
  LAPACKE_dgeqrf(LAPACK_COL_MAJOR, Q->M, R->N, Q->A, Q->M, tau);
  cpyMatToMat(R->M, R->N, Q, R, 0, 0, 0, 0);
  LAPACKE_dorgqr(LAPACK_COL_MAJOR, Q->M, Q->N, R->N, Q->A, Q->M, tau);
}

void updateSubU(struct Matrix* U, const struct Matrix* R1, const struct Matrix* R2) {
  cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, R1->N, U->N, 1., R1->A, R1->M, U->A, U->M);
  cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, R2->N, U->N, 1., R2->A, R2->M, &U->A[R1->N], U->M);
}

void svd_U(struct Matrix* A, struct Matrix* U, double* S) {
  int64_t rank_a = A->M < A->N ? A->M : A->N;
  LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A', 'N', A->M, A->N, A->A, A->M, S, U->A, A->M, NULL, A->N, &S[rank_a]);
}

void mul_AS(struct Matrix* A, double* S) {
  for (int64_t i = 0; i < A->N; i++)
    cblas_dscal(A->M, S[i], &(A->A)[i * A->M], 1);
}

void id_row(struct Matrix* U, int32_t arows[], double* work) {
  struct Matrix A = (struct Matrix){ work, U->M, U->N };
  cblas_dcopy(A.M * A.N, U->A, 1, A.A, 1);
  LAPACKE_dgetrf(LAPACK_COL_MAJOR, A.M, A.N, A.A, A.M, arows);
  cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, A.M, A.N, 1., A.A, A.M, U->A, A.M);
  cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit, A.M, A.N, 1., A.A, A.M, U->A, A.M);
}

void mmult(char ta, char tb, const struct Matrix* A, const struct Matrix* B, struct Matrix* C, double alpha, double beta) {
  int64_t k = (ta == 'N' || ta == 'n') ? A->N : A->M;
  CBLAS_TRANSPOSE tac = (ta == 'T' || ta == 't') ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE tbc = (tb == 'T' || tb == 't') ? CblasTrans : CblasNoTrans;
  cblas_dgemm(CblasColMajor, tac, tbc, C->M, C->N, k, alpha, A->A, A->M, B->A, B->M, beta, C->A, C->M);
}

void chol_decomp(struct Matrix* A) {
  LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', A->M, A->A, A->M);
}

void trsm_lowerA(struct Matrix* A, const struct Matrix* L) {
  cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, A->M, A->N, 1., L->A, L->M, A->A, A->M);
}

void rsr(const struct Matrix* R1, const struct Matrix* R2, struct Matrix* S) {
  cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, R1->N, S->N, 1., R1->A, R1->M, S->A, S->M);
  cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans, CblasNonUnit, S->M, R2->N, 1., R2->A, R2->M, S->A, S->M);
}

void mat_solve(char type, struct Matrix* X, const struct Matrix* A) {
  if (type == 'F' || type == 'f' || type == 'A' || type == 'a')
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, X->M, X->N, 1., A->A, A->M, X->A, X->M);
  if (type == 'B' || type == 'b' || type == 'A' || type == 'a')
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit, X->M, X->N, 1., A->A, A->M, X->A, X->M);
}

void nrm2_A(struct Matrix* A, double* nrm) {
  int64_t len_A = A->M * A->N;
  double nrm_A = cblas_dnrm2(len_A, A->A, 1);
  *nrm = nrm_A;
}

void scal_A(struct Matrix* A, double alpha) {
  int64_t len_A = A->M * A->N;
  cblas_dscal(len_A, alpha, A->A, 1);
}

void buildTree(int64_t* ncells, struct Cell* cells, struct Body* bodies, int64_t nbodies, int64_t levels) {
  int __mpi_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &__mpi_size);
  int64_t mpi_size = __mpi_size;

  struct Cell* root = &cells[0];
  root->Body[0] = 0;
  root->Body[1] = nbodies;
  root->Level = 0;
  root->Procs[0] = 0;
  root->Procs[1] = mpi_size;
  get_bounds(bodies, nbodies, root->R, root->C);

  int64_t len = 1;
  int64_t i = 0;
  while (i < len) {
    struct Cell* ci = &cells[i];
    ci->Child = -1;

    if (ci->Level < levels) {
      int64_t sdim = 0;
      double maxR = ci->R[0];
      if (ci->R[1] > maxR)
      { sdim = 1; maxR = ci->R[1]; }
      if (ci->R[2] > maxR)
      { sdim = 2; maxR = ci->R[2]; }

      int64_t i_begin = ci->Body[0];
      int64_t i_end = ci->Body[1];
      int64_t nbody_i = i_end - i_begin;
      sort_bodies(&bodies[i_begin], nbody_i, sdim);
      int64_t loc = i_begin + nbody_i / 2;

      struct Cell* c0 = &cells[len];
      struct Cell* c1 = &cells[len + 1];
      ci->Child = len;
      len = len + 2;

      c0->Body[0] = i_begin;
      c0->Body[1] = loc;
      c1->Body[0] = loc;
      c1->Body[1] = i_end;
      
      c0->Level = ci->Level + 1;
      c1->Level = ci->Level + 1;
      c0->Procs[0] = ci->Procs[0];
      c1->Procs[1] = ci->Procs[1];

      int64_t divp = (ci->Procs[1] - ci->Procs[0]) / 2;
      if (divp >= 1) {
        int64_t p = divp + ci->Procs[0];
        c0->Procs[1] = p;
        c1->Procs[0] = p;
      }
      else {
        c0->Procs[1] = ci->Procs[1];
        c1->Procs[0] = ci->Procs[0];
      }

      get_bounds(&bodies[i_begin], loc - i_begin, c0->R, c0->C);
      get_bounds(&bodies[loc], i_end - loc, c1->R, c1->C);
    }
    i++;
  }
  *ncells = len;
}

void buildTreeBuckets(struct Cell* cells, const struct Body* bodies, const int64_t buckets[], int64_t levels) {
  int64_t nleaf = (int64_t)1 << levels;
  int64_t count = 0;
  for (int64_t i = 0; i < nleaf; i++) {
    int64_t ci = i + nleaf - 1;
    cells[ci].Child = -1;
    cells[ci].Body[0] = count;
    cells[ci].Body[1] = count + buckets[i];
    cells[ci].Level = levels;
    get_bounds(&bodies[count], buckets[i], cells[ci].R, cells[ci].C);
    count = count + buckets[i];
  }

  for (int64_t i = nleaf - 2; i >= 0; i--) {
    int64_t c0 = (i << 1) + 1;
    int64_t c1 = (i << 1) + 2;
    int64_t begin = cells[c0].Body[0];
    int64_t len = cells[c1].Body[1] - begin;
    cells[i].Child = c0;
    cells[i].Body[0] = begin;
    cells[i].Body[1] = begin + len;
    cells[i].Level = cells[c0].Level - 1;
    get_bounds(&bodies[begin], len, cells[i].R, cells[i].C);
  }

  int __mpi_size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &__mpi_size);
  int64_t mpi_size = __mpi_size;
  cells[0].Procs[0] = 0;
  cells[0].Procs[1] = mpi_size;

  for (int64_t i = 0; i < nleaf - 1; i++) {
    struct Cell* ci = &cells[i];
    struct Cell* c0 = &cells[ci->Child];
    struct Cell* c1 = c0 + 1;
    int64_t divp = (ci->Procs[1] - ci->Procs[0]) / 2;
    if (divp >= 1) {
      int64_t p = divp + ci->Procs[0];
      c0->Procs[1] = p;
      c1->Procs[0] = p;
    }
    else {
      c0->Procs[1] = ci->Procs[1];
      c1->Procs[0] = ci->Procs[0];
    }
    c0->Procs[0] = ci->Procs[0];
    c1->Procs[1] = ci->Procs[1];
  }
}

int admis_check(double theta, const double C1[], const double C2[], const double R1[], const double R2[]) {
  double dCi[3];
  dCi[0] = C1[0] - C2[0];
  dCi[1] = C1[1] - C2[1];
  dCi[2] = C1[2] - C2[2];

  dCi[0] = dCi[0] * dCi[0];
  dCi[1] = dCi[1] * dCi[1];
  dCi[2] = dCi[2] * dCi[2];

  double dRi[3];
  dRi[0] = R1[0] * R1[0];
  dRi[1] = R1[1] * R1[1];
  dRi[2] = R1[2] * R1[2];

  double dRj[3];
  dRj[0] = R2[0] * R2[0];
  dRj[1] = R2[1] * R2[1];
  dRj[2] = R2[2] * R2[2];

  double dC = dCi[0] + dCi[1] + dCi[2];
  double dR = (dRi[0] + dRi[1] + dRi[2] + dRj[0] + dRj[1] + dRj[2]) * theta;
  return (int)(dC > dR);
}

void getList(char NoF, int64_t* len, int64_t rels[], int64_t ncells, const struct Cell cells[], int64_t i, int64_t j, double theta) {
  const struct Cell* Ci = &cells[i];
  const struct Cell* Cj = &cells[j];
  int64_t ilevel = Ci->Level;
  int64_t jlevel = Cj->Level; 
  if (ilevel == jlevel) {
    int admis = admis_check(theta, Ci->C, Cj->C, Ci->R, Cj->R);
    int write_far = NoF == 'F' || NoF == 'f';
    int write_near = NoF == 'N' || NoF == 'n';
    if (admis ? write_far : write_near) {
      int64_t n = *len;
      rels[n] = i + j * ncells;
      *len = n + 1;
    }
    if (admis)
      return;
  }
  if (ilevel <= jlevel && Ci->Child >= 0) {
    getList(NoF, len, rels, ncells, cells, Ci->Child, j, theta);
    getList(NoF, len, rels, ncells, cells, Ci->Child + 1, j, theta);
  }
  else if (jlevel <= ilevel && Cj->Child >= 0) {
    getList(NoF, len, rels, ncells, cells, i, Cj->Child, theta);
    getList(NoF, len, rels, ncells, cells, i, Cj->Child + 1, theta);
  }
}

int comp_int_64(const void *a, const void *b) {
  int64_t c = *(int64_t*)a - *(int64_t*)b;
  return c < 0 ? -1 : (int)(c > 0);
}

void traverse(char NoF, struct CSC* rels, int64_t ncells, const struct Cell* cells, double theta) {
  rels->M = ncells;
  rels->N = ncells;
  int64_t* rel_arr = (int64_t*)malloc(sizeof(int64_t) * (ncells * ncells + ncells + 1));
  int64_t len = 0;
  getList(NoF, &len, &rel_arr[ncells + 1], ncells, cells, 0, 0, theta);

  if (len < ncells * ncells)
    rel_arr = (int64_t*)realloc(rel_arr, sizeof(int64_t) * (len + ncells + 1));
  int64_t* rel_rows = &rel_arr[ncells + 1];
  qsort(rel_rows, len, sizeof(int64_t), comp_int_64);
  rels->ColIndex = rel_arr;
  rels->RowIndex = rel_rows;

  int64_t loc = -1;
  for (int64_t i = 0; i < len; i++) {
    int64_t r = rel_rows[i];
    int64_t x = r / ncells;
    int64_t y = r - x * ncells;
    rel_rows[i] = y;
    while (x > loc)
      rel_arr[++loc] = i;
  }
  for (int64_t i = loc + 1; i <= ncells; i++)
    rel_arr[i] = len;
}

void csc_free(struct CSC* csc) {
  free(csc->ColIndex);
}

void get_level(int64_t* begin, int64_t* end, const struct Cell* cells, int64_t level, int64_t mpi_rank) {
  int64_t low = *begin;
  int64_t high = *end;
  while (low < high) {
    int64_t mid = low + (high - low) / 2;
    const struct Cell* c = &cells[mid];
    int64_t l = c->Level - level;
    int ri = (int)(mpi_rank < c->Procs[0]) - (int)(mpi_rank >= c->Procs[1]);
    int cmp = l < 0 ? -1 : (l > 0 ? 1 : (mpi_rank == -1 ? 0 : ri));
    low = cmp < 0 ? mid + 1 : low;
    high = cmp < 0 ? high : mid;
  }
  *begin = high;

  low = high;
  high = *end;
  while (low < high) {
    int64_t mid = low + (high - low) / 2;
    const struct Cell* c = &cells[mid];
    int64_t l = c->Level - level;
    int ri = (int)(mpi_rank < c->Procs[0]) - (int)(mpi_rank >= c->Procs[1]);
    int cmp = l < 0 ? -1 : (l > 0 ? 1 : (mpi_rank == -1 ? 0 : ri));
    low = cmp <= 0 ? mid + 1 : low;
    high = cmp <= 0 ? high : mid;
  }
  *end = low;
}

void buildComm(struct CellComm* comms, int64_t ncells, const struct Cell* cells, const struct CSC* cellFar, const struct CSC* cellNear, int64_t levels) {
  int __mpi_rank = 0, __mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &__mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &__mpi_size);
  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);
  int64_t mpi_rank = __mpi_rank;
  int64_t mpi_size = __mpi_size;
  int* ranks = (int*)malloc(sizeof(int) * mpi_size);

  for (int64_t i = 0; i <= levels; i++) {
    int64_t ibegin = 0, iend = ncells;
    get_level(&ibegin, &iend, cells, i, -1);
    int64_t len_i = iend - ibegin;

    int64_t nbegin = cellNear->ColIndex[ibegin];
    int64_t nlen = cellNear->ColIndex[iend] - nbegin;
    int64_t fbegin = cellFar->ColIndex[ibegin];
    int64_t flen = cellFar->ColIndex[iend] - fbegin;
    int64_t len_arr = flen + nlen + mpi_size * 4 + 1;

    int64_t* rel_arr = (int64_t*)malloc(sizeof(int64_t) * len_arr);
    int64_t* rel_rows = &rel_arr[mpi_size + 1];

    for (int64_t j = 0; j < len_i; j++) {
      int64_t j_c = ibegin + j;
      int64_t src = cells[j_c].Procs[0];
      int64_t kj_hi = cellFar->ColIndex[j_c + 1];
      for (int64_t kj = cellFar->ColIndex[j_c]; kj < kj_hi; kj++) {
        int64_t k = cellFar->RowIndex[kj];
        int64_t tgt = cells[k].Procs[0];
        int64_t row_i = kj - fbegin;
        rel_rows[row_i] = tgt + src * mpi_size;
      }
    }

    for (int64_t j = 0; j < len_i; j++) {
      int64_t j_c = ibegin + j;
      int64_t src = cells[j_c].Procs[0];
      int64_t kj_hi = cellNear->ColIndex[j_c + 1];
      for (int64_t kj = cellNear->ColIndex[j_c]; kj < kj_hi; kj++) {
        int64_t k = cellNear->RowIndex[kj];
        int64_t tgt = cells[k].Procs[0];
        int64_t row_i = kj - nbegin + flen;
        rel_rows[row_i] = tgt + src * mpi_size;
      }
    }

    struct CSC* csc_i = &comms[i].Comms;
    csc_i->M = mpi_size;
    csc_i->N = mpi_size;
    qsort(rel_rows, nlen + flen, sizeof(int64_t), comp_int_64);
    int64_t* begin = rel_rows;
    int64_t* last = &begin[nlen + flen];
    int64_t* iter = begin;
    if (begin != last) {
      while (++begin != last)
        if (!(*iter == *begin) && ++iter != begin)
          *iter = *begin;
      iter++;
    }

    int64_t len = iter - rel_rows;
    if (len < flen + nlen) {
      len_arr = len + mpi_size * 4 + 1;
      rel_arr = (int64_t*)realloc(rel_arr, sizeof(int64_t) * len_arr);
      rel_rows = &rel_arr[mpi_size + 1];
    }
    csc_i->ColIndex = rel_arr;
    csc_i->RowIndex = rel_rows;
    int64_t* root_i = &rel_arr[len + mpi_size + 1];
    memset(root_i, 0xFF, sizeof(int64_t) * mpi_size);

    int64_t loc = -1;
    for (int64_t j = 0; j < len; j++) {
      int64_t r = rel_rows[j];
      int64_t x = r / mpi_size;
      int64_t y = r - x * mpi_size;
      rel_rows[j] = y;
      while (x > loc)
        rel_arr[++loc] = j;
      if (y == x)
        root_i[x] = j - rel_arr[x];
    }
    for (int64_t j = loc + 1; j <= mpi_size; j++)
      rel_arr[j] = len;

    comms[i].ProcRootI = root_i;
    comms[i].ProcBoxes = &root_i[mpi_size];
    comms[i].ProcBoxesEnd = &root_i[mpi_size * 2];
    for (int64_t j = 0; j < mpi_size; j++) {
      int64_t jbegin = ibegin, jend = iend;
      get_level(&jbegin, &jend, cells, i, j);
      comms[i].ProcBoxes[j] = jbegin - ibegin;
      comms[i].ProcBoxesEnd[j] = jend - ibegin;
    }

    comms[i].Comm_box = (MPI_Comm*)malloc(sizeof(MPI_Comm) * mpi_size);
    for (int64_t j = 0; j < mpi_size; j++) {
      int64_t jbegin = rel_arr[j];
      int64_t jlen = rel_arr[j + 1] - jbegin;
      if (jlen > 0) {
        const int64_t* row = &rel_rows[jbegin];
        for (int64_t k = 0; k < jlen; k++)
          ranks[k] = row[k];
        MPI_Group group_j;
        MPI_Group_incl(world_group, jlen, ranks, &group_j);
        MPI_Comm_create_group(MPI_COMM_WORLD, group_j, j, &comms[i].Comm_box[j]);
        MPI_Group_free(&group_j);
      }
      else
        comms[i].Comm_box[j] = MPI_COMM_NULL;
    }

    int64_t mbegin = ibegin, mend = iend;
    get_level(&mbegin, &mend, cells, i, mpi_rank);
    const struct Cell* cm = &cells[mbegin];
    int64_t p = cm->Procs[0];
    int64_t lenp = cm->Procs[1] - p;
    comms[i].Proc[0] = p;
    comms[i].Proc[1] = p + lenp;
    comms[i].Proc[2] = mpi_rank;
    comms[i].Comm_merge = MPI_COMM_NULL;
    comms[i].Comm_share = MPI_COMM_NULL;

    if (lenp > 1 && cm->Child >= 0) {
      const int64_t lenc = 2;
      int incl = 0;
      for (int64_t j = 0; j < lenc; j++) {
        ranks[j] = cells[cm->Child + j].Procs[0];
        incl = incl || (ranks[j] == mpi_rank);
      }
      if (incl) {
        MPI_Group group_merge;
        MPI_Group_incl(world_group, lenc, ranks, &group_merge);
        MPI_Comm_create_group(MPI_COMM_WORLD, group_merge, mpi_size, &comms[i].Comm_merge);
        MPI_Group_free(&group_merge);
      }
    }

    if (lenp > 1) {
      for (int64_t j = 0; j < lenp; j++)
        ranks[j] = j + p;
      MPI_Group group_share;
      MPI_Group_incl(world_group, lenp, ranks, &group_share);
      MPI_Comm_create_group(MPI_COMM_WORLD, group_share, mpi_size + 1, &comms[i].Comm_share);
      MPI_Group_free(&group_share);
    }
  }

  MPI_Group_free(&world_group);
  free(ranks);
}

void cellComm_free(struct CellComm* comm) {
  int64_t mpi_size = comm->Comms.M;
  for (int64_t j = 0; j < mpi_size; j++)
    if (comm->Comm_box[j] != MPI_COMM_NULL)
      MPI_Comm_free(&comm->Comm_box[j]);
  if (comm->Comm_share != MPI_COMM_NULL)
    MPI_Comm_free(&comm->Comm_share);
  if (comm->Comm_merge != MPI_COMM_NULL)
    MPI_Comm_free(&comm->Comm_merge);
  free(comm->Comms.ColIndex);
  free(comm->Comm_box);
}

void lookupIJ(int64_t* ij, const struct CSC* rels, int64_t i, int64_t j) {
  if (j < 0 || j >= rels->N)
  { *ij = -1; return; }
  const int64_t* row = rels->RowIndex;
  int64_t jbegin = rels->ColIndex[j];
  int64_t jend = rels->ColIndex[j + 1];
  const int64_t* row_iter = &row[jbegin];
  while (row_iter != &row[jend] && *row_iter != i)
    row_iter = row_iter + 1;
  int64_t k = row_iter - row;
  *ij = (k < jend) ? k : -1;
}

void i_local(int64_t* ilocal, const struct CellComm* comm) {
  int64_t iglobal = *ilocal;
  int64_t p = comm->Proc[0];
  const int64_t* ngbs = comm->Comms.RowIndex;
  int64_t nbegin = comm->Comms.ColIndex[p];
  int64_t nend = comm->Comms.ColIndex[p + 1];
  const int64_t* ngbs_iter = &ngbs[nbegin];
  int64_t slen = 0;
  while (ngbs_iter != &ngbs[nend] && 
  comm->ProcBoxesEnd[*ngbs_iter] <= iglobal) {
    slen = slen + comm->ProcBoxesEnd[*ngbs_iter] - comm->ProcBoxes[*ngbs_iter];
    ngbs_iter = ngbs_iter + 1;
  }
  int64_t k = ngbs_iter - ngbs;
  if (k < nend)
    *ilocal = slen + iglobal - comm->ProcBoxes[*ngbs_iter];
  else
    *ilocal = -1;
}

void i_global(int64_t* iglobal, const struct CellComm* comm) {
  int64_t ilocal = *iglobal;
  int64_t p = comm->Proc[0];
  const int64_t* ngbs = comm->Comms.RowIndex;
  int64_t nbegin = comm->Comms.ColIndex[p];
  int64_t nend = comm->Comms.ColIndex[p + 1];
  const int64_t* ngbs_iter = &ngbs[nbegin];
  while (ngbs_iter != &ngbs[nend] && 
  comm->ProcBoxesEnd[*ngbs_iter] <= (comm->ProcBoxes[*ngbs_iter] + ilocal)) {
    ilocal = ilocal - comm->ProcBoxesEnd[*ngbs_iter] + comm->ProcBoxes[*ngbs_iter];
    ngbs_iter = ngbs_iter + 1;
  }
  int64_t k = ngbs_iter - ngbs;
  if (0 <= ilocal && k < nend)
    *iglobal = comm->ProcBoxes[*ngbs_iter] + ilocal;
  else
    *iglobal = -1;
}

void self_local_range(int64_t* ibegin, int64_t* iend, const struct CellComm* comm) {
  int64_t p = comm->Proc[0];
  const int64_t* ngbs = comm->Comms.RowIndex;
  int64_t nbegin = comm->Comms.ColIndex[p];
  int64_t nend = comm->Comms.ColIndex[p + 1];
  const int64_t* ngbs_iter = &ngbs[nbegin];
  int64_t slen = 0;
  while (ngbs_iter != &ngbs[nend] && *ngbs_iter != p) {
    slen = slen + comm->ProcBoxesEnd[*ngbs_iter] - comm->ProcBoxes[*ngbs_iter];
    ngbs_iter = ngbs_iter + 1;
  }
  int64_t k = ngbs_iter - ngbs;
  if (k < nend) {
    *ibegin = slen;
    *iend = slen + comm->ProcBoxesEnd[*ngbs_iter] - comm->ProcBoxes[*ngbs_iter];
  }
  else {
    *ibegin = -1;
    *iend = -1;
  }
}

void content_length(int64_t* len, const struct CellComm* comm) {
  int64_t p = comm->Proc[0];
  const int64_t* ngbs = comm->Comms.RowIndex;
  int64_t nbegin = comm->Comms.ColIndex[p];
  int64_t nend = comm->Comms.ColIndex[p + 1];
  const int64_t* ngbs_iter = &ngbs[nbegin];
  int64_t slen = 0;
  while (ngbs_iter != &ngbs[nend]) {
    slen = slen + comm->ProcBoxesEnd[*ngbs_iter] - comm->ProcBoxes[*ngbs_iter];
    ngbs_iter = ngbs_iter + 1;
  }
  *len = slen;
}

void local_bodies(int64_t body[], int64_t ncells, const struct Cell cells[], int64_t levels) {
  int __mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &__mpi_rank);
  int64_t mpi_rank = __mpi_rank;
  int64_t ibegin = 0, iend = ncells;
  get_level(&ibegin, &iend, cells, levels, mpi_rank);
  body[0] = cells[ibegin].Body[0];
  body[1] = cells[iend - 1].Body[1];
}

void loadX(double* X, int64_t body[], const struct Body* bodies) {
  int64_t ibegin = body[0], iend = body[1];
  int64_t iboxes = iend - ibegin;
  const struct Body* blocal = &bodies[ibegin];
  for (int64_t i = 0; i < iboxes; i++)
    X[i] = blocal[i].B;
}

void relations(struct CSC rels[], int64_t ncells, const struct Cell* cells, const struct CSC* cellRel, int64_t levels) {
  int __mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &__mpi_rank);
  int64_t mpi_rank = __mpi_rank;
  
  for (int64_t i = 0; i <= levels; i++) {
    int64_t jbegin = 0, jend = ncells;
    get_level(&jbegin, &jend, cells, i, -1);
    int64_t ibegin = jbegin, iend = jend;
    get_level(&ibegin, &iend, cells, i, mpi_rank);
    int64_t nodes = iend - ibegin;
    struct CSC* csc = &rels[i];

    csc->M = jend - jbegin;
    csc->N = nodes;
    int64_t ent_max = nodes * csc->M;
    int64_t* cols = (int64_t*)malloc(sizeof(int64_t) * (nodes + 1 + ent_max));
    int64_t* rows = &cols[nodes + 1];

    int64_t count = 0;
    for (int64_t j = 0; j < nodes; j++) {
      int64_t lc = ibegin + j;
      cols[j] = count;
      int64_t cbegin = cellRel->ColIndex[lc];
      int64_t ent = cellRel->ColIndex[lc + 1] - cbegin;
      for (int64_t k = 0; k < ent; k++)
        rows[count + k] = cellRel->RowIndex[cbegin + k] - jbegin;
      count = count + ent;
    }

    if (count < ent_max)
      cols = (int64_t*)realloc(cols, sizeof(int64_t) * (nodes + 1 + count));
    cols[nodes] = count;
    csc->ColIndex = cols;
    csc->RowIndex = &cols[nodes + 1];
  }
}


void evalD(void(*ef)(double*), struct Matrix* D, int64_t ncells, const struct Cell* cells, const struct Body* bodies, const struct CSC* rels, int64_t level) {
  int __mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &__mpi_rank);
  int64_t mpi_rank = __mpi_rank;
  
  int64_t jbegin = 0, jend = ncells;
  get_level(&jbegin, &jend, cells, level, -1);
  int64_t ibegin = jbegin, iend = jend;
  get_level(&ibegin, &iend, cells, level, mpi_rank);
  int64_t nodes = iend - ibegin;

  for (int64_t i = 0; i < nodes; i++) {
    int64_t lc = ibegin + i;
    const struct Cell* ci = &cells[lc];
    int64_t nbegin = rels->ColIndex[i];
    int64_t nlen = rels->ColIndex[i + 1] - nbegin;
    const int64_t* ngbs = &rels->RowIndex[nbegin];
    int64_t x_begin = ci->Body[0];
    int64_t n = ci->Body[1] - x_begin;

    for (int64_t j = 0; j < nlen; j++) {
      int64_t lj = ngbs[j];
      const struct Cell* cj = &cells[lj + jbegin];
      int64_t y_begin = cj->Body[0];
      int64_t m = cj->Body[1] - y_begin;
      gen_matrix(ef, m, n, &bodies[y_begin], &bodies[x_begin], D[nbegin + j].A, NULL, NULL);
    }
  }
}

struct SampleBodies 
{ int64_t LTlen, *FarLens, *FarAvails, **FarBodies, *CloseLens, *CloseAvails, **CloseBodies, *SkeLens, **Skeletons; };

void buildSampleBodies(struct SampleBodies* sample, int64_t sp_max_far, int64_t sp_max_near, int64_t nbodies, int64_t ncells, const struct Cell* cells, 
const struct CSC* rels, const int64_t* lt_child, const struct Base* basis_lo, int64_t level) {
  int __mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &__mpi_rank);
  int64_t mpi_rank = __mpi_rank;
  const int64_t LEN_CHILD = 2;
  
  int64_t jbegin = 0, jend = ncells;
  get_level(&jbegin, &jend, cells, level, -1);
  int64_t ibegin = jbegin, iend = jend;
  get_level(&ibegin, &iend, cells, level, mpi_rank);
  int64_t nodes = iend - ibegin;
  int64_t* arr_ctrl = (int64_t*)malloc(sizeof(int64_t) * nodes * 5);
  int64_t** arr_list = (int64_t**)malloc(sizeof(int64_t*) * nodes * 3);
  sample->LTlen = nodes;
  sample->FarLens = arr_ctrl;
  sample->CloseLens = &arr_ctrl[nodes];
  sample->FarAvails = &arr_ctrl[nodes * 2];
  sample->CloseAvails = &arr_ctrl[nodes * 3];
  sample->SkeLens = &arr_ctrl[nodes * 4];
  sample->FarBodies = arr_list;
  sample->CloseBodies = &arr_list[nodes];
  sample->Skeletons = &arr_list[nodes * 2];

  int64_t count_f = 0, count_c = 0, count_s = 0;
  for (int64_t i = 0; i < nodes; i++) {
    int64_t li = ibegin + i;
    int64_t nbegin = rels->ColIndex[i];
    int64_t nlen = rels->ColIndex[i + 1] - nbegin;
    const int64_t* ngbs = &rels->RowIndex[nbegin];

    int64_t far_avail = nbodies;
    int64_t close_avail = 0;
    for (int64_t j = 0; j < nlen; j++) {
      int64_t lj = ngbs[j] + jbegin;
      const struct Cell* cj = &cells[lj];
      int64_t len = cj->Body[1] - cj->Body[0];
      far_avail = far_avail - len;
      if (lj != li)
        close_avail = close_avail + len;
    }

    int64_t lc = lt_child[i];
    int64_t ske_len = 0;
    if (basis_lo != NULL && lc >= 0)
      for (int64_t j = 0; j < LEN_CHILD; j++)
        ske_len = ske_len + basis_lo->DimsLr[lc + j];
    else
      ske_len = cells[li].Body[1] - cells[li].Body[0];

    int64_t far_len = sp_max_far < far_avail ? sp_max_far : far_avail;
    int64_t close_len = sp_max_near < close_avail ? sp_max_near : close_avail;
    arr_ctrl[i] = far_len;
    arr_ctrl[i + nodes] = close_len;
    arr_ctrl[i + nodes * 2] = far_avail;
    arr_ctrl[i + nodes * 3] = close_avail;
    arr_ctrl[i + nodes * 4] = ske_len;
    count_f = count_f + far_len;
    count_c = count_c + close_len;
    count_s = count_s + ske_len;
  }

  int64_t* arr_bodies = NULL;
  if ((count_f + count_c + count_s) > 0)
    arr_bodies = (int64_t*)malloc(sizeof(int64_t) * (count_f + count_c + count_s));
  const struct Cell* leaves = &cells[jbegin];
  count_s = count_f + count_c;
  count_c = count_f;
  count_f = 0;
  for (int64_t i = 0; i < nodes; i++) {
    int64_t nbegin = rels->ColIndex[i];
    int64_t nlen = rels->ColIndex[i + 1] - nbegin;
    const int64_t* ngbs = &rels->RowIndex[nbegin];

    int64_t* remote = &arr_bodies[count_f];
    int64_t* close = &arr_bodies[count_c];
    int64_t* skeleton = &arr_bodies[count_s];
    int64_t far_len = arr_ctrl[i];
    int64_t close_len = arr_ctrl[i + nodes];
    int64_t far_avail = arr_ctrl[i + nodes * 2];
    int64_t close_avail = arr_ctrl[i + nodes * 3];
    int64_t ske_len = arr_ctrl[i + nodes * 4];

    int64_t box_i = 0;
    int64_t s_lens = 0;
    int64_t ic = ngbs[box_i];
    int64_t offset_i = leaves[ic].Body[0];
    int64_t len_i = leaves[ic].Body[1] - offset_i;

    int64_t li = i + ibegin - jbegin;
    int64_t cpos = 0;
    while (cpos < nlen && ngbs[cpos] != li)
      cpos = cpos + 1;

    for (int64_t j = 0; j < far_len; j++) {
      int64_t loc = (int64_t)((double)(far_avail * j) / far_len);
      while (box_i < nlen && loc + s_lens >= offset_i) {
        s_lens = s_lens + len_i;
        box_i = box_i + 1;
        ic = box_i < nlen ? ngbs[box_i] : ic;
        offset_i = leaves[ic].Body[0];
        len_i = leaves[ic].Body[1] - offset_i;
      }
      remote[j] = loc + s_lens;
    }

    box_i = (int64_t)(cpos == 0);
    s_lens = 0;
    ic = box_i < nlen ? ngbs[box_i] : ic;
    offset_i = leaves[ic].Body[0];
    len_i = leaves[ic].Body[1] - offset_i;

    for (int64_t j = 0; j < close_len; j++) {
      int64_t loc = (int64_t)((double)(close_avail * j) / close_len);
      while (loc - s_lens >= len_i) {
        s_lens = s_lens + len_i;
        box_i = box_i + 1;
        box_i = box_i + (int64_t)(box_i == cpos);
        ic = ngbs[box_i];
        offset_i = leaves[ic].Body[0];
        len_i = leaves[ic].Body[1] - offset_i;
      }
      close[j] = loc + offset_i - s_lens;
    }

    int64_t lc = lt_child[i];
    int64_t sbegin = cells[i + ibegin].Body[0];
    if (basis_lo != NULL && lc >= 0)
      memcpy(skeleton, basis_lo->Multipoles + basis_lo->Offsets[lc], sizeof(int64_t) * ske_len);
    else
      for (int64_t j = 0; j < ske_len; j++)
        skeleton[j] = j + sbegin;

    arr_list[i] = remote;
    arr_list[i + nodes] = close;
    arr_list[i + nodes * 2] = skeleton;
    count_f = count_f + far_len;
    count_c = count_c + close_len;
    count_s = count_s + ske_len;
  }
}

void sampleBodies_free(struct SampleBodies* sample) {
  int64_t* data = sample->FarBodies[0];
  if (data)
    free(data);
  free(sample->FarLens);
  free(sample->FarBodies);
}

void dist_int_64_xlen(int64_t arr_xlen[], const struct CellComm* comm) {
  int64_t mpi_rank = comm->Proc[2];
  int64_t pbegin = comm->Comms.ColIndex[mpi_rank];
  int64_t plen = comm->Comms.ColIndex[mpi_rank + 1] - pbegin;
  const int64_t* row = &comm->Comms.RowIndex[pbegin];
#ifdef _PROF
  double stime = MPI_Wtime();
#endif
  for (int64_t i = 0; i < plen; i++) {
    int64_t p = row[i];
    int64_t lbegin = comm->ProcBoxes[p];
    int64_t len = comm->ProcBoxesEnd[p] - lbegin;
    i_local(&lbegin, comm);
    MPI_Bcast(&arr_xlen[lbegin], len, MPI_INT64_T, comm->ProcRootI[p], comm->Comm_box[p]);
  }

  int64_t xlen = 0;
  content_length(&xlen, comm);
  if (comm->Proc[1] - comm->Proc[0] > 1)
    MPI_Bcast(arr_xlen, xlen, MPI_INT64_T, 0, comm->Comm_share);
#ifdef _PROF
  double etime = MPI_Wtime() - stime;
  recordCommTime(etime);
#endif
}

void dist_int_64(int64_t arr[], const int64_t offsets[], const struct CellComm* comm) {
  int64_t mpi_rank = comm->Proc[2];
  int64_t pbegin = comm->Comms.ColIndex[mpi_rank];
  int64_t plen = comm->Comms.ColIndex[mpi_rank + 1] - pbegin;
  const int64_t* row = &comm->Comms.RowIndex[pbegin];
#ifdef _PROF
  double stime = MPI_Wtime();
#endif
  for (int64_t i = 0; i < plen; i++) {
    int64_t p = row[i];
    int64_t lbegin = comm->ProcBoxes[p];
    int64_t llen = comm->ProcBoxesEnd[p] - lbegin;
    i_local(&lbegin, comm);
    int64_t offset = offsets[lbegin];
    int64_t len = offsets[lbegin + llen] - offset;
    MPI_Bcast(&arr[offset], len, MPI_INT64_T, comm->ProcRootI[p], comm->Comm_box[p]);
  }

  int64_t xlen = 0;
  content_length(&xlen, comm);
  int64_t alen = offsets[xlen];
  if (comm->Proc[1] - comm->Proc[0] > 1)
    MPI_Bcast(arr, alen, MPI_INT64_T, 0, comm->Comm_share);
#ifdef _PROF
  double etime = MPI_Wtime() - stime;
  recordCommTime(etime);
#endif
}

void dist_double(double* arr[], const struct CellComm* comm) {
  int64_t mpi_rank = comm->Proc[2];
  int64_t pbegin = comm->Comms.ColIndex[mpi_rank];
  int64_t plen = comm->Comms.ColIndex[mpi_rank + 1] - pbegin;
  const int64_t* row = &comm->Comms.RowIndex[pbegin];
  double* data = arr[0];
#ifdef _PROF
  double stime = MPI_Wtime();
#endif
  for (int64_t i = 0; i < plen; i++) {
    int64_t p = row[i];
    int64_t lbegin = comm->ProcBoxes[p];
    int64_t llen = comm->ProcBoxesEnd[p] - lbegin;
    i_local(&lbegin, comm);
    int64_t offset = arr[lbegin] - data;
    int64_t len = arr[lbegin + llen] - arr[lbegin];
    MPI_Bcast(&data[offset], len, MPI_DOUBLE, comm->ProcRootI[p], comm->Comm_box[p]);
  }

  int64_t xlen = 0;
  content_length(&xlen, comm);
  int64_t alen = arr[xlen] - data;
  if (comm->Proc[1] - comm->Proc[0] > 1)
    MPI_Bcast(data, alen, MPI_DOUBLE, 0, comm->Comm_share);
#ifdef _PROF
  double etime = MPI_Wtime() - stime;
  recordCommTime(etime);
#endif
}

void buildBasis(void(*ef)(double*), struct Base basis[], int64_t ncells, struct Cell* cells, const struct CSC* rel_near, int64_t levels, 
const struct CellComm* comm, const struct Body* bodies, int64_t nbodies, double epi, int64_t mrank, int64_t sp_pts) {

  for (int64_t l = levels; l >= 0; l--) {
    int64_t xlen = 0;
    content_length(&xlen, &comm[l]);
    basis[l].Ulen = xlen;
    int64_t* arr_i = (int64_t*)malloc(sizeof(int64_t) * (xlen * 4 + 1));
    basis[l].Lchild = arr_i;
    basis[l].Dims = &arr_i[xlen];
    basis[l].DimsLr = &arr_i[xlen * 2];
    basis[l].Offsets = &arr_i[xlen * 3];
    basis[l].Multipoles = NULL;
    int64_t jbegin = 0, jend = ncells;
    get_level(&jbegin, &jend, cells, l, -1);
    for (int64_t j = 0; j < xlen; j++) {
      int64_t gj = j;
      i_global(&gj, &comm[l]);
      int64_t lc = cells[jbegin + gj].Child;
      arr_i[j] = -1;
      if (lc >= 0) {
        arr_i[j] = lc - jend;
        i_local(&arr_i[j], &comm[l + 1]);
      }
    }

    struct Matrix* arr_m = (struct Matrix*)calloc(xlen * 3, sizeof(struct Matrix));
    basis[l].Uo = arr_m;
    basis[l].Uc = &arr_m[xlen];
    basis[l].R = &arr_m[xlen * 2];

    int64_t ibegin = 0, iend = xlen;
    self_local_range(&ibegin, &iend, &comm[l]);
    int64_t nodes = iend - ibegin;

    struct SampleBodies samples;
    buildSampleBodies(&samples, sp_pts, sp_pts, nbodies, ncells, cells, &rel_near[l], &basis[l].Lchild[ibegin], l == levels ? NULL : &basis[l + 1], l);

    int64_t count = 0;
    int64_t count_m = 0;
    for (int64_t i = 0; i < nodes; i++) {
      int64_t ske_len = samples.SkeLens[i];
      int64_t len_m = samples.FarLens[i] < samples.CloseLens[i] ? samples.CloseLens[i] : samples.FarLens[i];
      basis[l].Dims[i + ibegin] = ske_len;
      count = count + ske_len;
      count_m = count_m + ske_len * (ske_len * 2 + len_m + 2);
    }

    int32_t* ipiv_data = (int32_t*)malloc(sizeof(int32_t) * count);
    int32_t** ipiv_ptrs = (int32_t**)malloc(sizeof(int32_t*) * nodes);
    double* matrix_data = (double*)malloc(sizeof(double) * count_m);
    double** matrix_ptrs = (double**)malloc(sizeof(double*) * (xlen + 1));

    count = 0;
    count_m = 0;
    for (int64_t i = 0; i < nodes; i++) {
      int64_t ske_len = samples.SkeLens[i];
      int64_t len_m = samples.FarLens[i] < samples.CloseLens[i] ? samples.CloseLens[i] : samples.FarLens[i];
      ipiv_ptrs[i] = &ipiv_data[count];
      matrix_ptrs[i + ibegin] = &matrix_data[count_m];
      count = count + ske_len;
      count_m = count_m + ske_len * (ske_len * 2 + len_m + 2);
    }

#pragma omp parallel for
    for (int64_t i = 0; i < nodes; i++) {
      int64_t ske_len = samples.SkeLens[i];
      int64_t len_s = samples.FarLens[i] + (samples.CloseLens[i] > 0 ? ske_len : 0);
      double* mat = matrix_ptrs[i + ibegin];
      struct Matrix S = (struct Matrix){ &mat[ske_len * ske_len], ske_len, len_s };

      if (len_s > 0) {
        struct Matrix S_dn = (struct Matrix){ &mat[ske_len * ske_len], ske_len, ske_len };
        double nrm_dn = 0.;
        double nrm_lr = 0.;
        if (samples.CloseLens[i] > 0) {
          struct Matrix S_dn_work = (struct Matrix){ &mat[ske_len * ske_len * 2], ske_len, samples.CloseLens[i] };;
          gen_matrix(ef, ske_len, samples.CloseLens[i], bodies, bodies, S_dn_work.A, samples.Skeletons[i], samples.CloseBodies[i]);
          mmult('N', 'T', &S_dn_work, &S_dn_work, &S_dn, 1., 0.);
          nrm2_A(&S_dn, &nrm_dn);
        }

        if (samples.FarLens[i] > 0) {
          struct Matrix S_lr = (struct Matrix){ &mat[ske_len * ske_len * 2], ske_len, samples.FarLens[i] };
          gen_matrix(ef, ske_len, samples.FarLens[i], bodies, bodies, S_lr.A, samples.Skeletons[i], samples.FarBodies[i]);
          nrm2_A(&S_lr, &nrm_lr);
          if (samples.CloseLens[i] > 0)
            scal_A(&S_dn, nrm_lr / nrm_dn);
        }
      }

      int64_t rank = ske_len < len_s ? ske_len : len_s;
      rank = mrank > 0 ? (mrank < rank ? mrank : rank) : rank;
      if (rank > 0) {
        struct Matrix Q = (struct Matrix){ mat, ske_len, ske_len };
        double* Svec = &mat[ske_len * (ske_len + len_s)];
        int32_t* pa = ipiv_ptrs[i];
        svd_U(&S, &Q, Svec);

        if (epi > 0.) {
          int64_t r = 0;
          double sepi = Svec[0] * epi;
          while(r < rank && Svec[r] > sepi)
            r += 1;
          rank = r;
        }

        struct Matrix Qo = (struct Matrix){ mat, ske_len, rank };
        struct Matrix R = (struct Matrix){ &mat[ske_len * ske_len], rank, rank };
        mul_AS(&Qo, Svec);
        id_row(&Qo, pa, S.A);

        int64_t lc = basis[l].Lchild[i + ibegin];
        if (lc >= 0)
          updateSubU(&Qo, &(basis[l + 1].R)[lc], &(basis[l + 1].R)[lc + 1]);
        qr_full(&Q, &R, Svec);

        for (int64_t j = 0; j < rank; j++) {
          int64_t piv = (int64_t)pa[j] - 1;
          if (piv != j) { 
            int64_t c = samples.Skeletons[i][piv];
            samples.Skeletons[i][piv] = samples.Skeletons[i][j];
            samples.Skeletons[i][j] = c;
          }
        }
      }

      basis[l].DimsLr[i + ibegin] = rank;
    }
    dist_int_64_xlen(basis[l].Dims, &comm[l]);
    dist_int_64_xlen(basis[l].DimsLr, &comm[l]);

    count = 0;
    count_m = 0;
    for (int64_t i = 0; i < xlen; i++) {
      int64_t m = basis[l].Dims[i];
      int64_t n = basis[l].DimsLr[i];
      basis[l].Offsets[i] = count;
      count = count + n;
      count_m = count_m + m * m + n * n;
    }
    basis[l].Offsets[xlen] = count;

    if (count > 0)
      basis[l].Multipoles = (int64_t*)malloc(sizeof(int64_t) * count);
    for (int64_t i = 0; i < nodes; i++) {
      int64_t offset = basis[l].Offsets[i + ibegin];
      int64_t n = basis[l].DimsLr[i + ibegin];
      if (n > 0)
        memcpy(&basis[l].Multipoles[offset], samples.Skeletons[i], sizeof(int64_t) * n);
    }
    dist_int_64(basis[l].Multipoles, basis[l].Offsets, &comm[l]);

    double* data_basis = NULL;
    if (count_m > 0)
      data_basis = (double*)malloc(sizeof(int64_t) * count_m);
    for (int64_t i = 0; i < xlen; i++) {
      int64_t m = basis[l].Dims[i];
      int64_t n = basis[l].DimsLr[i];
      int64_t size = m * m + n * n;
      if (ibegin <= i && i < iend && size > 0)
        memcpy(data_basis, matrix_ptrs[i], sizeof(double) * size);
      basis[l].Uo[i] = (struct Matrix){ data_basis, m, n };
      basis[l].Uc[i] = (struct Matrix){ &data_basis[m * n], m, m - n };
      basis[l].R[i] = (struct Matrix){ &data_basis[m * m], n, n };
      matrix_ptrs[i] = data_basis;
      data_basis = &data_basis[size];
    }
    matrix_ptrs[xlen] = data_basis;
    dist_double(matrix_ptrs, &comm[l]);

    free(ipiv_data);
    free(ipiv_ptrs);
    free(matrix_data);
    free(matrix_ptrs);
    sampleBodies_free(&samples);
  }
}

void basis_free(struct Base* basis) {
  double* data = basis->Uo[0].A;
  if (data)
    free(data);
  if (basis->Multipoles)
    free(basis->Multipoles);
  free(basis->Lchild);
  free(basis->Uo);
}

void evalS(void(*ef)(double*), struct Matrix* S, const struct Base* basis, const struct Body* bodies, const struct CSC* rels, const struct CellComm* comm) {
  int64_t ibegin = 0, iend = 0;
  self_local_range(&ibegin, &iend, comm);
  int64_t lbegin = ibegin;
  i_global(&lbegin, comm);

  for (int64_t x = 0; x < rels->N; x++) {
    for (int64_t yx = rels->ColIndex[x]; yx < rels->ColIndex[x + 1]; yx++) {
      int64_t y = rels->RowIndex[yx];
      int64_t box_y = y;
      i_local(&box_y, comm);
      int64_t m = basis->DimsLr[box_y];
      int64_t n = basis->DimsLr[x + ibegin];
      int64_t* multipoles = basis->Multipoles;
      int64_t off_y = basis->Offsets[box_y];
      int64_t off_x = basis->Offsets[x + ibegin];
      gen_matrix(ef, m, n, bodies, bodies, S[yx].A, &multipoles[off_y], &multipoles[off_x]);
      rsr(&basis->R[box_y], &basis->R[x + ibegin], &S[yx]);
    }
  }
}

void allocNodes(struct Node A[], const struct Base basis[], const struct CSC rels_near[], const struct CSC rels_far[], const struct CellComm comm[], int64_t levels) {
  for (int64_t i = 0; i <= levels; i++) {
    int64_t n_i = rels_near[i].N;
    int64_t nnz = rels_near[i].ColIndex[n_i];
    int64_t nnz_f = rels_far[i].ColIndex[n_i];
    int64_t len_arr = nnz * 4 + nnz_f;
    struct Matrix* arr_m = (struct Matrix*)malloc(sizeof(struct Matrix) * len_arr);
    A[i].A = arr_m;
    A[i].A_cc = &arr_m[nnz];
    A[i].A_oc = &arr_m[nnz * 2];
    A[i].A_oo = &arr_m[nnz * 3];
    A[i].S = &arr_m[nnz * 4];
    A[i].lenA = nnz;
    A[i].lenS = nnz_f;

    int64_t ibegin = 0, iend = 0;
    self_local_range(&ibegin, &iend, &comm[i]);

    int64_t count = 0;
    for (int64_t x = 0; x < rels_near[i].N; x++) {
      int64_t box_x = ibegin + x;
      int64_t dim_x = basis[i].Dims[box_x];
      int64_t diml_x = basis[i].DimsLr[box_x];
      int64_t dimc_x = dim_x - diml_x;

      for (int64_t yx = rels_near[i].ColIndex[x]; yx < rels_near[i].ColIndex[x + 1]; yx++) {
        int64_t y = rels_near[i].RowIndex[yx];
        int64_t box_y = y;
        i_local(&box_y, &comm[i]);
        int64_t dim_y = basis[i].Dims[box_y];
        int64_t diml_y = basis[i].DimsLr[box_y];
        int64_t dimc_y = dim_y - diml_y;
        arr_m[yx].M = dim_y; // A
        arr_m[yx].N = dim_x;
        arr_m[yx + nnz].M = dimc_y; // A_cc
        arr_m[yx + nnz].N = dimc_x;
        arr_m[yx + nnz * 2].M = diml_y; // A_oc
        arr_m[yx + nnz * 2].N = dimc_x;
        arr_m[yx + nnz * 3].M = diml_y; // A_oo
        arr_m[yx + nnz * 3].N = diml_x;
        count += dim_y * dim_x;
      }

      for (int64_t yx = rels_far[i].ColIndex[x]; yx < rels_far[i].ColIndex[x + 1]; yx++) {
        int64_t y = rels_far[i].RowIndex[yx];
        int64_t box_y = y;
        i_local(&box_y, &comm[i]);
        int64_t diml_y = basis[i].DimsLr[box_y];
        arr_m[yx + nnz * 4].M = diml_y; // S
        arr_m[yx + nnz * 4].N = diml_x;
        count += diml_y * diml_x;
      }
    }

    double* data = NULL;
    if (count > 0)
      data = (double*)calloc(count, sizeof(double));
    
    for (int64_t x = 0; x < nnz; x++) {
      int64_t dim_y = arr_m[x].M;
      int64_t dim_x = arr_m[x].N;
      int64_t dimc_y = arr_m[x + nnz].M;
      int64_t dimc_x = arr_m[x + nnz].N;
      arr_m[x].A = data; // A
      arr_m[x + nnz].A = data; // A_cc
      arr_m[x + nnz * 2].A = data + dimc_y * dimc_x; // A_oc
      arr_m[x + nnz * 3].A = data + dim_y * dimc_x; // A_oo
      data = data + dim_y * dim_x;
    }

    for (int64_t x = 0; x < nnz_f; x++) {
      int64_t y = x + nnz * 4;
      arr_m[y].A = data;
      data = data + arr_m[y].M * arr_m[y].N;
    }
  }
}

void node_free(struct Node* node) {
  double* data = node->A[0].A;
  if (data)
    free(data);
  free(node->A);
}

void factorNode(struct Matrix* A_cc, struct Matrix* A_oc, struct Matrix* A_oo, const struct Matrix* A, const struct Matrix* Uc, const struct Matrix* Uo, const struct CSC* rels, const struct CellComm* comm) {
  int64_t nnz = rels->ColIndex[rels->N];
  int64_t alen = (int64_t)(A[nnz - 1].A - A[0].A) + A[nnz - 1].M * A[nnz - 1].N;
  double* data = (double*)malloc(sizeof(double) * alen);
  struct Matrix* AV_c = (struct Matrix*)malloc(sizeof(struct Matrix) * nnz * 2);
  struct Matrix* AV_o = &AV_c[nnz];

  for (int64_t x = 0; x < nnz; x++) {
    int64_t dim_y = A[x].M;
    int64_t dim_x = A[x].N;
    int64_t dimc_x = A_cc[x].N;
    int64_t diml_x = dim_x - dimc_x;
    AV_c[x] = (struct Matrix){ data, dim_y, dimc_x };
    AV_o[x] = (struct Matrix){ &data[dim_y * dimc_x], dim_y, diml_x };
    data = data + dim_y * dim_x;
  }
  
  int64_t ibegin = 0, iend = 0;
  self_local_range(&ibegin, &iend, comm);
  int64_t lbegin = ibegin;
  i_global(&lbegin, comm);

#pragma omp parallel for
  for (int64_t x = 0; x < rels->N; x++) {
    for (int64_t yx = rels->ColIndex[x]; yx < rels->ColIndex[x + 1]; yx++) {
      int64_t y = rels->RowIndex[yx];
      i_local(&y, comm);
      mmult('N', 'N', &A[yx], &Uc[x + ibegin], &AV_c[yx], 1., 0.);
      mmult('N', 'N', &A[yx], &Uo[x + ibegin], &AV_o[yx], 1., 0.);
      mmult('T', 'N', &Uc[y], &AV_c[yx], &A_cc[yx], 1., 0.);
      mmult('T', 'N', &Uo[y], &AV_c[yx], &A_oc[yx], 1., 0.);
      mmult('T', 'N', &Uo[y], &AV_o[yx], &A_oo[yx], 1., 0.);
    }  // Skeleton and Redundancy decomposition

    int64_t xx;
    lookupIJ(&xx, rels, x + lbegin, x);
    chol_decomp(&A_cc[xx]); // Factorization

    for (int64_t yx = rels->ColIndex[x]; yx < rels->ColIndex[x + 1]; yx++) {
      int64_t y = rels->RowIndex[yx];
      trsm_lowerA(&A_oc[yx], &A_cc[xx]);
      if (y > x + lbegin)
        trsm_lowerA(&A_cc[yx], &A_cc[xx]);
    } // Lower elimination
    mmult('N', 'T', &A_oc[xx], &A_oc[xx], &A_oo[xx], -1., 1.); // Schur Complement
  }

  free(AV_c[0].A);
  free(AV_c);
}

void nextNode(struct Matrix* Mup, const struct CSC* rels_up, const struct Matrix* Mlow, const struct Matrix* Slow, const int64_t* lchild, 
const struct CSC* rels_low_near, const struct CSC* rels_low_far, const struct CellComm* comm_up, const struct CellComm* comm_low) {
  int64_t nloc = 0, nend = 0, ploc = 0, pend = 0;
  self_local_range(&nloc, &nend, comm_up);
  self_local_range(&ploc, &pend, comm_low);

  for (int64_t j = 0; j < rels_up->N; j++) {
    int64_t cj0 = lchild[j + nloc] - ploc;
    int64_t cj1 = cj0 + 1;

    for (int64_t ij = rels_up->ColIndex[j]; ij < rels_up->ColIndex[j + 1]; ij++) {
      int64_t li = rels_up->RowIndex[ij];
      i_local(&li, comm_up);
      int64_t ci0 = lchild[li];
      i_global(&ci0, comm_low);
      int64_t ci1 = ci0 + 1;

      int64_t i00, i01, i10, i11;
      lookupIJ(&i00, rels_low_near, ci0, cj0);
      lookupIJ(&i01, rels_low_near, ci0, cj1);
      lookupIJ(&i10, rels_low_near, ci1, cj0);
      lookupIJ(&i11, rels_low_near, ci1, cj1);

      if (i00 >= 0)
        cpyMatToMat(Mlow[i00].M, Mlow[i00].N, &Mlow[i00], &Mup[ij], 0, 0, 0, 0);
      if (i01 >= 0)
        cpyMatToMat(Mlow[i01].M, Mlow[i01].N, &Mlow[i01], &Mup[ij], 0, 0, 0, Mup[ij].N - Mlow[i01].N);
      if (i10 >= 0)
        cpyMatToMat(Mlow[i10].M, Mlow[i10].N, &Mlow[i10], &Mup[ij], 0, 0, Mup[ij].M - Mlow[i10].M, 0);
      if (i11 >= 0)
        cpyMatToMat(Mlow[i11].M, Mlow[i11].N, &Mlow[i11], &Mup[ij], 0, 0, Mup[ij].M - Mlow[i11].M, Mup[ij].N - Mlow[i11].N);

      lookupIJ(&i00, rels_low_far, ci0, cj0);
      lookupIJ(&i01, rels_low_far, ci0, cj1);
      lookupIJ(&i10, rels_low_far, ci1, cj0);
      lookupIJ(&i11, rels_low_far, ci1, cj1);

      if (i00 >= 0)
        cpyMatToMat(Slow[i00].M, Slow[i00].N, &Slow[i00], &Mup[ij], 0, 0, 0, 0);
      if (i01 >= 0)
        cpyMatToMat(Slow[i01].M, Slow[i01].N, &Slow[i01], &Mup[ij], 0, 0, 0, Mup[ij].N - Slow[i01].N);
      if (i10 >= 0)
        cpyMatToMat(Slow[i10].M, Slow[i10].N, &Slow[i10], &Mup[ij], 0, 0, Mup[ij].M - Slow[i10].M, 0);
      if (i11 >= 0)
        cpyMatToMat(Slow[i11].M, Slow[i11].N, &Slow[i11], &Mup[ij], 0, 0, Mup[ij].M - Slow[i11].M, Mup[ij].N - Slow[i11].N);
    }
  }
}

void merge_double(double* arr, int64_t alen, const struct CellComm* comm) {
#ifdef _PROF
  double stime = MPI_Wtime();
#endif
  if (comm->Comm_merge != MPI_COMM_NULL)
    MPI_Allreduce(MPI_IN_PLACE, arr, alen, MPI_DOUBLE, MPI_SUM, comm->Comm_merge);
  if (comm->Proc[1] - comm->Proc[0] > 1)
    MPI_Bcast(arr, alen, MPI_DOUBLE, 0, comm->Comm_share);
#ifdef _PROF
  double etime = MPI_Wtime() - stime;
  recordCommTime(etime);
#endif
}

void factorA(struct Node A[], const struct Base basis[], const struct CSC rels_near[], const struct CSC rels_far[], const struct CellComm comm[], int64_t levels) {
  for (int64_t i = levels; i > 0; i--) {
    factorNode(A[i].A_cc, A[i].A_oc, A[i].A_oo, A[i].A, basis[i].Uc, basis[i].Uo, &rels_near[i], &comm[i]);
    int64_t inxt = i - 1;
    nextNode(A[inxt].A, &rels_near[inxt], A[i].A_oo, A[i].S, basis[inxt].Lchild, &rels_near[i], &rels_far[i], &comm[inxt], &comm[i]);

    int64_t alst = rels_near[inxt].ColIndex[rels_near[inxt].N] - 1;
    int64_t alen = (int64_t)(A[inxt].A[alst].A - A[inxt].A[0].A) + A[inxt].A[alst].M * A[inxt].A[alst].N;
    merge_double(A[inxt].A[0].A, alen, &comm[inxt]);
  }
  chol_decomp(&A[0].A[0]);
}

void allocRightHandSides(char mvsv, struct RightHandSides rhs[], const struct Base base[], int64_t levels) {
  int use_c = (mvsv == 'S') || (mvsv == 's');
  for (int64_t i = 0; i <= levels; i++) {
    int64_t len = base[i].Ulen;
    int64_t len_arr = len * 4;
    struct Matrix* arr_m = (struct Matrix*)malloc(sizeof(struct Matrix) * len_arr);
    rhs[i].Xlen = len;
    rhs[i].X = arr_m;
    rhs[i].XcM = &arr_m[len];
    rhs[i].XoL = &arr_m[len * 2];
    rhs[i].B = &arr_m[len * 3];

    int64_t count = 0;
    for (int64_t j = 0; j < len; j++) {
      int64_t dim = base[i].Dims[j];
      int64_t diml = base[i].DimsLr[j];
      int64_t dimc = use_c ? dim - diml : diml;
      arr_m[j].M = dim; // X
      arr_m[j].N = 1;
      arr_m[j + len].M = dimc; // Xc
      arr_m[j + len].N = 1;
      arr_m[j + len * 2].M = diml; // Xo
      arr_m[j + len * 2].N = 1;
      arr_m[j + len * 3].M = dim; // B
      arr_m[j + len * 3].N = 1;
      count = count + dim * 2 + dimc + diml;
    }

    double* data = NULL;
    if (count > 0)
      data = (double*)calloc(count, sizeof(double));
    
    for (int64_t j = 0; j < len_arr; j++) {
      arr_m[j].A = data;
      int64_t len = arr_m[j].M;
      data = data + len;
    }
  }
}

void rightHandSides_free(struct RightHandSides* rhs) {
  double* data = rhs->X[0].A;
  if (data)
    free(data);
  free(rhs->X);
}

void svAccFw(struct Matrix* Xc, struct Matrix* Xo, const struct Matrix* X, const struct Matrix* Uc, const struct Matrix* Uo, const struct Matrix* A_cc, const struct Matrix* A_oc, const struct CSC* rels, const struct CellComm* comm) {
  int64_t ibegin = 0, iend = 0;
  self_local_range(&ibegin, &iend, comm);
  int64_t lbegin = ibegin;
  i_global(&lbegin, comm);

  for (int64_t x = 0; x < rels->N; x++) {
    mmult('T', 'N', &Uc[x + ibegin], &X[x + ibegin], &Xc[x + ibegin], 1., 1.);
    mmult('T', 'N', &Uo[x + ibegin], &X[x + ibegin], &Xo[x + ibegin], 1., 1.);
    int64_t xx;
    lookupIJ(&xx, rels, x + lbegin, x);
    mat_solve('F', &Xc[x + ibegin], &A_cc[xx]);

    for (int64_t yx = rels->ColIndex[x]; yx < rels->ColIndex[x + 1]; yx++) {
      int64_t y = rels->RowIndex[yx];
      int64_t box_y = y;
      i_local(&box_y, comm);
      if (y > x + lbegin)
        mmult('N', 'N', &A_cc[yx], &Xc[x + ibegin], &Xc[box_y], -1., 1.);
      mmult('N', 'N', &A_oc[yx], &Xc[x + ibegin], &Xo[box_y], -1., 1.);
    }
  }
}

void svAccBk(struct Matrix* Xc, const struct Matrix* Xo, struct Matrix* X, const struct Matrix* Uc, const struct Matrix* Uo, const struct Matrix* A_cc, const struct Matrix* A_oc, const struct CSC* rels, const struct CellComm* comm) {
  int64_t ibegin = 0, iend = 0;
  self_local_range(&ibegin, &iend, comm);
  int64_t lbegin = ibegin;
  i_global(&lbegin, comm);

  for (int64_t x = rels->N - 1; x >= 0; x--) {
    for (int64_t yx = rels->ColIndex[x]; yx < rels->ColIndex[x + 1]; yx++) {
      int64_t y = rels->RowIndex[yx];
      int64_t box_y = y;
      i_local(&box_y, comm);
      mmult('T', 'N', &A_oc[yx], &Xo[box_y], &Xc[x + ibegin], -1., 1.);
      if (y > x + lbegin)
        mmult('T', 'N', &A_cc[yx], &Xc[box_y], &Xc[x + ibegin], -1., 1.);
    }

    int64_t xx;
    lookupIJ(&xx, rels, x + lbegin, x);
    mat_solve('B', &Xc[x + ibegin], &A_cc[xx]);
    mmult('N', 'N', &Uc[x + ibegin], &Xc[x + ibegin], &X[x + ibegin], 1., 0.);
    mmult('N', 'N', &Uo[x + ibegin], &Xo[x + ibegin], &X[x + ibegin], 1., 1.);
  }
}

void permuteAndMerge(char fwbk, struct Matrix* px, struct Matrix* nx, const int64_t* lchild, const struct CellComm* comm) {
  int64_t nloc = 0, nend = 0;
  self_local_range(&nloc, &nend, comm);
  int64_t nboxes = nend - nloc;

  if (fwbk == 'F' || fwbk == 'f')
    for (int64_t i = 0; i < nboxes; i++) {
      int64_t c = i + nloc;
      int64_t c0 = lchild[c];
      int64_t c1 = c0 + 1;
      cpyMatToMat(px[c0].M, 1, &px[c0], &nx[c], 0, 0, 0, 0);
      cpyMatToMat(px[c1].M, 1, &px[c1], &nx[c], 0, 0, nx[c].M - px[c1].M, 0);
    }
  else if (fwbk == 'B' || fwbk == 'b')
    for (int64_t i = 0; i < nboxes; i++) {
      int64_t c = i + nloc;
      int64_t c0 = lchild[c];
      int64_t c1 = c0 + 1;
      cpyMatToMat(px[c0].M, 1, &nx[c], &px[c0], 0, 0, 0, 0);
      cpyMatToMat(px[c1].M, 1, &nx[c], &px[c1], nx[c].M - px[c1].M, 0, 0, 0);
    }
}

void dist_double_svfw(char fwbk, double* arr[], const struct CellComm* comm) {
  int64_t mpi_rank = comm->Proc[2];
  int64_t pbegin = comm->Comms.ColIndex[mpi_rank];
  int64_t plen = comm->Comms.ColIndex[mpi_rank + 1] - pbegin;
  const int64_t* row = &comm->Comms.RowIndex[pbegin];
  double* data = arr[0];
  int is_all = fwbk == 'A' || fwbk == 'a';
#ifdef _PROF
  double stime = MPI_Wtime();
#endif
  for (int64_t i = 0; i < plen; i++) {
    int64_t p = row[i];
    int is_fw = (fwbk == 'F' || fwbk == 'f') && p <= mpi_rank;
    int is_bk = (fwbk == 'B' || fwbk == 'b') && mpi_rank < p;
    if (is_all || is_fw || is_bk) {
      int64_t lbegin = comm->ProcBoxes[p];
      int64_t llen = comm->ProcBoxesEnd[p] - lbegin;
      i_local(&lbegin, comm);
      int64_t offset = arr[lbegin] - data;
      int64_t len = arr[lbegin + llen] - arr[lbegin];
      MPI_Allreduce(MPI_IN_PLACE, &data[offset], len, MPI_DOUBLE, MPI_SUM, comm->Comm_box[p]);
    }
  }

  int64_t xlen = 0;
  content_length(&xlen, comm);
  int64_t alen = arr[xlen] - data;
  if (comm->Proc[1] - comm->Proc[0] > 1)
    MPI_Bcast(data, alen, MPI_DOUBLE, 0, comm->Comm_share);
#ifdef _PROF
  double etime = MPI_Wtime() - stime;
  recordCommTime(etime);
#endif
}

void dist_double_svbk(char fwbk, double* arr[], const struct CellComm* comm) {
  int64_t mpi_rank = comm->Proc[2];
  int64_t pbegin = comm->Comms.ColIndex[mpi_rank];
  int64_t plen = comm->Comms.ColIndex[mpi_rank + 1] - pbegin;
  const int64_t* row = &comm->Comms.RowIndex[pbegin];
  double* data = arr[0];
  int is_all = fwbk == 'A' || fwbk == 'a';
#ifdef _PROF
  double stime = MPI_Wtime();
#endif
  for (int64_t i = plen - 1; i >= 0; i--) {
    int64_t p = row[i];
    int is_fw = (fwbk == 'F' || fwbk == 'f') && p <= mpi_rank;
    int is_bk = (fwbk == 'B' || fwbk == 'b') && mpi_rank < p;
    if (is_all || is_fw || is_bk) {
      int64_t lbegin = comm->ProcBoxes[p];
      int64_t llen = comm->ProcBoxesEnd[p] - lbegin;
      i_local(&lbegin, comm);
      int64_t offset = arr[lbegin] - data;
      int64_t len = arr[lbegin + llen] - arr[lbegin];
      MPI_Bcast(&data[offset], len, MPI_DOUBLE, comm->ProcRootI[p], comm->Comm_box[p]);
    }
  }

  int64_t xlen = 0;
  content_length(&xlen, comm);
  int64_t alen = arr[xlen] - data;
  if (comm->Proc[1] - comm->Proc[0] > 1)
    MPI_Bcast(data, alen, MPI_DOUBLE, 0, comm->Comm_share);
#ifdef _PROF
  double etime = MPI_Wtime() - stime;
  recordCommTime(etime);
#endif
}

void solveA(struct RightHandSides rhs[], const struct Node A[], const struct Base basis[], const struct CSC rels[], double* X, const struct CellComm comm[], int64_t levels) {
  int64_t ibegin = 0, iend = 0;
  self_local_range(&ibegin, &iend, &comm[levels]);
  int64_t lenX = (rhs[levels].X[iend - 1].A - rhs[levels].X[ibegin].A) + rhs[levels].X[iend - 1].M;
  memcpy(rhs[levels].X[ibegin].A, X, lenX * sizeof(double));

  for (int64_t i = levels; i > 0; i--) {
    int64_t xlen = rhs[i].Xlen;
    double** arr_comm = (double**)malloc(sizeof(double*) * (xlen + 1));
    for (int64_t j = 0; j < xlen; j++)
      arr_comm[j] = rhs[i].XcM[j].A;
    arr_comm[xlen] = arr_comm[xlen - 1] + rhs[i].XcM[xlen - 1].M;
    dist_double_svfw('F', arr_comm, &comm[i]);

    svAccFw(rhs[i].XcM, rhs[i].XoL, rhs[i].X, basis[i].Uc, basis[i].Uo, A[i].A_cc, A[i].A_oc, &rels[i], &comm[i]);
    dist_double_svfw('B', arr_comm, &comm[i]);

    for (int64_t j = 0; j < xlen; j++)
      arr_comm[j] = rhs[i].XoL[j].A;
    arr_comm[xlen] = arr_comm[xlen - 1] + rhs[i].XoL[xlen - 1].M;
    dist_double_svfw('A', arr_comm, &comm[i]);

    free(arr_comm);
    permuteAndMerge('F', rhs[i].XoL, rhs[i - 1].X, basis[i - 1].Lchild, &comm[i - 1]);
  }
  cpyMatToMat(rhs[0].X[0].M, 1, &rhs[0].X[0], &rhs[0].B[0], 0, 0, 0, 0);
  mat_solve('A', &rhs[0].B[0], &A[0].A[0]);
  
  for (int64_t i = 1; i <= levels; i++) {
    permuteAndMerge('B', rhs[i].XoL, rhs[i - 1].B, basis[i - 1].Lchild, &comm[i - 1]);
    int64_t xlen = rhs[i].Xlen;
    double** arr_comm = (double**)malloc(sizeof(double*) * (xlen + 1));
    for (int64_t j = 0; j < xlen; j++)
      arr_comm[j] = rhs[i].XoL[j].A;
    arr_comm[xlen] = arr_comm[xlen - 1] + rhs[i].XoL[xlen - 1].M;
    dist_double_svbk('A', arr_comm, &comm[i]);

    for (int64_t j = 0; j < xlen; j++)
      arr_comm[j] = rhs[i].XcM[j].A;
    arr_comm[xlen] = arr_comm[xlen - 1] + rhs[i].XcM[xlen - 1].M;
    dist_double_svbk('B', arr_comm, &comm[i]);
    
    svAccBk(rhs[i].XcM, rhs[i].XoL, rhs[i].B, basis[i].Uc, basis[i].Uo, A[i].A_cc, A[i].A_oc, &rels[i], &comm[i]);
    dist_double_svbk('F', arr_comm, &comm[i]);
    free(arr_comm);
  }
  memcpy(X, rhs[levels].B[ibegin].A, lenX * sizeof(double));
}

void horizontalPass(struct Matrix* B, const struct Matrix* X, const struct Matrix* A, const struct CSC* rels, const struct CellComm* comm) {
  int64_t ibegin = 0, iend = 0;
  self_local_range(&ibegin, &iend, comm);
  for (int64_t y = 0; y < rels->N; y++)
    for (int64_t xy = rels->ColIndex[y]; xy < rels->ColIndex[y + 1]; xy++) {
      int64_t x = rels->RowIndex[xy];
      i_local(&x, comm);
      mmult('T', 'N', &A[xy], &X[x], &B[y + ibegin], 1., 1.);
    }
}

void matVecA(struct RightHandSides rhs[], const struct Node A[], const struct Base basis[], const struct CSC rels_near[], const struct CSC rels_far[], double* X, const struct CellComm comm[], int64_t levels) {
  int64_t ibegin = 0, iend = 0;
  self_local_range(&ibegin, &iend, &comm[levels]);
  int64_t lenX = (rhs[levels].X[iend - 1].A - rhs[levels].X[ibegin].A) + rhs[levels].X[iend - 1].M;
  memcpy(rhs[levels].X[ibegin].A, X, lenX * sizeof(double));

  int64_t xlen = rhs[levels].Xlen;
  double** arr_comm = (double**)malloc(sizeof(double*) * (xlen + 1));
  for (int64_t j = 0; j < xlen; j++)
    arr_comm[j] = rhs[levels].X[j].A;
  arr_comm[xlen] = arr_comm[xlen - 1] + rhs[levels].X[xlen - 1].M;
  dist_double_svbk('A', arr_comm, &comm[levels]);
  free(arr_comm);

  for (int64_t i = levels; i > 0; i--) {
    self_local_range(&ibegin, &iend, &comm[i]);
    int64_t iboxes = iend - ibegin;
    for (int64_t j = 0; j < iboxes; j++)
      mmult('T', 'N', &basis[i].Uo[j + ibegin], &rhs[i].X[j + ibegin], &rhs[i].XcM[j + ibegin], 1., 0.);
    xlen = rhs[i].Xlen;
    arr_comm = (double**)malloc(sizeof(double*) * (xlen + 1));
    for (int64_t j = 0; j < xlen; j++)
      arr_comm[j] = rhs[i].XcM[j].A;
    arr_comm[xlen] = arr_comm[xlen - 1] + rhs[i].XcM[xlen - 1].M;
    dist_double_svbk('A', arr_comm, &comm[i]);
    free(arr_comm);
    permuteAndMerge('F', rhs[i].XcM, rhs[i - 1].X, basis[i - 1].Lchild, &comm[i - 1]);
  }
  
  for (int64_t i = 1; i <= levels; i++) {
    permuteAndMerge('B', rhs[i].XoL, rhs[i - 1].B, basis[i - 1].Lchild, &comm[i - 1]);
    horizontalPass(rhs[i].XoL, rhs[i].XcM, A[i].S, &rels_far[i], &comm[i]);
    self_local_range(&ibegin, &iend, &comm[i]);
    int64_t iboxes = iend - ibegin;
    for (int64_t j = 0; j < iboxes; j++)
      mmult('N', 'N', &basis[i].Uo[j + ibegin], &rhs[i].XoL[j + ibegin], &rhs[i].B[j + ibegin], 1., 0.);
  }
  horizontalPass(rhs[levels].B, rhs[levels].X, A[levels].A, &rels_near[levels], &comm[levels]);
  memcpy(X, rhs[levels].B[ibegin].A, lenX * sizeof(double));
}


void solveRelErr(double* err_out, const double* X, const double* ref, int64_t lenX) {
  double err[2] = { 0., 0. };
  for (int64_t i = 0; i < lenX; i++) {
    double diff = X[i] - ref[i];
    err[0] = err[0] + diff * diff;
    err[1] = err[1] + ref[i] * ref[i];
  }
  MPI_Allreduce(MPI_IN_PLACE, err, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  *err_out = sqrt(err[0] / err[1]);
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  double prog_time = MPI_Wtime();

  int64_t Nbody = argc > 1 ? atol(argv[1]) : 8192;
  double theta = argc > 2 ? atof(argv[2]) : 1;
  int64_t leaf_size = argc > 3 ? atol(argv[3]) : 256;
  double epi = argc > 4 ? atof(argv[4]) : 1.e-10;
  int64_t rank_max = argc > 5 ? atol(argv[5]) : 100;
  int64_t sp_pts = argc > 6 ? atol(argv[6]) : 2000;
  const char* fname = argc > 7 ? argv[7] : NULL;

  int64_t levels = (int64_t)log2((double)Nbody / leaf_size);
  int64_t Nleaf = (int64_t)1 << levels;
  int64_t ncells = Nleaf + Nleaf - 1;
  
  //void(*ef)(double*) = laplace3d;
  void(*ef)(double*) = yukawa3d;
  set_kernel_constants(1.e-3 / Nbody, 1.);
  
  struct Body* body = (struct Body*)malloc(sizeof(struct Body) * Nbody);
  struct Cell* cell = (struct Cell*)malloc(sizeof(struct Cell) * ncells);
  struct CSC cellNear, cellFar;
  struct CSC* rels_far = (struct CSC*)malloc(sizeof(struct CSC) * (levels + 1));
  struct CSC* rels_near = (struct CSC*)malloc(sizeof(struct CSC) * (levels + 1));
  struct CellComm* cell_comm = (struct CellComm*)malloc(sizeof(struct CellComm) * (levels + 1));
  struct Base* basis = (struct Base*)malloc(sizeof(struct Base) * (levels + 1));
  struct Node* nodes = (struct Node*)malloc(sizeof(struct Node) * (levels + 1));
  struct RightHandSides* rhs = (struct RightHandSides*)malloc(sizeof(struct RightHandSides) * (levels + 1));

  if (fname == NULL) {
    mesh_unit_sphere(body, Nbody);
    //mesh_unit_cube(body, Nbody);
    //uniform_unit_cube(body, Nbody, 3, 1234);
    buildTree(&ncells, cell, body, Nbody, levels);
  }
  else {
    int64_t* buckets = (int64_t*)malloc(sizeof(int64_t) * Nleaf);
    read_sorted_bodies(Nbody, Nleaf, body, buckets, fname);
    buildTreeBuckets(cell, body, buckets, levels);
    free(buckets);
  }
  body_neutral_charge(body, Nbody, 1., 0);

  int64_t body_local[2];
  local_bodies(body_local, ncells, cell, levels);
  int64_t lenX = body_local[1] - body_local[0];
  double* X1 = (double*)malloc(sizeof(double) * lenX);
  double* X2 = (double*)malloc(sizeof(double) * lenX);

  traverse('N', &cellNear, ncells, cell, theta);
  traverse('F', &cellFar, ncells, cell, theta);
  relations(rels_near, ncells, cell, &cellNear, levels);
  relations(rels_far, ncells, cell, &cellFar, levels);
  buildComm(cell_comm, ncells, cell, &cellFar, &cellNear, levels);

  double construct_time, construct_comm_time;
  startTimer(&construct_time, &construct_comm_time);
  buildBasis(ef, basis, ncells, cell, rels_near, levels, cell_comm, body, Nbody, epi, rank_max, sp_pts);
  stopTimer(&construct_time, &construct_comm_time);
  
  allocNodes(nodes, basis, rels_near, rels_far, cell_comm, levels);

  evalD(ef, nodes[levels].A, ncells, cell, body, &rels_near[levels], levels);
  for (int64_t i = 0; i <= levels; i++)
    evalS(ef, nodes[i].S, &basis[i], body, &rels_far[i], &cell_comm[i]);

  if (Nbody > 10000) {
    loadX(X1, body_local, body);
    allocRightHandSides('M', rhs, basis, levels);
    matVecA(rhs, nodes, basis, rels_near, rels_far, X1, cell_comm, levels);
    for (int64_t i = 0; i <= levels; i++)
      rightHandSides_free(&rhs[i]);
  }
  else 
    mat_vec_reference(ef, body_local[0], body_local[1], X1, Nbody, body);
  
  double factor_time, factor_comm_time;
  startTimer(&factor_time, &factor_comm_time);
  factorA(nodes, basis, rels_near, rels_far, cell_comm, levels);
  stopTimer(&factor_time, &factor_comm_time);

  allocRightHandSides('S', rhs, basis, levels);

  double solve_time, solve_comm_time;
  startTimer(&solve_time, &solve_comm_time);
  solveA(rhs, nodes, basis, rels_near, X1, cell_comm, levels);
  stopTimer(&solve_time, &solve_comm_time);

  loadX(X2, body_local, body);
  double err;
  solveRelErr(&err, X1, X2, lenX);

  int64_t mem_basis, mem_A, mem_X;
  basis_mem(&mem_basis, basis, levels);
  node_mem(&mem_A, nodes, levels);
  rightHandSides_mem(&mem_X, rhs, levels);

  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  prog_time = MPI_Wtime() - prog_time;
  double cm_time;
  getCommTime(&cm_time);

  if (mpi_rank == 0)
    printf("LORASP: %d,%d,%lf,%d,%d\nConstruct: %lf s. COMM: %lf s.\n"
      "Factorize: %lf s. COMM: %lf s.\n"
      "Solution: %lf s. COMM: %lf s.\n"
      "Basis Memory: %lf GiB.\n"
      "Matrix Memory: %lf GiB.\n"
      "Vector Memory: %lf GiB.\n"
      "Err: %e\n"
      "Program: %lf s. COMM: %lf s.\n",
      (int)Nbody, (int)(Nbody / Nleaf), theta, 3, (int)mpi_size,
      construct_time, construct_comm_time, factor_time, factor_comm_time, solve_time, solve_comm_time, 
      (double)mem_basis * 1.e-9, (double)mem_A * 1.e-9, (double)mem_X * 1.e-9, err, prog_time, cm_time);

  for (int64_t i = 0; i <= levels; i++) {
    csc_free(&rels_far[i]);
    csc_free(&rels_near[i]);
    basis_free(&basis[i]);
    node_free(&nodes[i]);
    rightHandSides_free(&rhs[i]);
    cellComm_free(&cell_comm[i]);
  }
  csc_free(&cellFar);
  csc_free(&cellNear);
  
  free(body);
  free(cell);
  free(rels_far);
  free(rels_near);
  free(cell_comm);
  free(basis);
  free(nodes);
  free(rhs);
  free(X1);
  free(X2);
  MPI_Finalize();
  return 0;
}
