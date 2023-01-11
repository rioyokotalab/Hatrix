#pragma once

#include "parsec.h"


#include <vector>

extern "C" {
  /* Cblacs declarations: https://netlib.org/blacs/BLACS/QRef.html */
  void Cblacs_pinfo(int*, int*);
  void Cblacs_get(int CONTEXT, int WHAT, int*VALUE);
  void Cblacs_gridinit(int*, const char*, int, int);
  // returns the co-ordinates of the process number PNUM in PROW and PCOL.
  void Cblacs_pcoord(int CONTEXT, int PNUM, int* PROW, int* PCOL);
  void Cblacs_gridexit(int CONTEXT);
  void Cblacs_barrier(int, const char*);
  void Cblacs_exit(int CONTINUE);
  void Cblacs_gridmap(int* CONTEXT, int* USERMAP, const int LDUMAP,
                      const int NPROW, const int NPCOL);
  void Cblacs_gridinfo(int CONTEXT, int *NPROW, int *NPCOL,
                       int *MYROW, int *MYCOL);

  // calculate the number of rows and cols owned by process IPROC.
  // IPROC :: (local input) INTEGER
  //          The coordinate of the process whose local array row or
  //          column is to be determined.
  // ISRCPROC :: The coordinate of the process that possesses the first
  //             row or column of the distributed matrix. Global input.
  int numroc_(const int* N, const int* NB, const int* IPROC, const int* ISRCPROC ,
              const int* NPROCS);

  // init descriptor for scalapack matrices.
  void descinit_(int *desc, const int *m,  const int *n, const int *mb,
                 const int *nb, const int *irsrc, const int *icsrc, const int *BLACS_CONTEXT,
                 const int *lld, int *info);

  // set values of the descriptor without error checking.
  void descset_(int *desc, const int *m,  const int *n, const int *mb,
                const int *nb, const int *irsrc, const int *icsrc, const int *BLACS_CONTEXT,
                const int *lld, int *info);

  void pdgemm_(const char *TRANSA, const char *TRANSB,
               const int *M, const int *N, const int *K,
               const double *ALPHA,
               double *A, int *IA, int *JA, int *DESCA,
               double *B, int *IB, int *JB, int *DESCB,
               double *	BETA,
               double *	C, int *IC, int *JC, int *DESCC);

  void pdgemv_(const char *TRANS, int *M, int *N, double *ALPHA,
               double *A, int *IA, int *JA, int *DESCA,
               double * X, int * IX, int *JX, int * DESCX,
               int * INCX,
               double *BETA, double *Y, int *IY, int *JY, int *DESCY,
               int *INCY);

  // Pivoted QR. A is replaced with elementary reflectors of A.
  void	pdgeqpf_(const int* m, const int* n, double* a, const int* ia,
                 const int* ja, const int* desca, int* ipiv, double* tau,
                 double* work, const int* lwork, int* info);

  // Obtain QR factors from elementary reflectors obtained from geqpf.
  void	pdorgqr_(const int* m, const int* n, const int* k,
                 double* a, const int* ia, const int* ja,
                 const int* desca, const double* tau, double* work,
                 const int* lwork, int* info);

  // Copy general matrices from one context to another.
  void pdgemr2d_(const int *m, const int *n,
                 const double *a, const int *ia, const int *ja, const int *desca,
                 double *b, const int *ib, const int *jb, const int *descb,
                 const int *ictxt);

  // frobenius norm
  double pdlange_(const char* norm, const int* m, const int* n,
                   const double* a, const int* ia, const int* ja,
                   const int* desca, double* work);

  // cholesky
  void pdpotrf_(const char* uplo, const int* n, double* a,
                const int* ia, const int* ja, const int* desca,
                int* info);

  void pdtrsm_(const char *side, const char *uplo, const char *transa, const char *diag,
               const int *m, const int *n, const double *alpha,
               const double *a, const int *ia, const int *ja, const int *desca,
               double *b, const int *ib, const int *jb, const int *descb );

  void pdgesvd_(const char* jobu, const char* jobvt,
                const int* m, const int* n, const double* a, const int* ia, const int* ja,
                const int* desca, double* s, double* u, const int* iu, const int* ju,
                const int* descu, double* vt, const int* ivt, const int* jvt,
                const int* descvt, double* work, const int* lwork, int* info);


//   // scalapack copying routines
//   // https://netlib.org/scalapack/slug/node164.html
}

// translate global indices to local indices. INDXGLOB is the global index for the
// row/col. Returns the local FORTRAN-style index. NPROCS is the number of processes
// in that row or col.
int indxg2l(int INDXGLOB, int NB, int NPROCS);
int indxl2g(int indxloc, int nb, int iproc, int isrcproc, int nprocs);
int indxg2p(int INDXGLOB, int NB, int ISRCPROC, int NPROCS);
int mpi_rank(int i);
int mpi_rank(int i, int j);

extern parsec_context_t *parsec;
extern parsec_taskpool_t *dtd_tp;
extern int MPIRANK, MPISIZE, MPIGRID[2], MYROW, MYCOL;
extern int N; // total matrix dimension.
extern int BLACS_CONTEXT;
extern int info;

extern int ZERO, ONE, MINUS_ONE;
extern const char NOTRANS;
extern const char TRANS;
extern int DENSE_NBROW, DENSE_NBCOL;
extern int DENSE_local_rows, DENSE_local_cols;

constexpr int DESC_LEN = 9;

// SCALAPACK data for the large dense matrix used for construction and verification.
extern double* DENSE_MEM;
extern std::vector<int> DENSE;
