

Headers:
handle.h \ funcs.h

Namespace: gpu

A streamed cublas / cusolver wrapper that symplifies stream / handle / workspace management.

Why not lu(Matrix *A, Matrix *L, Matrix *U) interface?

1. Difficult to know the number of synchronizations required (If launched on same stream as A, L.sync?, U.sync?)

2. Workspace required by cusolver / cublas. 
  Alloc and dealloc within the func, will make call synchronous;
  Global workspace? have to deal with race condition and management for different streams;

Solution here:

1. Making a bundle for blas solver handles / stream / workspace / device_info, the "Stream" class.
  The creation of a stream include creating handles that are binding to the stream, as well as allocation of a
  dedicated amount of workspace.
  convinient s.sync function to synchronize stream.

2. An intermediate level between Mat-Mat operations and BLAS / LAPACK, so that the calls inside satisfies:
  a. Takes only data pointer, controls, and a Stream object as parameter
  b. All compute functions (not memory functions) has no memory alloc / dealloc inside
  c. All calls are asynchronously launched to stream, with only the exception of cudaMalloc in memory functions
  d. Does input parameter error checks and workspace size checks

Example code:

void gpu::dgetrf(Stream& s, double* a, int m, int n, int lda) {
  int Lwork;
  cusolverDnDgetrf_bufferSize(s.solvH, m, n, a, lda, &Lwork);
  if (Lwork <= (size_t)s.Lwork)
    cusolverDnDgetrf(s.solvH, m, n, a, lda, s.work, nullptr, s.dinfo);
  else
    std::cerr << "Insufficient work for DGETRF." << std::endl;
}

Explained:
  cusolverDnDgetrf takes workspace & device_info information in the function parameters, they have been hided.
  Typical practice requires allocating workspace as its size has been retrieved by bufferSize function, we pre-allocated and do a check on size instead.
  cusolver handle that is stored in s has been configured to use the cudaStream in s, so that the function launches asynchronously.