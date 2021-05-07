
#include <handle.h>
#include <funcs.h>

#include <cuda_runtime_api.h>
#include <stdio.h>

int main() {

	using namespace Hatrix::gpu;

	const int m = 5, n = 5, min_mn = 5, rank = 5;

	double h_A[m * n] = {
		0.76420743, 0.61411544, 0.81724151, 0.42040879, 0.03446089,
		0.03697287, 0.85962444, 0.67584086, 0.45594666, 0.02074835,
		0.42018265, 0.39204509, 0.12657948, 0.90250559, 0.23076218,
		0.50339844, 0.92974961, 0.21213988, 0.63962457, 0.58124562,
		0.58325673, 0.11589871, 0.39831112, 0.21492685, 0.00540355
	};

	double S_ref[min_mn] = {
	2.36539241,
	0.81117785,
	0.68562255,
	0.41390509,
	0.01519322 };

	/* singular values computed by GESVDR */
	double S_gpu[min_mn] = { 0, 0, 0, 0, 0 };

	Stream s;
	double* d_A, *d_U, *d_V, *d_S;
	
	cudaMalloc((void**)&d_A, sizeof(double) * m * n);
	cudaMalloc((void**)&d_U, sizeof(double) * m * m);
	cudaMalloc((void**)&d_V, sizeof(double) * n * n);
	cudaMalloc((void**)&d_S, sizeof(double) * n);

	cudaMemcpy(d_A, h_A, sizeof(double) * m * n, cudaMemcpyHostToDevice);

	dgesvdr(s, 'S', 'S', m, n, rank, 2, 2, d_A, m, d_S, d_U, m, d_V, n);
	s.sync();

	cudaMemcpy(S_gpu, d_S, rank * sizeof(double), cudaMemcpyDeviceToHost);

	printf("compare singular values.\n");
	double max_err = 0;
	double max_relerr = 0;
	for (int i = 0; i < rank; i++) {
		const double lambda_ref = S_ref[i];
		const double lambda_gpu = S_gpu[i];
		const double AbsErr = fabs(lambda_ref - lambda_gpu);
		const double RelErr = AbsErr / lambda_ref;

		max_err = max_err > AbsErr ? max_err : AbsErr;
		max_relerr = max_relerr > RelErr ? max_relerr : RelErr;

		printf("S_ref[%d]=%f  S_gpu=[%d]=%f  AbsErr=%E  RelErr=%E\n",
			i, lambda_ref, i, lambda_gpu, AbsErr, RelErr);
	}
	printf("\n");

	double eps = 1.E-8;
	printf("max_err = %E, max_relerr = %E, eps = %E\n", max_err, max_relerr, eps);

	if (max_relerr > eps) {
		printf("Error: max_relerr is bigger than eps\n");
		printf("try to increase oversampling or iters\n");
		printf("otherwise, reduce eps\n");
	}
	else {
		printf("Success: max_relerr is smaller than eps\n");
	}

	/* Deallocate host and device workspace */
	if (d_A) { cudaFree(d_A); }
	if (d_U) { cudaFree(d_U); }
	if (d_V) { cudaFree(d_V); }
	if (d_S) { cudaFree(d_S); }

  return 0;
}
