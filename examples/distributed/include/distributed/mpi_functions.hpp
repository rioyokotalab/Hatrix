#pragma once

extern int MPIRANK, MPISIZE, MPIGRID[2], MYROW, MYCOL;

int mpi_rank(int i);
int mpi_rank(int i, int j);
