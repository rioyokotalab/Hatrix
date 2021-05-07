#!/bin/bash

#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=01:00:00
#$ -N HATRIX_TEST
#$ -o HATRIX_TEST_out.log
#$ -e HATRIX_TEST_err.log
#$ -m ea
#$ -M deshmukh.s.aa@m.titech.ac.jp

set -e
cd $HOME/dev/sandbox/Hatrix

printf "#### Building without CUDA... \n"
printf "#### Setting up environment... "
source /etc/profile.d/modules.sh
#################### CPU only build                         ####################
module load cmake/3.19 gcc intel-mkl

# Build
source run_cmake_tests.sh
