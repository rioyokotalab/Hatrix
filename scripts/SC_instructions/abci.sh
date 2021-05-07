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

printf "#### Setting up environment... "
source /etc/profile.d/modules.sh
printf "Done\n"

cd $HOME/dev/sandbox/Hatrix
printf "#### Done\n"

#################### CPU only build                         ####################
printf "#### Building without CUDA... \n"
# Necessary modules
# module load intel/2020

# Build
mkdir build
cd build
cmake ..
make -j4
ctest
printf "#### Done\n\n"
cd ..

#################### Final cleanup                          ####################
printf "#### Cleanup... "
cd ..
rm -rf Hatrix
printf "Done\n"

exit 0
