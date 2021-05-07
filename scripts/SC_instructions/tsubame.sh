#!/bin/bash
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=00:20:00
#$ -N HATRIX_TSUBAME
#$ -o HATRIX_TSUBAME_out.log
#$ -e HATRIX_TSUBAME_err.log
#$ -m ea
#$ -M ${MY_EMAIL}

set -e

printf "#### Setting up environment... "
cd $HOME/dev/sandbox/Hatrix

printf "#### Done\n"

#################### CPU only build                         ####################
printf "#### Building without CUDA... \n"
# Necessary modules
. /etc/profile.d/modules.sh
module load cmake/3.17.2 gcc/8.3.0

# Build
mkdir build
cd build
cmake ..
make -j
ctest

exit 0
