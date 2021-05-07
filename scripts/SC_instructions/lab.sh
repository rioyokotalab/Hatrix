#!/bin/bash
#!/bin/bash
#YBATCH -r epyc-7502_8
#SBATCH -N 1
#SBATCH -J TEST_HATRIX
#SBATCH --time=10:00:00
#SBATCH --mail-user=deshmukh.s.aa@m.titech.ac.jp
#SBATCH --mail-type=ALL

set -e

if [ $1 == "" ]; then
  exit 1
else
  HATRIX_BRANCH=$1
fi

printf "#### Setting up environment... "
source /etc/profile.d/modules.sh
printf "Done\n"

cd $HOME/dev/sandbox/Hatrix
printf "#### Done\n"

#################### CPU only build                         ####################
printf "#### Building without CUDA... \n"
# Necessary modules
module load intel/2020

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
