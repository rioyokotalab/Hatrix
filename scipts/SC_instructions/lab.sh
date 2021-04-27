#!/bin/bash
set -e

if [ $1 == "" ]; then
  exit 1
else
  HATRIX_BRANCH=$1
fi

printf "#### Setting up environment... "
source /etc/profile.d/modules.sh
printf "Done\n"

printf "#### Cloning Hatrix to ~/dev/sandbox/Hatrix... \n"
cd dev/sandbox
git clone --depth 1 --branch $HATRIX_BRANCH git@github.com:rioyokotalab/Hatrix.git
cd Hatrix
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
