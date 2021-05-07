#!/bin/bash
#!/bin/bash
#YBATCH -r any_1
#SBATCH -N 1
#SBATCH -J TEST_HATRIX
#SBATCH --time=01:00:00
#SBATCH --mail-user=deshmukh.s.aa@m.titech.ac.jp
#SBATCH --mail-type=ALL

set -e

source /etc/profile.d/modules.sh

printf "#### Setting up environment... "
printf "#### Done\n"
#################### CPU only build                         ####################
module load intel/2020

cd $HOME/dev/sandbox/Hatrix
source $PWD/scripts/SC_instructions/run_cmake_tests.sh
