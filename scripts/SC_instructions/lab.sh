#!/bin/bash
#!/bin/bash
#YBATCH -r any_1
#SBATCH -N 1
#SBATCH -J TEST_HATRIX
#SBATCH --time=01:00:00
#SBATCH --output TEST_HATRIX.out
#SBATCH --error TEST_HATRIX.err


export PATH=$PATH:/home/sameer.deshmukh/gitrepos/jobscheduler2slack

source /etc/profile.d/modules.sh

printf "#### Setting up environment... "
printf "#### Done\n"
#################### CPU only build                         ####################
module load intel/2020

cd $HOME/dev/sandbox/Hatrix
source $PWD/scripts/SC_instructions/run_cmake_tests.sh

post_message "OUTPUT FILE"
post_message "$(cat /home/sameer.deshmukh/dev/sandbox/Hatrix/scripts/SC_instructions/TEST_HATRIX.out)"
post_message "ERROR FILE"
post_message "$(cat /home/sameer.deshmukh/dev/sandbox/Hatrix/scripts/SC_instructions/TEST_HATRIX.err)"
