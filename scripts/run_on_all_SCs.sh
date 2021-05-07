#!/bin/bash

set -e
# CURRENT_GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# printf "Building branch $CURRENT_GIT_BRANCH on all supercomputers...\n"

# SCRIPTS_PATH=SC_instructions
# for SCRIPT in $(find $SCRIPTS_PATH -type f -printf "%f\n"); do
#     SUPERCOMPUTER=${SCRIPT%.sh}
#     printf "\n######### Starting build on $SUPERCOMPUTER...\n"
#     scp $SCRIPTS_PATH/$SCRIPT
#     ssh $SUPERCOMPUTER 'bash -s' < $SCRIPTS_PATH/$SCRIPT $CURRENT_GIT_BRANCH
#     printf "\n######### Build on $SUPERCOMPUTER finished succesfully!\n\n"
# done


for server in abci; do
    ssh  $server 'bash -l -s' < submit_script.sh
done

echo "Submitted batch scripts for all machines!"
