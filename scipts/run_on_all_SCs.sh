#!/bin/bash

CURRENT_GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

printf "Building branch $CURRENT_GIT_BRANCH on all supercomputers...\n"

SCRIPTS_PATH=SC_instructions
for SCRIPT in $(find $SCRIPTS_PATH -type f -printf "%f\n"); do
    SUPERCOMPUTER=${SCRIPT%.sh}
    printf "\n######### Starting build on $SUPERCOMPUTER...\n"
    ssh $SUPERCOMPUTER 'bash -s' < $SCRIPTS_PATH/$SCRIPT $CURRENT_GIT_BRANCH
    printf "\n######### Build on $SUPERCOMPUTER finished succesfully!\n\n"
done

echo "Build on all supercomputers finished!"
