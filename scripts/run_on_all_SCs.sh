#!/bin/bash

for server in lab abci; do
    ssh $server 'bash -l -s' < submit_script.sh
done

echo "Submitted batch scripts for all machines!"
