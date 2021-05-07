#!/bin/bash

set -e

for server in lab; do
    ssh  $server 'bash -l -s' < submit_script.sh
done

echo "Submitted batch scripts for all machines!"
