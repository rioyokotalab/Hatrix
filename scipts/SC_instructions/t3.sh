#!/bin/bash
set -e

if [ $1 == "" ]; then
  exit 1
else
  HATRIX_BRANCH=$1
fi

echo "Found $HATRIX_BRANCH!"

exit 0
