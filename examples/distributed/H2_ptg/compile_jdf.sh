#!/bin/bash

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/Users/sameer/gitrepos/parsec/build/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Users/sameer/gitrepos/parsec/build/lib
export PATH=/Users/sameer/gitrepos/parsec/build/bin:$PATH

parsec-ptgpp -E -i h2_factorize.jdf -o h2_factorize
clang $(pkg-config --cflags parsec) h2_factorize.c -c -o h2_factorize.o
