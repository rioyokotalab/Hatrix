#!/bin/bash

export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/Users/sameer/gitrepos/parsec/build/lib/pkgconfig
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Users/sameer/gitrepos/parsec/build/lib
export PATH=/Users/sameer/gitrepos/parsec/build/bin:$PATH

rm h2_factorize.c h2_factorize.h h2_factorize.o
parsec-ptgpp -E -i h2_factorize.jdf -o h2_factorize
clang $(pkg-config --cflags parsec) -I../include/distributed -O0 -g h2_factorize.c -c -o h2_factorize.o
