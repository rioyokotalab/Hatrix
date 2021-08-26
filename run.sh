make -j all
valgrind --leak-check=full ./bin/Qsparse_weak_1level 1000 10 100
