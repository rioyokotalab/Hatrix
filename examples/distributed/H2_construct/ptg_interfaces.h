#ifndef PTG_INTERFACES_H
#define PTG_INTERFACES_H

#include <stdint.h>
#include "parsec.h"

/* Global declarations for use with parsec declarations. */
typedef struct H2_ptg_t H2_ptg_t;
extern struct H2_ptg_t DENSE_BLOCKS;

int64_t near_neighbours_size(int64_t node, int64_t level);
int64_t near_neighbours_index(int64_t node, int64_t level, int64_t array_loc);

int64_t far_neighbours_size(int64_t node, int64_t level);
int64_t far_neighbours_index(int64_t node, int64_t level, int64_t array_loc);

#endif
