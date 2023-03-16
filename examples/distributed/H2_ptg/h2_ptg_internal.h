#ifndef H2_PTG_INTERNAL_H
#define H2_PTG_INTERNAL_H
typedef struct level_list {
  int length;
  int* indices;
} level_list;

typedef struct h2_block_list {
  int length;
  level_list *level_block_list;
} h2_block_list;

/* Put this in a C struct cuz parsec will not work with the CPP struct. */
typedef struct h2_factorize_params {
  int min_level;
  int max_level;
  int max_rank;
  int nleaf;
  h2_block_list *row_near_list;
  h2_block_list *col_near_list;
  h2_block_list *far_list;
  h2_block_list *row_fill_in_list, *col_fill_in_list;
} h2_factorize_params_t;

#endif
