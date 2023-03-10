#ifndef H2_PTG_INTERNAL_H
#define H2_PTG_INTERNAL_H
/* Put this in a C struct cuz parsec will not work with the CPP struct. */
typedef struct h2_factorize_params {
  int min_level;
  int max_level;
  int max_rank;
  int nleaf;
} h2_factorize_params_t;

#endif
