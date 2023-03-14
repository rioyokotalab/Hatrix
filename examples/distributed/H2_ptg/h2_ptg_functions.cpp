#include "distributed/distributed.hpp"
#include "h2_ptg_functions.hpp"
#include "h2_ptg_internal.h"
#include "h2_factorize.h"
#include "h2_operations.hpp"

h2_dc_t parsec_U, parsec_S, parsec_D, parsec_F,
  parsec_temp_fill_in_rows, parsec_temp_fill_in_cols,
  parsec_US, parsec_r, parsec_t;

Hatrix::RowColLevelMap<Hatrix::Matrix> F;
Hatrix::RowColMap<Hatrix::Matrix> r, t;
// store temporary fill-ins.
Hatrix::RowColMap<Hatrix::Matrix> temp_fill_in_rows, temp_fill_in_cols;

using namespace Hatrix;

parsec_data_t*
data_of_key(parsec_data_collection_t* desc, parsec_data_key_t key) {
  h2_dc_t* dc = (h2_dc_t*)desc;
  parsec_data_t* data = NULL;
  if (dc->data_map.count(key) != 0) {
    data = dc->data_map[key];
  }

  if (NULL == data) {
    Matrix* mat = dc->matrix_map[key];
    // if (mat) {
      data = parsec_data_create(&dc->data_map[key], desc, key, &(*mat),
                                mat->numel() *
                                parsec_datadist_getsizeoftype(PARSEC_MATRIX_DOUBLE),
                                PARSEC_DATA_FLAG_PARSEC_MANAGED);
    // }
    // else {
    //   data = parsec_data_create(&dc->data_map[key], desc, key, NULL, 0,
    //                             PARSEC_DATA_FLAG_PARSEC_MANAGED);
    // }
    dc->data_map[key] = data;
  }

  return data;
}

uint32_t rank_of_key(parsec_data_collection_t* desc, parsec_data_key_t key) {
  h2_dc_t *dc = (h2_dc_t*)desc;
  return dc->mpi_ranks[key];
}

parsec_data_key_t data_key_1d(parsec_data_collection_t* dc, ...) {
  int b = -1, level = -1;

  /* Get coordinates */
  va_list ap;
  va_start(ap, dc);
  b = va_arg(ap, int);
  level = va_arg(ap, int);
  va_end(ap);

  if (b == -1 || level == -1) {
    std::cout << "[HATRIX_ERROR] data_key_1d() -> received wrong parameters b = "
              << b << " and level = " << level << ". Aborting.\n";
    abort();
  }

  parsec_data_key_t key = 0;
  for (int i = 0; i < level; ++i) {
    key += pow(2, i);
  }

  key += b;

  return key;
}

parsec_data_key_t
data_key_2d(parsec_data_collection_t* dc, ...) {
  int m = -1, n = -1, level = -1;

  va_list ap;
  va_start(ap, dc);
  m = va_arg(ap, int);
  n = va_arg(ap, int);
  level = va_arg(ap, int);
  va_end(ap);

  if (m == -1 || n == -1 || level == -1) {
    std::cout << "[HATRIX ERROR] data_key_2d() -> received a wrong key in m, n or level.\n";
    abort();
  }

  parsec_data_key_t key = 1;
  for (int i = 0; i < level; ++i) {
    key += pow(pow(2, i), 2);
  }

  int stride = pow(2, level);
  auto data_key = key + m * stride + n;

  return data_key;
}

parsec_data_t*
data_of_1d(parsec_data_collection_t* desc, ...) {
  int b = -1, level = -1;

  /* Get coordinates */
  va_list ap;
  va_start(ap, desc);
  b = va_arg(ap, int);
  level = va_arg(ap, int);
  va_end(ap);

  if (b == -1 || level == -1) {
    std::cout << "[HATRIX_ERROR] data_key_1d() -> received wrong parameters b = "
              << b << " and level = " << level << ". Aborting.\n";
  }
  parsec_data_key_t key = data_key_1d(desc, b, level);

  return data_of_key(desc, key);
}

parsec_data_t*
data_of_2d(parsec_data_collection_t* desc, ...) {
  int m = -1, n = -1, level = -1;

  va_list ap;
  va_start(ap, desc);
  m = va_arg(ap, int);
  n = va_arg(ap, int);
  level = va_arg(ap, int);
  va_end(ap);

  if (m == -1 || n == -1 || level == -1) {
    std::cout << "[HATRIX ERROR] data_key_2d() -> received a wrong key in m, n or level.\n";
    abort();
  }

  parsec_data_key_t key = data_key_2d(desc, m, n, level);

  // std::cout << "m: " << m  << " n: " << n << " level: " << level <<  " data of 2d: " << key << std::endl;

  return data_of_key(desc, key);
}

uint32_t rank_of_1d(parsec_data_collection_t* desc, ...) {
  int b = -1, level = -1;

  /* Get coordinates */
  va_list ap;
  va_start(ap, desc);
  b = va_arg(ap, int);
  level = va_arg(ap, int);
  va_end(ap);

  if (b == -1 || level == -1) {
    std::cout << "[HATRIX_ERROR] data_key_1d() -> received wrong parameters b = "
              << b << " and level = " << level << ". Aborting.\n";
    abort();
  }

  return mpi_rank(b);
}

uint32_t
rank_of_2d_as_1d(parsec_data_collection_t* desc, ...) {
  int m = -1, n = -1, level = -1;

  va_list ap;
  va_start(ap, desc);
  m = va_arg(ap, int);
  n = va_arg(ap, int);
  level = va_arg(ap, int);
  va_end(ap);

  return rank_of_1d(desc, m, level);
}

// block cyclic distribution.
uint32_t rank_of_2d(parsec_data_collection_t* desc, ...) {
  int m = -1, n = -1, level = -1;

  va_list ap;
  va_start(ap, desc);
  m = va_arg(ap, int);
  n = va_arg(ap, int);
  level = va_arg(ap, int);
  va_end(ap);

  if (m == -1 || n == -1 || level == -1) {
    std::cout << "[HATRIX ERROR] data_key_2d() -> received a wrong key in m, n or level.\n";
    abort();
  }

  return mpi_rank(m, n);
}

int32_t vpid_of_1d(parsec_data_collection_t* desc, ...) {
  int b = -1, level = -1;

  /* Get coordinates */
  va_list ap;
  va_start(ap, desc);
  b = va_arg(ap, int);
  level = va_arg(ap, int);
  va_end(ap);

  return (int32_t)rank_of_1d(desc, b, level);
}

int32_t vpid_of_2d_as_1d(parsec_data_collection_t* desc, ...) {
  int m = -1, n = -1, level = -1;

  va_list ap;
  va_start(ap, desc);
  m = va_arg(ap, int);
  n = va_arg(ap, int);
  level = va_arg(ap, int);
  va_end(ap);

  return (int32_t)vpid_of_1d(desc, m, level);
}

int32_t vpid_of_2d(parsec_data_collection_t* desc, ...) {
  int m = -1, n = -1, level = -1;

  va_list ap;
  va_start(ap, desc);
  m = va_arg(ap, int);
  n = va_arg(ap, int);
  level = va_arg(ap, int);
  va_end(ap);

  return (int32_t)rank_of_2d(desc, m, n, level);
}

int32_t vpid_of_key(parsec_data_collection_t *desc, parsec_data_key_t key) {
  return (int32_t)rank_of_key(desc, key);
}

void
h2_dc_init(h2_dc_t& parsec_data,
           parsec_data_key_t (*data_key_func)(parsec_data_collection_t*, ...),
           uint32_t (*rank_of_func)(parsec_data_collection_t*, ...),
           parsec_data_t* (*data_of_func)(parsec_data_collection_t*, ...),
           int32_t (*vpid_of_func)(parsec_data_collection_t*, ...)) {
  parsec_data_collection_t *o = (parsec_data_collection_t*)(&parsec_data);
  parsec_data_collection_init(o, MPISIZE, MPIRANK);

  o->data_key = data_key_func;

  o->rank_of = rank_of_func;
  o->rank_of_key = rank_of_key;

  o->data_of = data_of_func;
  o->data_of_key = data_of_key;

  o->vpid_of = vpid_of_func;
  o->vpid_of_key = vpid_of_key;

  parsec_dtd_data_collection_init(o);
}

void h2_dc_init_maps() {
  h2_dc_init(parsec_U, data_key_1d, rank_of_1d, data_of_1d, vpid_of_1d);
  h2_dc_init(parsec_D, data_key_2d, rank_of_2d_as_1d, data_of_2d, vpid_of_2d_as_1d);
  h2_dc_init(parsec_S, data_key_2d, rank_of_2d_as_1d, data_of_2d, vpid_of_2d_as_1d);
  h2_dc_init(parsec_F, data_key_2d, rank_of_2d_as_1d, data_of_2d, vpid_of_2d_as_1d);
  h2_dc_init(parsec_temp_fill_in_rows, data_key_1d, rank_of_1d, data_of_1d, vpid_of_1d);
  h2_dc_init(parsec_temp_fill_in_cols, data_key_1d, rank_of_1d, data_of_1d, vpid_of_1d);
  h2_dc_init(parsec_US, data_key_1d, rank_of_1d, data_of_1d, vpid_of_1d);
  h2_dc_init(parsec_r, data_key_1d, rank_of_1d, data_of_1d, vpid_of_1d);
  h2_dc_init(parsec_t, data_key_1d, rank_of_1d, data_of_1d, vpid_of_1d);
}

void
h2_dc_destroy_maps() {
  h2_dc_destroy(parsec_U);
  h2_dc_destroy(parsec_D);
  h2_dc_destroy(parsec_S);
  h2_dc_destroy(parsec_F);
  h2_dc_destroy(parsec_temp_fill_in_rows);
  h2_dc_destroy(parsec_temp_fill_in_cols);
  h2_dc_destroy(parsec_US);
  h2_dc_destroy(parsec_r);
  h2_dc_destroy(parsec_t);
}

void
h2_dc_destroy(h2_dc_t& parsec_data) {
  parsec_data_collection_t *o = (parsec_data_collection_t*)(&parsec_data);
  if (parsec_data.data_map.size()) {
    parsec_dtd_data_collection_fini(o);
    for (auto iter = parsec_data.data_map.begin(); iter != parsec_data.data_map.end(); ++iter) {
      parsec_data_destroy((iter->second));
    }

    parsec_data_collection_destroy(o);
  }
}


void
update_parsec_pointers(SymmetricSharedBasisMatrix& A, const Domain& domain,
                       int64_t level) {
  const int64_t nblocks = pow(2, level);

  // setup pointers to data for use with parsec.
  for (int64_t i = 0; i < nblocks; ++i) { // U
    parsec_data_key_t U_data_key = parsec_U.super.data_key(&parsec_U.super, i, level);
    parsec_data_key_t US_data_key = parsec_US.super.data_key(&parsec_US.super, i, level);
    parsec_data_key_t r_data_key = parsec_r.super.data_key(&parsec_r.super, i, level);
    parsec_data_key_t t_data_key = parsec_t.super.data_key(&parsec_t.super, i, level);

    if (mpi_rank(i) == MPIRANK) {
      Matrix& U_i = A.U(i, level);
      parsec_U.matrix_map[U_data_key] = std::addressof(U_i);

      Matrix& US_i = A.US(i, level);
      parsec_US.matrix_map[US_data_key] = std::addressof(US_i);

      Matrix& r_i = r(i, level);
      parsec_r.matrix_map[r_data_key] = std::addressof(r_i);

      Matrix& t_i = t(i, level);
      parsec_t.matrix_map[t_data_key] = std::addressof(t_i);
    }
    parsec_U.mpi_ranks[U_data_key] = mpi_rank(i);
    parsec_US.mpi_ranks[US_data_key] = mpi_rank(i);
    parsec_r.mpi_ranks[r_data_key] = mpi_rank(i);
    parsec_t.mpi_ranks[t_data_key] = mpi_rank(i);
  }

  for (int64_t i = 0; i < nblocks; ++i) {
    for (int64_t j = 0; j <= i; ++j) {
      int64_t row_size, col_size;
      row_size = A.ranks(i, level);
      col_size = A.ranks(j, level);
      parsec_data_key_t S_data_key = parsec_S.super.data_key(&parsec_S.super, i, j, level);

      if (exists_and_admissible(A, i, j, level) && mpi_rank(i) == MPIRANK) {     // S blocks.
        Matrix& S_ij = A.S(i, j, level);
        parsec_S.matrix_map[S_data_key] = std::addressof(S_ij);
      }
      parsec_S.mpi_ranks[S_data_key] = mpi_rank(i);

      row_size = get_dim(A, domain, i, level), col_size = get_dim(A, domain, j, level);
      parsec_data_key_t D_data_key = parsec_D.super.data_key(&parsec_D.super, i, j, level);
      parsec_D.mpi_ranks[D_data_key] = mpi_rank(i);

      if (exists_and_inadmissible(A, i, j, level) && (mpi_rank(i) == MPIRANK)) { // D blocks.
        Matrix& D_ij = A.D(i, j, level);
        parsec_D.matrix_map[D_data_key] = std::addressof(D_ij);
      }
    }
  }
}

void
preallocate_blocks(SymmetricSharedBasisMatrix& A) {
  // data structure r for storing the projection of the bases for a level.
  for (int level = A.max_level; level >= A.min_level-1; --level) {
    for (int64_t i = 0; i < pow(2, level); ++i) {
      if (mpi_rank(i) == MPIRANK) {
        int64_t rank = A.ranks(i, level);
        r.insert(i, level, generate_identity_matrix(rank, rank));

        Matrix t_i(rank, rank);
        t.insert(i, level, generate_identity_matrix(rank, rank));
      }
    }
  }

  for (int level = A.max_level - 1; level >= A.min_level; --level) {
    int nblocks = pow(2, level);
    int child_level = level + 1;

    // D blocks for holding merged blocks on level.
    for (int i = 0; i < nblocks; ++i) {
      for (int j = 0; j <= i; ++j) {
        if (exists_and_inadmissible(A, i, j, level)) {
          if (mpi_rank(i) == MPIRANK) {
            std::vector<int64_t> i_children({i * 2, i * 2 + 1}), j_children({j * 2, j * 2 + 1});

            const int64_t c_rows = A.ranks(i_children[0], child_level) +
              A.ranks(i_children[1], child_level);
            const int64_t c_cols = A.ranks(j_children[0], child_level) +
              A.ranks(j_children[1], child_level);
            Matrix D_unelim(c_rows, c_cols);

            A.D.insert(i, j, level, std::move(D_unelim));
          }
        }
      }
    }
  }

  int level = A.min_level - 1;
  int child_level = level + 1;
  int nblocks = pow(2, level);

  for (int i = 0; i < nblocks; ++i) {
    for (int j = 0; j <= i; ++j) {
      if (mpi_rank(i) == MPIRANK) {
        std::vector<int64_t> i_children({i * 2, i * 2 + 1}), j_children({j * 2, j * 2 + 1});

        const int64_t c_rows = A.ranks(i_children[0], child_level) +
          A.ranks(i_children[1], child_level);
        const int64_t c_cols = A.ranks(j_children[0], child_level) +
          A.ranks(j_children[1], child_level);
        Matrix D_unelim(c_rows, c_cols);
        A.D.insert(i, j, level, std::move(D_unelim));
      }
    }
  }
}

void
factorize_setup(SymmetricSharedBasisMatrix& A, Hatrix::Domain& domain,
                const Hatrix::Args& opts) {
  h2_dc_init_maps();

  preallocate_blocks(A);
  update_parsec_pointers(A, domain, A.max_level);

  for (int64_t level = A.max_level; level >= A.min_level; --level) {
    update_parsec_pointers(A, domain, level-1);
  }

  for (int level = A.max_level; level >= A.min_level; --level) {
    int nblocks = pow(2, level);

    for (int i = 0; i < nblocks; ++i) {
      for (int j = i + 1; j < nblocks; ++j) {
        if (A.is_admissible.exists(i, j, level)) {
          A.is_admissible.erase(i, j, level);
        }
      }
    }
  }
}


void factorize_teardown() {
  h2_dc_destroy_maps();
}

void
h2_factorize_Destruct(parsec_h2_factorize_taskpool_t *h2_factorize) {
  parsec_del2arena( &h2_factorize->arenas_datatypes[PARSEC_h2_factorize_DEFAULT_ADT_IDX] );
  // parsec_del2arena( &h2_factorize->arenas_datatypes[PARSEC_h2_factorize_D_ARENA_ADT_IDX] );
  // parsec_del2arena( &h2_factorize->arenas_datatypes[PARSEC_h2_factorize_U_ARENA_ADT_IDX] );

  parsec_taskpool_free((parsec_taskpool_t*)h2_factorize);
}

parsec_h2_factorize_taskpool_t *
h2_factorize_New(SymmetricSharedBasisMatrix& A, Hatrix::Domain& domain,
                 h2_factorize_params_t* h2_params) {
  parsec_data_collection_t *parsec_D_dc = &parsec_D.super;
  parsec_data_collection_t *parsec_U_dc = &parsec_U.super;

  parsec_h2_factorize_taskpool_t* h2_factorize = parsec_h2_factorize_new(parsec_D_dc,
                                                                         parsec_U_dc,
                                                                         h2_params);

  parsec_add2arena(&h2_factorize->arenas_datatypes[PARSEC_h2_factorize_DEFAULT_ADT_IDX],
                   parsec_datatype_double_t, PARSEC_MATRIX_FULL, 1,
                   h2_params->nleaf, h2_params->nleaf, h2_params->nleaf, PARSEC_ARENA_ALIGNMENT_SSE, -1);

  // parsec_add2arena(&h2_factorize_tasks->arenas_datatypes[PARSEC_h2_factorize_D_ARENA_ADT_IDX],
  //                  parsec_datatype_double_t, PARSEC_MATRIX_FULL, 1,
  //                  h2_params->nleaf, h2_params->nleaf, h2_params->nleaf, PARSEC_ARENA_ALIGNMENT_SSE, -1);

  // parsec_add2arena(&h2_factorize_tasks->arenas_datatypes[PARSEC_h2_factorize_U_ARENA_ADT_IDX],
  //                  parsec_datatype_double_t, PARSEC_MATRIX_FULL, 1,
  //                  h2_params->nleaf, h2_params->max_rank, h2_params->nleaf, PARSEC_ARENA_ALIGNMENT_SSE, -1);

  return h2_factorize;
}
