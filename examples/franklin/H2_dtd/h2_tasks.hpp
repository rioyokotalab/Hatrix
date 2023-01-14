#pragma once

#include "parsec.h"

/* we need the DTD internals to get access to the context members. */
#include "parsec/interfaces/dtd/insert_function_internal.h"

#include "parsec/interfaces/dtd/insert_function.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data.h"

parsec_hook_return_t
task_multiply_full_complement(parsec_execution_stream_t* es, parsec_task_t* this_task);

parsec_hook_return_t
task_multiply_partial_complement_left(parsec_execution_stream_t* es, parsec_task_t* this_task);

parsec_hook_return_t
task_multiply_partial_complement_right(parsec_execution_stream_t* es, parsec_task_t* this_task);

parsec_hook_return_t
task_factorize_diagonal(parsec_execution_stream_t* es, parsec_task_t* this_task);

parsec_hook_return_t
task_trsm_co(parsec_execution_stream_t* es, parsec_task_t* this_task);

parsec_hook_return_t
task_trsm_cc_oc(parsec_execution_stream_t* es, parsec_task_t* this_task);

parsec_hook_return_t
task_partial_syrk(parsec_execution_stream_t* es, parsec_task_t* this_task);

parsec_hook_return_t
task_partial_matmul(parsec_execution_stream_t* es, parsec_task_t* this_task);

parsec_hook_return_t
task_copy_blocks(parsec_execution_stream_t* es, parsec_task_t* this_task);

parsec_hook_return_t
task_nb_nb_fill_in(parsec_execution_stream_t* es, parsec_task_t* this_task);

parsec_hook_return_t
task_nb_rank_fill_in(parsec_execution_stream_t* es, parsec_task_t* this_task);

parsec_hook_return_t
task_cholesky_full(parsec_execution_stream_t* es, parsec_task_t* this_task);

parsec_hook_return_t
task_solve_triangular_full(parsec_execution_stream_t* es, parsec_task_t* this_task);

parsec_hook_return_t
task_syrk_full(parsec_execution_stream_t* es, parsec_task_t* this_task);

parsec_hook_return_t
task_matmul_full(parsec_execution_stream_t* es, parsec_task_t* this_task);

parsec_hook_return_t
task_fill_in_addition(parsec_execution_stream_t* es, parsec_task_t* this_task);

parsec_hook_return_t
task_fill_in_QR(parsec_execution_stream_t* es, parsec_task_t* this_task);

parsec_hook_return_t
task_project_S(parsec_execution_stream_t* es, parsec_task_t* this_task);

parsec_hook_return_t
task_fill_in_cols_addition(parsec_execution_stream_t* es, parsec_task_t* this_task);
