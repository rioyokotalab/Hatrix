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
task_multiply_partial_complement(parsec_execution_stream_t* es, parsec_task_t* this_task);

parsec_hook_return_t
task_factorize_diagonal(parsec_execution_stream_t* es, parsec_task_t* this_task);

parsec_hook_return_t
task_partial_trsm(parsec_execution_stream_t* es, parsec_task_t* this_task);

parsec_hook_return_t
task_partial_syrk(parsec_execution_stream_t* es, parsec_task_t* this_task);

parsec_hook_return_t
task_partial_matmul(parsec_execution_stream_t* es, parsec_task_t* this_task);
