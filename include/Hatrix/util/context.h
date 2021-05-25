#pragma once

#include <cstddef>

namespace Hatrix {

    constexpr size_t DEFAULT_LWORK = 0x80000000ULL;
    constexpr size_t DEFAULT_LWORK_HOST = 0x4000000ULL;

    enum class mode_t : int {
      SERIAL = 0, PARALLEL = 1
    };

    enum class arg_t : int {
      STREAM = 0, BLAS = 1, SOLV = 2, RAND = 3, BLAS_SOL = 4
    };

    void init(int nstream, size_t Lwork = DEFAULT_LWORK, size_t Lwork_host = DEFAULT_LWORK_HOST);

    void term();

    void sync(int stream = -1);

    mode_t parallel_mode(mode_t mode);

    void runtime_args(void** args, arg_t type);

    void generator_seed(long long unsigned int seed);

    void time_start(int stream = -1);

    void time_end(int stream = -1);

    float get_time(int stream = -1);

}  // namespace Hatrix
