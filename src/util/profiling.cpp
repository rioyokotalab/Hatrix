#include <vector>
#include <iostream>
#include <cassert>

#include "Hatrix/util/profiling.hpp"

namespace Hatrix {
  namespace profiling {
    PAPI::PAPI() {
#ifdef HATRIX_ENABLE_PAPI
      event_set = PAPI_NULL;
      int retval = PAPI_library_init(PAPI_VER_CURRENT);
      if (retval != PAPI_VER_CURRENT && retval > 0) {
        std::cerr << "PAPI library init error: "
                  << retval << std::endl;
      }
      num_hardware_counters = PAPI_num_hwctrs();
      papi_counters.resize(num_hardware_counters, 0);
#endif
    }

    void PAPI::add_fp_ops(int FP_OPS_INDEX) {
#ifdef HATRIX_ENABLE_PAPI
      assert(FP_OPS_INDEX < num_hardware_counters &&
             FP_OPS_INDEX >= 0);
      PAPI_FP_OPS_INDEX = FP_OPS_INDEX;
      assert(PAPI_create_eventset(&event_set) == PAPI_OK);
      assert(PAPI_query_event(PAPI_FP_OPS) == PAPI_OK);
      assert(PAPI_add_event(event_set, PAPI_FP_OPS) == PAPI_OK);
#endif
    }

    void PAPI::start() {
#ifdef HATRIX_ENABLE_PAPI
      PAPI_start(event_set);
#endif
    }

    void PAPI::read() {
#ifdef HATRIX_ENABLE_PAPI
      if (PAPI_FP_OPS_INDEX != -1) {
        PAPI_read(event_set,
                  &papi_counters[PAPI_FP_OPS_INDEX]);
        PAPI_reset(event_set);
      }
#endif
    }

    long long int
    PAPI::fp_ops() {
#ifdef HATRIX_ENABLE_PAPI
      assert(PAPI_FP_OPS_INDEX != -1);
      read();
      return papi_counters[PAPI_FP_OPS_INDEX];
#else
      return 0;
#endif
    }

    PAPI::~PAPI() {
#ifdef HATRIX_ENABLE_PAPI
      PAPI_shutdown();
#endif
    }
  }
}
