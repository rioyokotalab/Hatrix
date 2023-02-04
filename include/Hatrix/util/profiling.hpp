#ifdef HATRIX_ENABLE_PAPI
#include "papi.h"
#endif

namespace Hatrix {
  namespace profiling {
    // Wrapper class for PAPI. Create an object and call add_*.
    // This will create an entry for your counter in the papi_counters vector.
    // You can then access the counter using something like fp_ops().
    class PAPI {
    private:
      int event_set = 0;
      int num_hardware_counters;
    public:
      std::vector<long long int> papi_counters;
      int PAPI_FP_OPS_INDEX = -1;

      PAPI();
      void add_fp_ops(int FP_OPS_INDEX);
      void read();
      void start();
      long long int fp_ops();
      ~PAPI();
    };
  }
}
