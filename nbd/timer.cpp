
#include "timer.h"

#include <map>
#include <cstdint>

#ifdef _MSC_VER
#include <Windows.h>

int gettimeofday(struct timeval * tp, struct timezone * tzp) {
  static const uint64_t EPOCH = ((uint64_t) 116444736000000000ULL);

  SYSTEMTIME  system_time;
  FILETIME    file_time;
  uint64_t    time;

  GetSystemTime( &system_time );
  SystemTimeToFileTime( &system_time, &file_time );
  time =  ((uint64_t)file_time.dwLowDateTime )      ;
  time += ((uint64_t)file_time.dwHighDateTime) << 32;

  tp->tv_sec  = (long) ((time - EPOCH) / 10000000L);
  tp->tv_usec = (long) (system_time.wMilliseconds * 1000);
  return 0;
}

#else
#include <sys/time.h>
#endif

using namespace nbd;

static timeval t;
static std::map<std::string,timeval> timer;

void nbd::start(std::string event) {
  gettimeofday(&t, NULL);
  timer[event] = t;
}

void nbd::stop(std::string event) {
  gettimeofday(&t, NULL);
  printf("%-20s : %f s\n", event.c_str(), (int64_t)t.tv_sec-timer[event].tv_sec+((int64_t)t.tv_usec-timer[event].tv_usec)*1e-6);
}
