#pragma once

#include <chrono>
#include <map>
#include <string>
#include <vector>

namespace Hatrix {
namespace timing {

class Timer;

// Functions to manipulate global timer state machine
/**
 * @brief Starts timing of an event
 * 
 * Starts the timer for a specified event.
 * 
 * @param event the name of the event
 * @return Timer& reference to the timer
 */
Timer& start(const std::string event);

/**
 * @brief Stops the timing of an event
 * 
 * Stops the timer for a specified event.
 * Returns 0 if there is no corresponding event.
 * 
 * @param event the name of the event
 * @return double elapsed time in seconds
 */
double stop(const std::string event);

/**
 * @brief Resets all timers
 * 
 * Clears all stored events and thus resets all timings.
 * 
 */
void clearTimers();

/**
 * @brief Stops the timing of an event and prints the result
 * 
 * Stops the timer for a specified event and prints the result
 * up to a specified depth.
 * The depth specifies the granularity of the result
 * (i.e. it also displays sub-timers)
 * 
 * @param event the name of the event
 * @param depth the number of sub-timers to print
 */
void stopAndPrint(const std::string event, const int depth = 0);

/**
 * @brief Prints the timings for an event
 * 
 * Prints the timings for a specified event
 * up to a specified depth.
 * The depth specifies the granularity of the result
 * (i.e. it also displays sub-timers)
 * 
 * @param event the name of the event
 * @param depth the number of sub-timers to print
 */
void printTime(const std::string event, const int depth = 0);

/**
 * @brief Calculates the total time for an event
 * 
 * Total time represents the sum of the timer and all
 * sub-timers
 * 
 * @param event name of the event
 * @return double elapsed time in seconds
 */
double getTotalTime(const std::string event);

/**
 * @brief Get the number of timings for an event
 * 
 * Total number of timings for an event
 * (i.e. number of sub-timers).
 * 
 * @param event name of the event
 * @return unsigned int number of timings
 */
unsigned int getNRuns(const std::string event);

// Interface of the Timer class if user wants to create own timers
class Timer {
public:
  Timer();

  Timer(const std::string name, Timer* parent=nullptr);

  void start();

  void start_subtimer(const std::string event);

  double stop();

  void clear();

  std::string get_name() const;

  Timer* get_parent() const;

  std::vector<double> get_times() const;

  size_t get_n_runs() const;

  double get_total_time() const;

  const std::map<std::string, double> get_subtimers() const;

  const Timer& operator[](const std::string event) const;

  Timer& operator[](const std::string event);

  void print_to_depth(const int depth) const;

private:
  using clock = std::chrono::high_resolution_clock;
  typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_point;
  typedef std::chrono::duration<double> seconds;
  std::string name = "";
  Timer* parent = nullptr;
  bool running = false;
  time_point start_time;
  std::vector<seconds> times;
  seconds total_time = seconds::zero();
  std::map<std::string, Timer> subtimers;

  void print_to_depth(const int depth, const int at_depth, const double root_total_time = 0, const std::string tag_pre = "") const;
}; 

}  // namespace timing
}  // namespace Hatrix

