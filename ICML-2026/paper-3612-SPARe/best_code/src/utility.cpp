/*
* Copyright (c) 2026, University of Florida. All rights reserved.
*
* This file is part of the simulation software package developed by
* the team members of Prof. Xiaoyi Lu's group at University of Florida.
*
* For detailed copyright and licensing information, please refer to the license
* file LICENSE in the top level directory.
*
*/

#include "utility.h"

#include <algorithm>
#include <chrono>

#include <simgrid/s4u.hpp>

XBT_LOG_NEW_DEFAULT_CATEGORY(ordp_util, "ORDP utility log");

namespace util {

static int g_worker_id_width = 1;

void set_worker_id_width(int world_size)
{
  int max_id = std::max(0, world_size - 1);
  int width = 1;
  while (max_id >= 10) {
    width++;
    max_id /= 10;
  }
  g_worker_id_width = width;
}

int worker_id_width()
{
  return g_worker_id_width;
}

// Normalize host name by stripping GPU suffix to keep locality sorting stable.
std::string host_locality_key(const std::string& name)
{
  auto pos = name.rfind("_g");
  if (pos != std::string::npos) return name.substr(0, pos);
  return name;
}

// Seconds since epoch as a simple runtime seed when RNG_SEED is unset.
unsigned seed_now()
{
  return static_cast<unsigned>(
      std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

// Standard log format for control/communication events.
void log_event(int id, int step, int total_steps, const char* label)
{
  XBT_INFO("[worker%*d %.6f] step %03d/%d -- %-12s",
           worker_id_width(),
           id,
           simgrid::s4u::Engine::get_clock(),
           step + 1,
           total_steps,
           label);
}

// Log format for events that should include shard assignment.
void log_event_with_shard(int id, int step, int total_steps, int shard, const char* label)
{
  XBT_INFO("[worker%*d %.6f] step %03d/%d -- %-12s [shard %d]",
           worker_id_width(),
           id,
           simgrid::s4u::Engine::get_clock(),
           step + 1,
           total_steps,
           label,
           shard);
}

// Draw next failure interval from the configured distribution.
double sample_failure_interval(std::mt19937& rng, const FailureParams& cfg)
{
  if (cfg.dist == "weibull") {
    if (cfg.weibull_shape <= 0.0 || cfg.weibull_scale <= 0.0) return -1.0;
    std::weibull_distribution<double> fail_dist(cfg.weibull_shape, cfg.weibull_scale);
    return std::max(1e-9, fail_dist(rng));
  }
  if (cfg.dist == "exponential") {
    if (cfg.exp_rate <= 0.0) return -1.0;
    std::exponential_distribution<double> fail_dist(cfg.exp_rate);
    return std::max(1e-9, fail_dist(rng));
  }
  return -1.0;
}

// Lazily initialize next_failure_time on first use.
double ensure_next_failure_time(std::mt19937& rng,
                               double now,
                               double& next_failure_time,
                               const FailureParams& cfg)
{
  if (next_failure_time <= 0.0) {
    double interval = sample_failure_interval(rng, cfg);
    if (interval <= 0.0) return -1.0;
    next_failure_time = now + interval;
  }
  return next_failure_time;
}

// After a failure, draw the next interval based on the current time.
void reschedule_next_failure_time(std::mt19937& rng,
                                  double from_time,
                                  double& next_failure_time,
                                  const FailureParams& cfg)
{
  double interval = sample_failure_interval(rng, cfg);
  if (interval > 0.0) next_failure_time = from_time + interval;
}

// Choose a failure time within the remaining window or return -1 if no failure.
double pick_failure_time(std::mt19937& rng,
                         std::uniform_real_distribution<double>& uni_dist,
                         double now,
                         double remaining,
                         double& next_failure_time,
                         const FailureParams& cfg)
{
  if (cfg.dist == "weibull" || cfg.dist == "exponential") {
    double next_time = ensure_next_failure_time(rng, now, next_failure_time, cfg);
    if (next_time > 0.0 && next_time < now + remaining) return next_time;
    return -1.0;
  }
  if (uni_dist(rng) < cfg.failure_prob) return now + (remaining * uni_dist(rng));
  return -1.0;
}

// Return whether a failure triggers at current_time based on configured distribution.
bool should_fail(std::mt19937& rng,
                 std::uniform_real_distribution<double>& uni_dist,
                 double current_time,
                 double& next_failure_time,
                 const FailureParams& cfg)
{
  if (cfg.dist == "weibull" || cfg.dist == "exponential") {
    double next_time = ensure_next_failure_time(rng, current_time, next_failure_time, cfg);
    if (next_time <= 0.0 || current_time < next_time) return false;
    reschedule_next_failure_time(rng, current_time, next_failure_time, cfg);
    return true;
  }
  return uni_dist(rng) < cfg.failure_prob;
}

} // namespace util
