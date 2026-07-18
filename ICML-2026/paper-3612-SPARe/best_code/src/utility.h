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

#ifndef ORDP_UTILITY_H
#define ORDP_UTILITY_H

#include <random>
#include <string>

namespace util {

// Failure model parameters shared by all simulators.
struct FailureParams {
  std::string dist;
  double failure_prob;
  double exp_rate;
  double weibull_shape;
  double weibull_scale;
};

// Strip GPU suffix from host name to keep locality sorting stable.
std::string host_locality_key(const std::string& name);

// Seconds since epoch for seeding when RNG_SEED is unset.
unsigned seed_now();

// Configure and query worker-id field width for log alignment.
void set_worker_id_width(int world_size);
int worker_id_width();

// Standard log helpers with explicit total-step count.
void log_event(int id, int step, int total_steps, const char* label);
void log_event_with_shard(int id, int step, int total_steps, int shard, const char* label);

// Failure sampling helpers for Bernoulli / exponential / Weibull models.
double sample_failure_interval(std::mt19937& rng, const FailureParams& cfg);
double ensure_next_failure_time(std::mt19937& rng,
                               double now,
                               double& next_failure_time,
                               const FailureParams& cfg);
void reschedule_next_failure_time(std::mt19937& rng,
                                  double from_time,
                                  double& next_failure_time,
                                  const FailureParams& cfg);
double pick_failure_time(std::mt19937& rng,
                         std::uniform_real_distribution<double>& uni_dist,
                         double now,
                         double remaining,
                         double& next_failure_time,
                         const FailureParams& cfg);
bool should_fail(std::mt19937& rng,
                 std::uniform_real_distribution<double>& uni_dist,
                 double current_time,
                 double& next_failure_time,
                 const FailureParams& cfg);

} // namespace util

#endif
