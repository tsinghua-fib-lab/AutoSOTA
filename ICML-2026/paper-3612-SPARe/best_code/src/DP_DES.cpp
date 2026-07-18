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

// Vanilla Data Parallelism without duplication

#include <simgrid/s4u.hpp>

#include "utility.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace simgrid;

XBT_LOG_NEW_DEFAULT_CATEGORY(ordp_des, "ORDP DES log");

static int WORLD_SIZE = 4;
static int STEPS = 10;
static int CHECKPOINT_INTERVAL = 2;
static double MODEL_COMPUTE_FLOPS = 5e13;// 50 TFLOPs
static double CONT_COMPUTE_FLOPS = 5e6;
static double MODEL_SIZE = 4e9;          // 4 GB
static double ALLREDUCE_FAIL_SCALE = 1.0;
static double ALLREDUCE_TIME = 0.0;      // Seconds per allreduce (constant)
static double DATA_SIZE = 4e6;           // 4 MB
static double CPU_SCALE = 1;             // CPU scale down
static double FAILURE_PROB = 0.1;
static std::string FAILURE_DIST = "exponential";
static double EXP_FAILURE_RATE = 0.01;   // Exponential lambda (1/seconds)
static double WEIBULL_SHAPE = 1.5;       // Weibull k
static double WEIBULL_SCALE = 100.0;     // Weibull lambda (seconds)
static double SYSTEM_RECOVER_TIME = 20.0;
static double COMPUTE_JITTER = 0.0;      // Computation jitter stddev
static unsigned RNG_SEED = 42;
static std::string PLATFORM_PATH = "platform/platform.xml";

struct StatsPayload {
  double runtime;
  double compute_time;
  double full_recovery_time;
  double partial_recovery_time;
  int compute_steps;
};

static std::vector<int> SHARED_SHARD_ASSIGN;
static std::vector<int> SHARED_FAIL_FLAGS;
static std::vector<int> SHARED_CKPT_STEPS;
static int SHARED_CKPT_MAX = 0;
static std::vector<StatsPayload> SHARED_STATS;

// Simulate full system recovery and sync the latest checkpoint across workers.
// id/step identify the worker and step; rng/uni_dist drive failures; failure_cfg selects the model.
static void run_recovery(int id,
                         int step,
                         int world_size,
                         std::mt19937& rng,
                         std::uniform_real_distribution<double>& uni_dist,
                         double& next_failure_time,
                         const util::FailureParams& failure_cfg,
                         int& last_success_ckpt_step,
                         const s4u::BarrierPtr& recovery_barrier)
{
  if (id == 0) util::log_event(id, step, STEPS, "Control: Starting System-recovery...");
  double remaining = SYSTEM_RECOVER_TIME;
  while (remaining > 0.0) {
    double now = s4u::Engine::get_clock();
    double fail_time = util::pick_failure_time(rng, uni_dist, now, remaining, next_failure_time, failure_cfg);
    if (fail_time < 0.0) {
      s4u::this_actor::sleep_for(remaining);
      break;
    }
    double sleep_time = std::max(0.0, fail_time - now);
    if (sleep_time > 0.0) {
      s4u::this_actor::sleep_for(sleep_time);
    }
    util::log_event(id, step, STEPS, "Control: System-recovery-fail");
    util::reschedule_next_failure_time(rng, fail_time, next_failure_time, failure_cfg);
    remaining = SYSTEM_RECOVER_TIME;
  }

  SHARED_CKPT_STEPS[id] = last_success_ckpt_step;
  recovery_barrier->wait();
  if (id == 0) {
    int max_ckpt = last_success_ckpt_step;
    for (int w = 0; w < world_size; w++) {
      max_ckpt = std::max(max_ckpt, SHARED_CKPT_STEPS[w]);
    }
    SHARED_CKPT_MAX = max_ckpt;
  }
  recovery_barrier->wait();
  last_success_ckpt_step = SHARED_CKPT_MAX;
  if (id == 0) util::log_event(id, step, STEPS, "Control: System-recovery");
  if (id == 0) util::log_event(id, step, STEPS, "Communication: Recover-sync");
  recovery_barrier->wait();
}

// Worker actor: run DP steps with data assignment, compute, and recovery.
// id/world_size define the worker rank and total size; barriers synchronize phases.
void worker(int id,
            int world_size,
            const s4u::BarrierPtr& allreduce_barrier,
            const s4u::BarrierPtr& recovery_barrier,
            const s4u::BarrierPtr& final_barrier)
{
  auto host = s4u::this_actor::get_host();
  auto disk = host->get_disks().front();
  std::mt19937 rng(RNG_SEED + static_cast<unsigned>(id));
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  const util::FailureParams failure_cfg{FAILURE_DIST,
                                        FAILURE_PROB,
                                        EXP_FAILURE_RATE,
                                        WEIBULL_SHAPE,
                                        WEIBULL_SCALE};
  int last_success_ckpt_step = 0;
  // next_failure_time tracks the next scheduled failure for non-Bernoulli models.
  double next_failure_time = 0.0;
  std::vector<int> worker_fail_flags(world_size, 0); // Per-worker local fail flags
  int data_shard = -1;
  double start_time = s4u::Engine::get_clock();
  double compute_time = 0.0;
  double full_recovery_time = 0.0;
  double partial_recovery_time = 0.0;
  bool ever_failed = false;
  int compute_steps = 0;

  for (int step = 0; step < STEPS; step++) {

    /* ---- Data assignment ---- */
    if (id == 0) {
      std::vector<int> shard_order(world_size);
      std::iota(shard_order.begin(), shard_order.end(), 0);
      std::shuffle(shard_order.begin(), shard_order.end(), rng);
      for (int i = 0; i < world_size; i++) {
        SHARED_SHARD_ASSIGN[i] = shard_order[i];
      }
    }
    allreduce_barrier->wait();
    data_shard = SHARED_SHARD_ASSIGN[id];
    std::normal_distribution<double> jitter_dist(1.0, COMPUTE_JITTER);
    double jitter = std::max(0.1, jitter_dist(rng));
    s4u::this_actor::execute(CONT_COMPUTE_FLOPS * CPU_SCALE * jitter);
    util::log_event_with_shard(id, step, STEPS, data_shard, "Control: Data-assignment");

    /* ---- Data retrieval from disk  ---- */
    disk->read(DATA_SIZE);
    util::log_event_with_shard(id, step, STEPS, data_shard, "I/O: Data-read");

    /* ---- Compute: forward + backward ---- */
    std::normal_distribution<double> compute_jitter_dist(1.0, COMPUTE_JITTER);
    double compute_jitter = std::max(0.1, compute_jitter_dist(rng));
    double compute_flops = MODEL_COMPUTE_FLOPS * compute_jitter;
    double current_time = s4u::Engine::get_clock();

    // Failure check uses configured distribution; next_failure_time tracks scheduled failures.
    bool compute_failed = util::should_fail(rng, dist, current_time, next_failure_time, failure_cfg);
    bool worker_failed = compute_failed || worker_fail_flags[id];
    if (worker_fail_flags[id]) {
      util::log_event_with_shard(id, step, STEPS, data_shard,
                                 "Computation: Fwd+Bwd-fail due to pending fail");
    } else if (compute_failed) {
      worker_fail_flags[id] = 1;
      ever_failed = true;
      util::log_event_with_shard(id, step, STEPS, data_shard, "Computation: Fwd+Bwd-fail");
    } else {
      double exec_start = s4u::Engine::get_clock();
      s4u::this_actor::execute(compute_flops);
      compute_time += s4u::Engine::get_clock() - exec_start;
      compute_steps++;
      util::log_event_with_shard(id, step, STEPS, data_shard, "Computation: Fwd+Bwd");
    }

    /* ---- AllReduce (Ring-based, explicit reduce + broadcast) ---- */
    current_time = s4u::Engine::get_clock();
    bool allreduce_failed = util::should_fail(rng, dist, current_time, next_failure_time, failure_cfg);
    if (allreduce_failed) {
      worker_fail_flags[id] = 1;
      ever_failed = true;
      util::log_event(id, step, STEPS, "Communication: Allreduce-fail");
    }
    int local_fail_flag = (worker_failed || allreduce_failed) ? 1 : 0;
    SHARED_FAIL_FLAGS[id] = local_fail_flag;
    allreduce_barrier->wait();
    bool system_failed = false;
    for (int w = 0; w < world_size; w++) {
      if (SHARED_FAIL_FLAGS[w] != 0) {
        system_failed = true;
        break;
      }
    }

    bool fast_allreduce = system_failed;
    if (id == 0) util::log_event(id, step, STEPS, fast_allreduce ? "[Allreduce-fast]" : "[Allreduce-norm]");
    double allreduce_sleep = ALLREDUCE_TIME * (fast_allreduce ? ALLREDUCE_FAIL_SCALE : 1.0);
    if (allreduce_sleep > 0.0) {
      s4u::this_actor::sleep_for(allreduce_sleep);
    }

    allreduce_barrier->wait();

    if (id == 0) util::log_event(id, step, STEPS, "Communication: Gradient-sync");

    /* ---- Recover ---- */
    if (system_failed && SYSTEM_RECOVER_TIME > 0.0) {
      double rec_start = s4u::Engine::get_clock();
      run_recovery(id,
                   step,
                   world_size,
                   rng,
                   dist,
                   next_failure_time,
                   failure_cfg,
                   last_success_ckpt_step,
                   recovery_barrier);
      full_recovery_time += s4u::Engine::get_clock() - rec_start;
      worker_fail_flags[id] = 0;
      step = last_success_ckpt_step - 1;
      continue;
    }

    /* ---- Checkpoint ---- */
    if ((CHECKPOINT_INTERVAL > 0 && ((step + 1) % CHECKPOINT_INTERVAL == 0)) || (step + 1) == STEPS) {
      bool checkpoint_failed = false;
      if (id == 0) {
        current_time = s4u::Engine::get_clock();
        checkpoint_failed = util::should_fail(rng, dist, current_time, next_failure_time, failure_cfg);
        if (checkpoint_failed) {
          util::log_event(id, step, STEPS, "I/O: Checkpoint-fail");
        }
      }

      if (checkpoint_failed) {
        worker_fail_flags[id] = 1;
        ever_failed = true;
      } else if (id == 0) {
        disk->write(MODEL_SIZE);
        util::log_event(id, step, STEPS, "I/O: Checkpoint-saving");
        last_success_ckpt_step = step;
      }
      if (step + 1 == STEPS) {
        final_barrier->wait();
        if (id == 0) util::log_event(id, step, STEPS, "Communication: Final-sync");
      }
    }
  }

  double runtime = s4u::Engine::get_clock() - start_time;
  StatsPayload local_stats{runtime,
                           compute_time,
                           full_recovery_time,
                           partial_recovery_time,
                           compute_steps};

  SHARED_STATS[id] = local_stats;
  allreduce_barrier->wait();
  if (id == 0) {
    double total_runtime = SHARED_STATS[0].runtime;
    double total_full = SHARED_STATS[0].full_recovery_time;
    double total_partial = SHARED_STATS[0].partial_recovery_time;
    double compute_sum = (SHARED_STATS[0].compute_steps > 0) ? SHARED_STATS[0].compute_time : 0.0;
    int compute_count = (SHARED_STATS[0].compute_steps > 0) ? 1 : 0;
    for (int i = 1; i < world_size; i++) {
      const auto& stats = SHARED_STATS[i];
      total_runtime = std::max(total_runtime, stats.runtime);
      total_full = std::max(total_full, stats.full_recovery_time);
      total_partial = std::max(total_partial, stats.partial_recovery_time);
      if (stats.compute_steps > 0) {
        compute_sum += stats.compute_time;
        compute_count++;
      }
    }
    double down_time = total_full + total_partial;
    double up_time = std::max(0.0, total_runtime - down_time);
    double avg_compute = compute_count > 0 ? compute_sum / compute_count : 0.0;
    XBT_INFO("[worker%*d %.6f] Summary: total=%.6f up=%.6f down=%.6f full-recovery=%.6f "
             "partial-recovery=%.6f avg-compute=%.6f",
             util::worker_id_width(),
             0,
             s4u::Engine::get_clock(),
             total_runtime,
             up_time,
             down_time,
             total_full,
             total_partial,
             avg_compute);
    XBT_INFO("[worker%*d %.6f] Params: steps=%d ckpt=%d workers=%d compute=%.6g control=%.6g "
             "model=%.6g data=%.6g allreduce-time=%.6g allreduce-fail-scale=%.6g cpu-scale=%.6g "
             "fail-dist=%s fail-prob=%.6g exp-rate=%.6g weibull-shape=%.6g weibull-scale=%.6g "
             "recover=%.6g compute-jitter=%.6g seed=%u platform=%s",
             util::worker_id_width(),
             0,
             s4u::Engine::get_clock(),
             STEPS,
             CHECKPOINT_INTERVAL,
             world_size,
             MODEL_COMPUTE_FLOPS,
             CONT_COMPUTE_FLOPS,
             MODEL_SIZE,
             DATA_SIZE,
             ALLREDUCE_TIME,
             ALLREDUCE_FAIL_SCALE,
             CPU_SCALE,
             FAILURE_DIST.c_str(),
             FAILURE_PROB,
             EXP_FAILURE_RATE,
             WEIBULL_SHAPE,
             WEIBULL_SCALE,
             SYSTEM_RECOVER_TIME,
             COMPUTE_JITTER,
             RNG_SEED,
             PLATFORM_PATH.c_str());
  }
}

// Entry point: parse CLI args, configure the platform, and launch workers.
int main(int argc, char** argv)
{
  xbt_log_control_set("root.fmt:%m%n");
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg.rfind("--steps=", 0) == 0) {
      STEPS = std::max(1, std::atoi(arg.c_str() + 8));
    } else if (arg.rfind("--ckpt=", 0) == 0) {
      CHECKPOINT_INTERVAL = std::max(1, std::atoi(arg.c_str() + 7));
    } else if (arg.rfind("--compute=", 0) == 0) {
      MODEL_COMPUTE_FLOPS = std::max(0.0, std::atof(arg.c_str() + 10));
    } else if (arg.rfind("--control=", 0) == 0) {
      CONT_COMPUTE_FLOPS = std::max(0.0, std::atof(arg.c_str() + 10));
    } else if (arg.rfind("--model=", 0) == 0) {
      MODEL_SIZE = std::max(0.0, std::atof(arg.c_str() + 8));
    } else if (arg.rfind("--allreduce-fail-scale=", 0) == 0) {
      ALLREDUCE_FAIL_SCALE = std::clamp(std::atof(arg.c_str() + 24), 0.0, 1.0);
    } else if (arg.rfind("--allreduce-time=", 0) == 0) {
      ALLREDUCE_TIME = std::max(0.0, std::atof(arg.c_str() + 17));
    } else if (arg.rfind("--data=", 0) == 0) {
      DATA_SIZE = std::max(0.0, std::atof(arg.c_str() + 7));
    } else if (arg.rfind("--cpu-scale=", 0) == 0) {
      CPU_SCALE = std::max(0.0, std::atof(arg.c_str() + 12));
    } else if (arg.rfind("--fail-prob=", 0) == 0) {
      FAILURE_PROB = std::clamp(std::atof(arg.c_str() + 12), 0.0, 1.0);
    } else if (arg.rfind("--fail-dist=", 0) == 0) {
      FAILURE_DIST = arg.substr(12);
      if (FAILURE_DIST != "exponential" && FAILURE_DIST != "weibull" && FAILURE_DIST != "bernoulli") {
        XBT_WARN("Unknown --fail-dist=%s, fallback to bernoulli", FAILURE_DIST.c_str());
        FAILURE_DIST = "bernoulli";
      }
    } else if (arg.rfind("--exp-rate=", 0) == 0) {
      EXP_FAILURE_RATE = std::max(0.0, std::atof(arg.c_str() + 11));
    } else if (arg.rfind("--weibull-shape=", 0) == 0) {
      WEIBULL_SHAPE = std::max(0.0, std::atof(arg.c_str() + 16));
    } else if (arg.rfind("--weibull-scale=", 0) == 0) {
      WEIBULL_SCALE = std::max(0.0, std::atof(arg.c_str() + 16));
    } else if (arg.rfind("--recover=", 0) == 0) {
      SYSTEM_RECOVER_TIME = std::max(0.0, std::atof(arg.c_str() + 10));
    } else if (arg.rfind("--workers=", 0) == 0) {
      WORLD_SIZE = std::max(1, std::atoi(arg.c_str() + 10));
    } else if (arg.rfind("--compute-jitter=", 0) == 0) {
      COMPUTE_JITTER = std::max(0.0, std::atof(arg.c_str() + 17));
    } else if (arg.rfind("--seed=", 0) == 0) {
      RNG_SEED = static_cast<unsigned>(std::strtoul(arg.c_str() + 7, nullptr, 10));
    } else if (arg.rfind("--platform=", 0) == 0) {
      PLATFORM_PATH = arg.substr(11);
    } else {
      XBT_WARN("Ignoring unknown argument: %s", arg.c_str());
    }
  }
  if (RNG_SEED == 0) {
    RNG_SEED = util::seed_now();
  }

  s4u::Engine e(&argc, argv);
  e.load_platform(PLATFORM_PATH);

  auto hosts = e.get_all_hosts();
  std::sort(hosts.begin(), hosts.end(), [](const s4u::Host* a, const s4u::Host* b) {
    std::string key_a = util::host_locality_key(a->get_name());
    std::string key_b = util::host_locality_key(b->get_name());
    if (key_a == key_b)
      return a->get_name() < b->get_name();
    return key_a < key_b;
  });

  int world_size = std::min(WORLD_SIZE, static_cast<int>(hosts.size()));
  if (world_size < WORLD_SIZE) {
    XBT_WARN("Requested workers=%d but only %d hosts found", WORLD_SIZE, world_size);
  }
  util::set_worker_id_width(world_size);
  SHARED_SHARD_ASSIGN.assign(world_size, 0);
  SHARED_FAIL_FLAGS.assign(world_size, 0);
  SHARED_CKPT_STEPS.assign(world_size, 0);
  SHARED_STATS.resize(world_size);

  auto allreduce_barrier = s4u::Barrier::create(world_size);
  auto recovery_barrier = s4u::Barrier::create(world_size);
  auto final_barrier = s4u::Barrier::create(world_size);
  for (int i = 0; i < world_size; i++) {
    s4u::Actor::create(
        "worker" + std::to_string(i),
        hosts[i],
        worker,
        i,
        world_size,
        allreduce_barrier,
        recovery_barrier,
        final_barrier);
  }

  e.run();
  return 0;
}
