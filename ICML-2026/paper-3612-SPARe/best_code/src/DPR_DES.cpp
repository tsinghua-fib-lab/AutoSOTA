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

// Data Parallelism with Replication

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
static double PARTIAL_RECOVER_TIME = 2.0;
static double COMPUTE_JITTER = 0.0;      // Computation jitter stddev
static unsigned RNG_SEED = 42;
static unsigned REPLICATE_LEVEL = 2;
static double RECOVERY_DECISION_FLOPS = 5e6; // 50 TFLOPs
static std::string PLATFORM_PATH = "platform/platform.xml";

struct StatsPayload {
  double runtime;
  double compute_time;
  double full_recovery_time;
  double partial_recovery_time;
  int compute_steps;
};

// Per-worker communication state for current ring and recovery flags.
struct CommConfig {
  int active;
  int active_count;
  int prev;
  int next;
  int full_recovery;
  int partial_recovery;
  int leader_id;
};

static std::vector<int> SHARED_FAIL_FLAGS;
static std::vector<CommConfig> SHARED_COMM_CONFIG;
static std::vector<int> SHARED_ACTIVE_FLAGS;
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
                         const s4u::BarrierPtr& recovery_barrier,
                         int leader_id,
                         const std::vector<int>& full_recovery_failed)
{
  // Full system recovery: simulate time passage and sync latest checkpoint step.
  if (id == leader_id) {
    std::string failed_info = "failed:";
    for (int i = 0; i < world_size; i++) {
      if (full_recovery_failed[i] != 0) {
        failed_info += " w";
        failed_info += std::to_string(i);
      }
    }
    if (failed_info == "failed:") {
      failed_info += " none";
    }
    XBT_INFO("[worker%*d %.6f] step %03d/%d -- %-12s %s",
             util::worker_id_width(),
             id,
             s4u::Engine::get_clock(),
             step + 1,
             STEPS,
             "Control: Starting FULL system recovery due to",
             failed_info.c_str());
  }
  double remaining = SYSTEM_RECOVER_TIME;
  while (remaining > 0.0) {
    double now = s4u::Engine::get_clock();
    double fail_time = util::pick_failure_time(rng,
                                               uni_dist,
                                               now,
                                               remaining,
                                               next_failure_time,
                                               failure_cfg);
    if (fail_time < 0.0) {
      s4u::this_actor::sleep_for(remaining);
      break;
    }
    double sleep_time = std::max(0.0, fail_time - now);
    if (sleep_time > 0.0) {
      s4u::this_actor::sleep_for(sleep_time);
    }
    util::log_event(id, step, STEPS, "Control: Full-system-recovery-fail");
    util::reschedule_next_failure_time(rng, fail_time, next_failure_time, failure_cfg);
    remaining = SYSTEM_RECOVER_TIME;
  }

  SHARED_CKPT_STEPS[id] = last_success_ckpt_step;
  recovery_barrier->wait();
  if (id == leader_id) {
    int max_ckpt = last_success_ckpt_step;
    for (int w = 0; w < world_size; w++) {
      max_ckpt = std::max(max_ckpt, SHARED_CKPT_STEPS[w]);
    }
    SHARED_CKPT_MAX = max_ckpt;
  }
  recovery_barrier->wait();
  last_success_ckpt_step = SHARED_CKPT_MAX;
  if (id == leader_id) util::log_event(id, step, STEPS, "Control: Full-system-recovery");
  recovery_barrier->wait();
}

// Worker actor: run replicated DP steps with partial/full recovery handling.
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
  int data_shard = -1;
  int replicate_level = std::max(1, static_cast<int>(REPLICATE_LEVEL));
  int group_count = std::max(1, (world_size + replicate_level - 1) / replicate_level);
  std::vector<int> active_flags(world_size, 1);
  std::vector<int> removed_flags(world_size, 0);
  std::vector<int> fail_agg(world_size, 0);
  std::vector<int> full_recovery_failed(world_size, 0);
  std::vector<CommConfig> comm_config(world_size);
  bool pending_fail = false;
  CommConfig comm_state{1,
                        world_size,
                        (id - 1 + world_size) % world_size,
                        (id + 1) % world_size,
                        0,
                        0,
                        0};
  double start_time = s4u::Engine::get_clock();
  double compute_time = 0.0;
  double full_recovery_time = 0.0;
  double partial_recovery_time = 0.0;
  bool ever_failed = false;
  int compute_steps = 0;

  for (int step = 0; step < STEPS; step++) {
    int leader_id = comm_state.leader_id;

    /* ---- Data assignment ---- */
    data_shard = id / replicate_level;
    std::normal_distribution<double> jitter_dist(1.0, COMPUTE_JITTER);
    double jitter = std::max(0.1, jitter_dist(rng));
    s4u::this_actor::execute(CONT_COMPUTE_FLOPS * CPU_SCALE * jitter);
    if (id == leader_id) {
      util::log_event(id, step, STEPS, "Control: Data-assignment");
    }

    /* ---- Data retrieval from disk  ---- */
    if (comm_state.active) {
      disk->read(DATA_SIZE);
      if (id == leader_id)  util::log_event_with_shard(id, step, STEPS, data_shard, "I/O: Data-read");
    }

    /* ---- Compute: forward + backward ---- */
    std::normal_distribution<double> compute_jitter_dist(1.0, COMPUTE_JITTER);
    double compute_jitter = std::max(0.1, compute_jitter_dist(rng));
    double compute_flops = MODEL_COMPUTE_FLOPS * compute_jitter;
    double current_time = s4u::Engine::get_clock();

    // Failure check uses configured distribution; next_failure_time tracks scheduled failures.
    bool compute_failed = comm_state.active &&
                          util::should_fail(rng, dist, current_time, next_failure_time, failure_cfg);
    bool worker_failed = compute_failed || pending_fail;
    if (pending_fail) {
      util::log_event_with_shard(id, step, STEPS, data_shard,
                                 "Computation: Fwd+Bwd-fail due to pending fail");
    } else if (compute_failed) {
      pending_fail = true;
      ever_failed = true;
      util::log_event_with_shard(id, step, STEPS, data_shard, "Computation: Fwd+Bwd-fail");
    } else {
      double exec_start = s4u::Engine::get_clock();
      s4u::this_actor::execute(compute_flops);
      compute_time += s4u::Engine::get_clock() - exec_start;
      compute_steps++;
      if (id == leader_id) util::log_event_with_shard(id, step, STEPS, data_shard, "Computation: Fwd+Bwd");
    }

    /* ---- AllReduce (Ring-based, explicit reduce + broadcast) ---- */
    current_time = s4u::Engine::get_clock();
    bool allreduce_failed = false;
    if (comm_state.active) {
      allreduce_failed = util::should_fail(rng, dist, current_time, next_failure_time, failure_cfg);
      if (allreduce_failed) {
        pending_fail = true;
        ever_failed = true;
        util::log_event(id, step, STEPS, "Communication: Allreduce-fail");
      }
    }
    int local_fail_flag = (comm_state.active && (worker_failed || allreduce_failed)) ? 1 : 0;
    std::fill(fail_agg.begin(), fail_agg.end(), 0);
    if (comm_state.active) {
      fail_agg[id] = local_fail_flag;
    } else {
      local_fail_flag = 0;
    }
    SHARED_FAIL_FLAGS[id] = local_fail_flag;
    allreduce_barrier->wait();

    bool system_failed = false;
    for (int w = 0; w < world_size; w++) {
      fail_agg[w] = SHARED_FAIL_FLAGS[w];
      if (fail_agg[w] != 0) {
        system_failed = true;
        break;
      }
    }

    bool fast_allreduce = comm_state.active && system_failed;
    if (id == leader_id) util::log_event(id, step, STEPS, fast_allreduce ? "[Allreduce-fast]" : "[Allreduce-norm]");
    double allreduce_sleep = ALLREDUCE_TIME * (fast_allreduce ? ALLREDUCE_FAIL_SCALE : 1.0);
    if (comm_state.active && allreduce_sleep > 0.0) {
      s4u::this_actor::sleep_for(allreduce_sleep);
    }


    allreduce_barrier->wait();

    if (id == leader_id) util::log_event(id, step, STEPS, "Communication: Gradient-sync");

    /* ---- Recovery decision ---- */
    if (system_failed && id == leader_id) {
      double recover_decision_flops = RECOVERY_DECISION_FLOPS * compute_jitter * CPU_SCALE;
      s4u::this_actor::execute(recover_decision_flops);
      util::log_event(id, step, STEPS, "Control: Recovery-decision");
    }

    bool do_full_recovery = false;
    bool do_partial_recovery = false;
    if (id == leader_id) { // Test if need full recovery
      if (system_failed) {
        bool replication_ok = true;
        for (int shard = 0; shard < group_count; shard++) { // Test each data shard
          bool shard_ok = false;
          for (int i = 0; i < world_size; i++) {
            if (active_flags[i] && fail_agg[i] == 0 && (i / replicate_level) == shard) {
              shard_ok = true;
              break;
            }
          }
          if (!shard_ok) { 
            replication_ok = false;
            break;
          }
        }

        if (replication_ok) {
          do_partial_recovery = true;
          std::string total_affected = "total affected:";
          std::string round_affected = "this round appended:";
          for (int i = 0; i < world_size; i++) {
            if (active_flags[i] && fail_agg[i] != 0) {
              active_flags[i] = 0;
              removed_flags[i] = 1;
              round_affected += " w";
              round_affected += std::to_string(i);
            }
          }
          for (int i = 0; i < world_size; i++) {
            if (removed_flags[i]) {
              total_affected += " w";
              total_affected += std::to_string(i);
            }
          }
          XBT_INFO("[worker%*d %.6f] step %03d/%d -- %-12s %s %s",
                   util::worker_id_width(),
                   id,
                   s4u::Engine::get_clock(),
                   step + 1,
                   STEPS,
                   "Control: Starting PARTIAL system recovery due to",
                   total_affected.c_str(),
                   round_affected.c_str());
        } else {
          do_full_recovery = true;
          // Snapshot all failed workers for the full-recovery log before resetting state.
          std::fill(full_recovery_failed.begin(), full_recovery_failed.end(), 0);
          for (int i = 0; i < world_size; i++) {
            if (fail_agg[i] != 0 || removed_flags[i] != 0) {
              full_recovery_failed[i] = 1;
            }
          }
          std::fill(active_flags.begin(), active_flags.end(), 1);
        }
      }

      std::vector<int> active_workers;
      active_workers.reserve(world_size);
      for (int i = 0; i < world_size; i++) {
        if (active_flags[i]) active_workers.push_back(i);
      }
      int active_count = static_cast<int>(active_workers.size());
      int new_leader_id = active_workers.empty() ? leader_id : active_workers.front();
      if (do_full_recovery) new_leader_id = leader_id;

      for (int i = 0; i < world_size; i++) { // Change the system config
        CommConfig cfg{};
        cfg.active = active_flags[i] ? 1 : 0;
        cfg.active_count = active_count;
        cfg.full_recovery = do_full_recovery ? 1 : 0;
        cfg.partial_recovery = do_partial_recovery ? 1 : 0;
        cfg.leader_id = new_leader_id;
        if (cfg.active && active_count > 0) {
          auto it = std::find(active_workers.begin(), active_workers.end(), i);
          int rank = static_cast<int>(std::distance(active_workers.begin(), it));
          int prev_index = (rank - 1 + active_count) % active_count;
          int next_index = (rank + 1) % active_count;
          cfg.prev = active_workers[prev_index];
          cfg.next = active_workers[next_index];
        } else {
          cfg.prev = -1;
          cfg.next = -1;
        }
        comm_config[i] = cfg;
        SHARED_COMM_CONFIG[i] = cfg;
        if (i == leader_id) comm_state = comm_config[i];
      }

      SHARED_ACTIVE_FLAGS = active_flags;
    }

    allreduce_barrier->wait();

    comm_state = SHARED_COMM_CONFIG[id];
    if (comm_state.leader_id == id && leader_id != id) {
      active_flags = SHARED_ACTIVE_FLAGS;
      for (int i = 0; i < world_size; i++) {
        removed_flags[i] = active_flags[i] ? 0 : 1;
      }
    }

    // Only emit Go-ahead when the system proceeds without any failure in this step.
    if (comm_state.partial_recovery && comm_state.active && PARTIAL_RECOVER_TIME > 0.0) {
      double rec_start = s4u::Engine::get_clock();
      s4u::this_actor::sleep_for(PARTIAL_RECOVER_TIME); // Partial recovery time
      partial_recovery_time += s4u::Engine::get_clock() - rec_start;
      if (id == leader_id) {
        util::log_event(id, step, STEPS, "Control: Partial Recovery done");
      }
    } else if (id == leader_id && !system_failed) {
      XBT_INFO("[worker%*d %.6f] step %03d/%d -- [Go-ahead active %d/%d]",
               util::worker_id_width(),
               id,
               s4u::Engine::get_clock(),
               step + 1,
               STEPS,
               comm_state.active_count,
               world_size);
    }

    allreduce_barrier->wait();

    /* ---- Recover ---- */
    if (comm_state.full_recovery && SYSTEM_RECOVER_TIME > 0.0) {
      double rec_start = s4u::Engine::get_clock();
      run_recovery(id,
                   step,
                   world_size,
                   rng,
                   dist,
                   next_failure_time,
                   failure_cfg,
                   last_success_ckpt_step,
                   recovery_barrier,
                   comm_state.leader_id,
                   full_recovery_failed);
      full_recovery_time += s4u::Engine::get_clock() - rec_start;
      pending_fail = false;
      comm_state.active = 1;
      comm_state.active_count = world_size;
      comm_state.prev = (id - 1 + world_size) % world_size;
      comm_state.next = (id + 1) % world_size;
      comm_state.leader_id = 0;
      comm_state.full_recovery = 0;
      comm_state.partial_recovery = 0;
      std::fill(active_flags.begin(), active_flags.end(), 1);
      std::fill(removed_flags.begin(), removed_flags.end(), 0);
      step = last_success_ckpt_step - 1;
      continue;
    }

    /* ---- Checkpoint ---- */
    if ((CHECKPOINT_INTERVAL > 0 && ((step + 1) % CHECKPOINT_INTERVAL == 0)) || (step + 1) == STEPS) {
      bool checkpoint_failed = false;
      if (id == leader_id) {
        current_time = s4u::Engine::get_clock();
        checkpoint_failed = util::should_fail(rng, dist, current_time, next_failure_time, failure_cfg);
        if (checkpoint_failed) {
          util::log_event(id, step, STEPS, "I/O: Checkpoint-fail");
        }
      }

      if (checkpoint_failed) {
        pending_fail = true;
        ever_failed = true;
      } else if (id == leader_id) {
        disk->write(MODEL_SIZE);
        util::log_event(id, step, STEPS, "I/O: Checkpoint-saving");
        last_success_ckpt_step = step;
      }
      if (step + 1 == STEPS) {
        final_barrier->wait();
        if (id == leader_id) util::log_event(id, step, STEPS, "Communication: Final-sync");
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
    XBT_INFO("[worker%*d %.6f] Params: steps=%d ckpt=%d workers=%d replicate=%u compute=%.6g "
             "control=%.6g decision=%.6g model=%.6g data=%.6g allreduce-time=%.6g "
             "allreduce-fail-scale=%.6g cpu-scale=%.6g fail-dist=%s fail-prob=%.6g exp-rate=%.6g "
             "weibull-shape=%.6g weibull-scale=%.6g recover=%.6g partial-recover=%.6g "
             "compute-jitter=%.6g seed=%u platform=%s",
             util::worker_id_width(),
             0,
             s4u::Engine::get_clock(),
             STEPS,
             CHECKPOINT_INTERVAL,
             world_size,
             REPLICATE_LEVEL,
             MODEL_COMPUTE_FLOPS,
             CONT_COMPUTE_FLOPS,
             RECOVERY_DECISION_FLOPS,
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
             PARTIAL_RECOVER_TIME,
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
    } else if (arg.rfind("--decision=", 0) == 0) {
      RECOVERY_DECISION_FLOPS = std::max(0.0, std::atof(arg.c_str() + 10));
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
    } else if (arg.rfind("--partial-recover-time=", 0) == 0) {
      PARTIAL_RECOVER_TIME = std::max(0.0, std::atof(arg.c_str() + 23));
    } else if (arg.rfind("--workers=", 0) == 0) {
      WORLD_SIZE = std::max(1, std::atoi(arg.c_str() + 10));
    } else if (arg.rfind("--compute-jitter=", 0) == 0) {
      COMPUTE_JITTER = std::max(0.0, std::atof(arg.c_str() + 17));
    } else if (arg.rfind("--seed=", 0) == 0) {
      RNG_SEED = static_cast<unsigned>(std::strtoul(arg.c_str() + 7, nullptr, 10));
    } else if (arg.rfind("--replicate_level=", 0) == 0) {
      REPLICATE_LEVEL = std::max(1u, static_cast<unsigned>(std::strtoul(arg.c_str() + 18, nullptr, 10)));
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
  SHARED_FAIL_FLAGS.assign(world_size, 0);
  SHARED_COMM_CONFIG.assign(world_size, CommConfig{});
  SHARED_ACTIVE_FLAGS.assign(world_size, 1);
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
