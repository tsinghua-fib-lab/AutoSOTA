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

// SPARe with resequencing (paper-aligned: ONE all-reduce after S_A stacks)

#include <simgrid/s4u.hpp>

#include "utility.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <vector>

using namespace simgrid;

XBT_LOG_NEW_DEFAULT_CATEGORY(ordp_des, "ORDP DES log");

static int WORLD_SIZE = 4;
static int STEPS = 10;
static int CHECKPOINT_INTERVAL = 2;
static double MODEL_COMPUTE_FLOPS = 5e13; // 50 TFLOPs
static double CONT_COMPUTE_FLOPS = 5e6;
static double MODEL_SIZE = 4e9; // 4 GB
static double ALLREDUCE_FAIL_SCALE = 1.0;
static double ALLREDUCE_TIME = 0.0; // Seconds per allreduce (constant)
static double DATA_SIZE = 4e6;      // 4 MB
static double CPU_SCALE = 1;        // CPU scale down
static double FAILURE_PROB = 0.1;
static std::string FAILURE_DIST = "exponential";
static double EXP_FAILURE_RATE = 0.01; // Exponential lambda (1/seconds)
static double WEIBULL_SHAPE = 1.5;     // Weibull k
static double WEIBULL_SCALE = 100.0;   // Weibull lambda (seconds)
static double SYSTEM_RECOVER_TIME = 20.0;
static double PARTIAL_RECOVER_TIME = 2.0;
static double COMPUTE_JITTER = 0.0; // Computation jitter stddev
static unsigned RNG_SEED = 42;
static unsigned REPLICATE_LEVEL = 2;
static double RECOVERY_DECISION_FLOPS = 5e6;
static std::string PLATFORM_PATH = "platform/platform.xml";

struct StatsPayload {
  double runtime;
  double compute_time;
  double full_recovery_time;
  double partial_recovery_time;
  int compute_steps;
};

// Precomputed Golomb ruler offsets for replicate levels 1..28.
// FIXED: remove out-of-bounds writes in original initializer.
static const std::vector<std::vector<int>> RULER_TABLE = []() {
  std::vector<std::vector<int>> table(28);
  for (int i = 1; i <= (int)table.size(); ++i) {
    switch (i) {
      case 1:  table[0]  = {0}; break;
      case 2:  table[1]  = {0, 1}; break;
      case 3:  table[2]  = {0, 1, 3}; break;
      case 4:  table[3]  = {0, 1, 4, 6}; break;
      case 5:  table[4]  = {0, 1, 4, 9, 11}; break;
      case 6:  table[5]  = {0, 1, 4, 10, 12, 17}; break;
      case 7:  table[6]  = {0, 1, 4, 10, 18, 23, 25}; break;
      case 8:  table[7]  = {0, 1, 4, 9, 15, 22, 32, 34}; break;
      case 9:  table[8]  = {0, 1, 5, 12, 25, 27, 35, 41, 44}; break;
      case 10: table[9]  = {0, 1, 6, 10, 23, 26, 34, 41, 53, 55}; break;
      case 11: table[10] = {0, 1, 4, 13, 28, 33, 47, 54, 64, 70, 72}; break;
      case 12: table[11] = {0, 2, 6, 24, 29, 40, 43, 55, 68, 75, 76, 85}; break;
      case 13: table[12] = {0, 2, 5, 25, 37, 43, 59, 70, 85, 89, 98, 99, 106}; break;
      case 14: table[13] = {0, 4, 6, 20, 35, 52, 59, 77, 78, 86, 89, 99, 122, 127}; break;
      case 15: table[14] = {0, 4, 20, 30, 57, 59, 62, 76, 100, 111, 123, 136, 144, 145, 151}; break;
      case 16: table[15] = {0, 1, 4, 11, 26, 32, 56, 68, 76, 115, 117, 134, 150, 163, 168, 177}; break;
      case 17: table[16] = {0, 5, 7, 17, 52, 56, 67, 80, 81, 100, 122, 138, 159, 165, 168, 191, 199}; break;
      case 18: table[17] = {0, 2, 10, 22, 53, 56, 82, 83, 89, 98, 130, 148, 153, 167, 188, 192, 205, 216}; break;
      case 19: table[18] = {0, 1, 6, 25, 32, 72, 100, 108, 120, 130, 153, 169, 187, 190, 204, 231, 233, 242, 246}; break;
      case 20: table[19] = {0, 1, 8, 11, 68, 77, 94, 116, 121, 156, 158, 179, 194, 208, 212, 228, 240, 253, 259, 283}; break;
      case 21: table[20] = {0, 2, 24, 56, 77, 82, 83, 95, 129, 144, 179, 186, 195, 255, 265, 285, 293, 296, 310, 329, 333}; break;
      case 22: table[21] = {0, 1, 9, 14, 43, 70, 106, 122, 124, 128, 159, 179, 204, 223, 253, 263, 270, 291, 330, 341, 353, 356}; break;
      case 23: table[22] = {0, 3, 7, 17, 61, 66, 91, 99, 114, 159, 171, 199, 200, 226, 235, 246, 277, 316, 329, 348, 350, 366, 372}; break;
      case 24: table[23] = {0, 9, 33, 37, 38, 97, 122, 129, 140, 142, 152, 191, 205, 208, 252, 278, 286, 326, 332, 353, 368, 384, 403, 425}; break;
      case 25: table[24] = {0, 12, 29, 39, 72, 91, 146, 157, 160, 161, 166, 191, 207, 214, 258, 290, 316, 354, 372, 394, 396, 431, 459, 467, 480}; break;
      case 26: table[25] = {0, 1, 33, 83, 104, 110, 124, 163, 185, 200, 203, 249, 251, 258, 314, 318, 343, 356, 386, 430, 440, 456, 464, 475, 487, 492}; break;
      case 27: table[26] = {0, 3, 15, 41, 66, 95, 97, 106, 142, 152, 220, 221, 225, 242, 295, 330, 338, 354, 382, 388, 402, 415, 486, 504, 523, 546, 553}; break;
      case 28: table[27] = {0, 3, 15, 41, 66, 95, 97, 106, 142, 152, 220, 221, 225, 242, 295, 330, 338, 354, 382, 388, 402, 415, 486, 504, 523, 546, 553, 585}; break;
      default: break;
    }
  }
  return table;
}();

// Logging helpers (unchanged)
static void log_event_with_data_id(int id, int step, int data_id, const char* label)
{
  XBT_INFO("[worker%*d %.6f] step %03d/%d -- %-12s [data d%d]",
           util::worker_id_width(), id, s4u::Engine::get_clock(), step + 1, STEPS, label, data_id);
}

static void log_event_with_stack_depth(int id, int step, int current_stack, int total_stack)
{
  XBT_INFO("[worker%*d %.6f] step %03d/%d -- [Stack-depth %d/%d]",
           util::worker_id_width(), id, s4u::Engine::get_clock(), step + 1, STEPS, current_stack, total_stack);
}

static void log_event_with_recompute_stack(int id, int step, int current_stack, int total_stack)
{
  XBT_INFO("[worker%*d %.6f] step %03d/%d -- [Stack %d/%d Recompute]",
           util::worker_id_width(), id, s4u::Engine::get_clock(), step + 1, STEPS, current_stack, total_stack);
}

// Hopcroft-Karp matching (unchanged)
static int hopcroft_karp(const std::vector<std::vector<int>>& adj, int n_left, int n_right)
{
  const int INF = std::numeric_limits<int>::max() / 4;
  std::vector<int> pair_u(n_left, -1);
  std::vector<int> pair_v(n_right, -1);
  std::vector<int> dist(n_left, INF);

  auto bfs = [&]() -> bool {
    std::queue<int> q;
    bool found_free_right = false;
    for (int u = 0; u < n_left; u++) {
      if (pair_u[u] == -1) {
        dist[u] = 0;
        q.push(u);
      } else {
        dist[u] = INF;
      }
    }
    while (!q.empty()) {
      int u = q.front();
      q.pop();
      for (int v : adj[u]) {
        int u2 = pair_v[v];
        if (u2 == -1) {
          found_free_right = true;
        } else if (dist[u2] == INF) {
          dist[u2] = dist[u] + 1;
          q.push(u2);
        }
      }
    }
    return found_free_right;
  };

  auto dfs = [&](int u, const auto& self) -> bool {
    for (int v : adj[u]) {
      int u2 = pair_v[v];
      if (u2 == -1 || (dist[u2] == dist[u] + 1 && self(u2, self))) {
        pair_u[u] = v;
        pair_v[v] = u;
        return true;
      }
    }
    dist[u] = INF;
    return false;
  };

  int matching = 0;
  while (bfs()) {
    for (int u = 0; u < n_left; u++) {
      if (pair_u[u] == -1) {
        if (dfs(u, dfs)) matching++;
      }
    }
  }
  return matching;
}

// Min-cost max-flow (SPFA) (unchanged)
struct MinCostEdge {
  int to;
  int rev;
  int cap;
  int cost;
};

static void add_min_cost_edge(std::vector<std::vector<MinCostEdge>>& graph,
                              int from, int to, int cap, int cost)
{
  graph[from].push_back(MinCostEdge{to, static_cast<int>(graph[to].size()), cap, cost});
  graph[to].push_back(MinCostEdge{from, static_cast<int>(graph[from].size()) - 1, 0, -cost});
}

static int min_cost_max_flow(std::vector<std::vector<MinCostEdge>>& graph,
                             int source, int sink, int max_flow, int& total_cost)
{
  const int INF = std::numeric_limits<int>::max() / 4;
  int flow = 0;
  total_cost = 0;
  int n = static_cast<int>(graph.size());
  while (flow < max_flow) {
    std::vector<int> dist(n, INF);
    std::vector<int> prev_v(n, -1);
    std::vector<int> prev_e(n, -1);
    std::vector<int> in_queue(n, 0);
    std::queue<int> q;
    dist[source] = 0;
    q.push(source);
    in_queue[source] = 1;
    while (!q.empty()) {
      int v = q.front();
      q.pop();
      in_queue[v] = 0;
      for (int i = 0; i < (int)graph[v].size(); i++) {
        const auto& e = graph[v][i];
        if (e.cap <= 0) continue;
        int to = e.to;
        int ndist = dist[v] + e.cost;
        if (ndist < dist[to]) {
          dist[to] = ndist;
          prev_v[to] = v;
          prev_e[to] = i;
          if (!in_queue[to]) {
            q.push(to);
            in_queue[to] = 1;
          }
        }
      }
    }
    if (dist[sink] == INF) break;
    int add = max_flow - flow;
    for (int v = sink; v != source; v = prev_v[v]) {
      add = std::min(add, graph[prev_v[v]][prev_e[v]].cap);
    }
    for (int v = sink; v != source; v = prev_v[v]) {
      auto& e = graph[prev_v[v]][prev_e[v]];
      e.cap -= add;
      graph[v][e.rev].cap += add;
      total_cost += add * e.cost;
    }
    flow += add;
  }
  return flow;
}

// ------------------------------
// SPARe controller (paper-aligned, ONE all-reduce after S_A stacks)
// ------------------------------

// Phase 0: fixed stacks on survivors cover all types within current S_A (coverage test).
static bool fixed_valid_global(const std::vector<std::vector<int>>& seq,
                               const std::vector<int>& survivors,
                               int SA,
                               int world_size)
{
  std::vector<int> seen(world_size, 0);
  int covered = 0;
  for (int w : survivors) {
    int upto = std::min(SA, (int)seq[w].size());
    for (int p = 0; p < upto; ++p) {
      int data_id = seq[w][p];
      if (data_id >= 0 && data_id < world_size && !seen[data_id]) {
        seen[data_id] = 1;
        if (++covered >= world_size) return true;
      }
    }
  }
  return covered >= world_size;
}

// HK-FREE feasibility: do we have enough slots among survivors at depth S, given hosting relation data_workers?
static bool feasible_free_global(const std::vector<std::vector<int>>& data_workers,
                                 const std::vector<int>& survivors,
                                 int S,
                                 int world_size)
{
  if (world_size == 0) return true;
  if (survivors.empty()) return false;
  int n_left = world_size;                 // types 0..world_size-1
  int n_right = (int)survivors.size() * S; // slots (survivor_idx, pos)
  if (n_right < n_left) return false;

  std::vector<int> worker_index(world_size, -1);
  for (size_t i = 0; i < survivors.size(); ++i) {
    worker_index[survivors[i]] = (int)i;
  }

  std::vector<std::vector<int>> adj(n_left);
  for (int type = 0; type < world_size; ++type) {
    for (int w : data_workers[type]) {
      int wi = (w >= 0 && w < world_size) ? worker_index[w] : -1;
      if (wi < 0) continue; // not a survivor
      int base = wi * S;
      for (int p = 0; p < S; ++p) {
        adj[type].push_back(base + p);
      }
    }
  }

  return hopcroft_karp(adj, n_left, n_right) == n_left;
}

// Find minimal S_A in [current_SA..max_depth] (monotone under shrinking; still exact).
static int find_minimal_SA(const std::vector<std::vector<int>>& data_workers,
                           const std::vector<int>& survivors,
                           int current_SA,
                           int max_depth,
                           int world_size)
{
  int start = std::max(1, current_SA);
  for (int S = start; S <= max_depth; ++S) {
    if (feasible_free_global(data_workers, survivors, S, world_size)) return S;
  }
  return -1;
}

// Phase 2: min-cost assignment of each type to one (worker, pos<SA) slot; cost=0 if unchanged.
static bool min_cost_assignment_global(const std::vector<std::vector<int>>& data_workers,
                                       const std::vector<std::vector<int>>& seq,
                                       const std::vector<int>& survivors,
                                       int SA,
                                       int world_size,
                                       std::vector<std::pair<int, int>>& assignment_out)
{
  assignment_out.clear();
  if (world_size == 0) return true;
  if ((int)survivors.size() * SA < world_size) return false;

  std::vector<int> worker_index(world_size, -1);
  for (size_t i = 0; i < survivors.size(); ++i) {
    worker_index[survivors[i]] = (int)i;
  }

  // Slots are enumerated as: slot_idx = survivor_idx * SA + pos, mapped to (w, pos).
  std::vector<std::pair<int, int>> slots;
  slots.reserve(survivors.size() * SA);
  for (int w : survivors) {
    for (int p = 0; p < SA; ++p) {
      slots.emplace_back(w, p);
    }
  }

  int n_left = world_size;         // types
  int n_right = (int)slots.size(); // slots

  int source = 0;
  int left_start = 1;
  int right_start = left_start + n_left;
  int sink = right_start + n_right;

  std::vector<std::vector<MinCostEdge>> graph(sink + 1);

  for (int i = 0; i < n_left; ++i) add_min_cost_edge(graph, source, left_start + i, 1, 0);
  for (int j = 0; j < n_right; ++j) add_min_cost_edge(graph, right_start + j, sink, 1, 0);

  for (int type = 0; type < world_size; ++type) {
    for (int w : data_workers[type]) {
      int wi = (w >= 0 && w < world_size) ? worker_index[w] : -1;
      if (wi < 0) continue;
      int base = wi * SA;
      for (int p = 0; p < SA; ++p) {
        int slot_idx = base + p;
        int cost = (p < (int)seq[w].size() && seq[w][p] == type) ? 0 : 1;
        add_min_cost_edge(graph, left_start + type, right_start + slot_idx, 1, cost);
      }
    }
  }

  int total_cost = 0;
  int flow = min_cost_max_flow(graph, source, sink, n_left, total_cost);
  if (flow != n_left) return false;

  assignment_out.assign(n_left, {-1, -1}); // type -> (worker, pos)
  for (int type = 0; type < n_left; ++type) {
    int node = left_start + type;
    for (const auto& e : graph[node]) {
      if (e.to >= right_start && e.to < right_start + n_right && e.cap == 0) {
        int slot_idx = e.to - right_start;
        assignment_out[type] = slots[slot_idx];
        break;
      }
    }
    if (assignment_out[type].first < 0) return false;
  }

  return true;
}

// Apply assignment: for each survivor w, set its first SA positions to the types assigned to (w,pos),
// then fill the rest with remaining hosted types in original order to preserve permutation.
static void apply_assignment_global(const std::vector<int>& survivors,
                                   const std::vector<std::pair<int, int>>& assignment_by_type,
                                   int SA,
                                   int replicate_level,
                                   int world_size,
                                   std::vector<std::vector<int>>& seq)
{
  // Build per-worker prefix of size SA initialized to -1.
  std::vector<std::vector<int>> new_prefix(world_size, std::vector<int>(replicate_level, -1));
  std::vector<std::vector<unsigned char>> used(world_size, std::vector<unsigned char>(world_size, 0));

  // Place assigned types.
  for (int type = 0; type < world_size; ++type) {
    int w = assignment_by_type[type].first;
    int p = assignment_by_type[type].second;
    xbt_assert(p >= 0 && p < SA, "Assignment pos out of range");
    xbt_assert(w >= 0 && w < world_size, "Assignment worker out of range");
    // Ensure type actually exists in w's hosted set.
    bool hosted = false;
    for (int x : seq[w]) if (x == type) { hosted = true; break; }
    xbt_assert(hosted, "Assignment illegal: worker %d does not host type %d", w, type);

    xbt_assert(new_prefix[w][p] == -1, "Slot collision in assignment (should not happen)");
    new_prefix[w][p] = type;
    used[w][type] = 1;
  }

  // For each survivor, fill any unfilled prefix positions (rare) and then fill suffix by original order.
  for (int w : survivors) {
    const auto& old = seq[w];
    std::vector<int> out(replicate_level, -1);

    // First SA positions from assignment (should be fully filled if assignment is complete,
    // but a worker may have fewer than SA assigned types; then remaining prefix spots stay -1 and must be filled
    // with other hosted types to maintain a permutation).
    for (int p = 0; p < SA; ++p) out[p] = new_prefix[w][p];

    // Fill prefix holes with any unused hosted types in original order.
    for (int p = 0; p < SA; ++p) {
      if (out[p] != -1) continue;
      for (int x : old) {
        if (!used[w][x]) {
          out[p] = x;
          used[w][x] = 1;
          break;
        }
      }
      xbt_assert(out[p] != -1, "Failed to fill prefix hole for worker %d", w);
    }

    // Fill remaining positions with unused hosted types in original order.
    int idx = SA;
    for (int x : old) {
      if (!used[w][x]) {
        xbt_assert(idx < replicate_level, "Permutation overflow for worker %d", w);
        out[idx++] = x;
        used[w][x] = 1;
      }
    }
    xbt_assert(idx == replicate_level, "Permutation not completed for worker %d", w);

    seq[w] = std::move(out);
  }
}

// Per-worker communication state
struct CommConfig {
  int active;
  int active_count;
  int prev;
  int next;
  int full_recovery;
  int partial_recovery;
  int leader_id;
  int stack_depth; // S_A
  int seq_updated;
  int recompute_updated;
  int step_done;
};

static std::vector<int> SHARED_FAIL_FLAGS;
static std::vector<CommConfig> SHARED_COMM_CONFIG;
static std::vector<int> SHARED_ACTIVE_FLAGS;
static std::vector<int> SHARED_SEQ_SNAPSHOT;
static std::vector<int> SHARED_RECOMPUTE_START;
static std::vector<int> SHARED_RECOMPUTE_END;
static std::vector<int> SHARED_CKPT_STEPS;
static int SHARED_CKPT_MAX = 0;
static std::vector<StatsPayload> SHARED_STATS;

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
             "[Starting FULL system recovery due to",
             failed_info.c_str(), "]");
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

void worker(int id,
            int world_size,
            const s4u::BarrierPtr& allreduce_barrier,
            const s4u::BarrierPtr& recovery_barrier,
            const s4u::BarrierPtr& checkpoint_barrier)
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
  double next_failure_time = 0.0;

  int replicate_level = std::max(1, static_cast<int>(REPLICATE_LEVEL));
  const auto& ruler_offsets = RULER_TABLE[replicate_level - 1];

  auto mod_index = [world_size](int value) {
    int result = value % world_size;
    return result < 0 ? result + world_size : result;
  };

  // worker_data[w] is w's stack permutation (length = replicate_level)
  std::vector<std::vector<int>> worker_data(world_size);
  // data_workers[type] are workers that host this type
  std::vector<std::vector<int>> data_workers(world_size);

  for (int w = 0; w < world_size; w++) {
    worker_data[w].resize(ruler_offsets.size());
    for (size_t idx = 0; idx < ruler_offsets.size(); idx++)
      worker_data[w][idx] = mod_index(w + ruler_offsets[idx]);
  }
  for (int data_id = 0; data_id < world_size; data_id++) {
    data_workers[data_id].reserve(ruler_offsets.size());
    for (int offset : ruler_offsets)
      data_workers[data_id].push_back(mod_index(data_id - offset));
  }

  std::vector<std::vector<int>> base_worker_data = worker_data;

  int stack_depth = 1; // S_A
  std::vector<int> active_flags(world_size, 1);
  std::vector<int> removed_flags(world_size, 0);
  std::vector<int> fail_agg(world_size, 0);
  std::vector<int> full_recovery_failed(world_size, 0);

  bool pending_fail = false;
  CommConfig comm_state{1,
                        world_size,
                        (id - 1 + world_size) % world_size,
                        (id + 1) % world_size,
                        0,
                        0,
                        0,
                        stack_depth,
                        0,
                        0,
                        0};

  double start_time = s4u::Engine::get_clock();
  double compute_time = 0.0;
  double full_recovery_time = 0.0;
  double partial_recovery_time = 0.0;
  int compute_steps = 0;

  for (int step = 0; step < STEPS; step++) {
    bool restart_step = false;
    bool full_recovery_applied = false;
    int leader_id = comm_state.leader_id;
    double current_time = 0.0;
    bool assignment_logged = false;
    int recompute_start = 0;
    int recompute_end = 0;
    int attempt = 0;
    int prev_attempt_stack_depth = -1;

    // Retry loop: a step may need to be re-attempted after a failed all-reduce + recovery.
    while (true) {
      leader_id = comm_state.leader_id;
      int attempt_stack_depth = stack_depth;

      // ------------------------------
      // Compute phase: do S_A stacks, no all-reduce in between
      // ------------------------------
      int prior_depth = prev_attempt_stack_depth;
      int loop_end = stack_depth;
      if (recompute_end > 0) loop_end = std::min(loop_end, recompute_end);
      for (int s = recompute_start; s < loop_end; ++s) {
        int current_data = worker_data[id][s];

        std::normal_distribution<double> jitter_dist(1.0, COMPUTE_JITTER);
        double jitter = std::max(0.1, jitter_dist(rng));
        s4u::this_actor::execute(CONT_COMPUTE_FLOPS * CPU_SCALE * jitter);

        if (!assignment_logged && id == leader_id) {
          util::log_event(id, step, STEPS, "Control: Data-assignment");
          assignment_logged = true;
        }
        if (id == leader_id) {
          if (attempt > 0 && prior_depth >= 0 && s < prior_depth) {
            log_event_with_recompute_stack(id, step, s + 1, stack_depth);
          } else {
            log_event_with_stack_depth(id, step, s + 1, stack_depth);
          }
        }

        // If already failed earlier in this attempt, skip subsequent IO/compute and just reach the all-reduce.
        if (!comm_state.active || pending_fail) continue;

        // Data I/O
        disk->read(DATA_SIZE);
        if (id == leader_id) log_event_with_data_id(id, step, current_data, "I/O: Data-read");

        // Compute
        std::normal_distribution<double> compute_jitter_dist(1.0, COMPUTE_JITTER);
        double compute_jitter = std::max(0.1, compute_jitter_dist(rng));
        double compute_flops = MODEL_COMPUTE_FLOPS * compute_jitter;

        current_time = s4u::Engine::get_clock();
        bool compute_failed = util::should_fail(rng, dist, current_time, next_failure_time, failure_cfg);
        if (compute_failed) {
          pending_fail = true;
          log_event_with_data_id(id, step, current_data, "Computation: Fwd+Bwd-fail");
          // do not execute compute; proceed to end of compute phase
        } else {
          double exec_start = s4u::Engine::get_clock();
          s4u::this_actor::execute(compute_flops);
          compute_time += s4u::Engine::get_clock() - exec_start;
          compute_steps++;
          if (id == leader_id) log_event_with_data_id(id, step, current_data, "Computation: Fwd+Bwd");
        }
      }

      // No extra recompute summary line; depth logging above reflects the recompute state.

      // Sync before all-reduce
      allreduce_barrier->wait();

      // ------------------------------
      // One all-reduce per step attempt
      // ------------------------------
      current_time = s4u::Engine::get_clock();
      bool allreduce_failed = false;
      if (comm_state.active && !pending_fail) {
        allreduce_failed = util::should_fail(rng, dist, current_time, next_failure_time, failure_cfg);
        if (allreduce_failed) {
          pending_fail = true;
          util::log_event(id, step, STEPS, "Communication: Allreduce-fail");
        }
      }

      int local_fail_flag = (comm_state.active && pending_fail) ? 1 : 0;
      std::fill(fail_agg.begin(), fail_agg.end(), 0);
      if (comm_state.active) fail_agg[id] = local_fail_flag;
      SHARED_FAIL_FLAGS[id] = local_fail_flag;

      allreduce_barrier->wait();

      bool system_failed = false;
      for (int w = 0; w < world_size; w++) {
        fail_agg[w] = SHARED_FAIL_FLAGS[w];
        if (fail_agg[w] != 0) system_failed = true;
      }

      bool fast_allreduce = comm_state.active && system_failed;
      if (id == leader_id) util::log_event(id, step, STEPS, fast_allreduce ? "[Allreduce-fast]" : "[Allreduce-norm]");
      double allreduce_sleep = ALLREDUCE_TIME * (fast_allreduce ? ALLREDUCE_FAIL_SCALE : 1.0);
      if (comm_state.active && allreduce_sleep > 0.0) {
        s4u::this_actor::sleep_for(allreduce_sleep);
      }

      allreduce_barrier->wait();

      if (id == leader_id) util::log_event(id, step, STEPS, "Communication: Gradient-sync");

      // ------------------------------
      // Controller at gradient sync failure (paper semantics)
      // ------------------------------
      bool do_full_recovery = false;
      bool do_partial_recovery = false;
      bool seq_updated = false;
      bool recompute_updated = false;
      bool step_done = false;
      std::vector<int> recomp_start;
      std::vector<int> recomp_end;

      if (id == leader_id) {
        int old_stack_depth = stack_depth;
        if (!system_failed) {
          step_done = true;
        } else {
          // Controller cost
          std::normal_distribution<double> compute_jitter_dist(1.0, COMPUTE_JITTER);
          double compute_jitter = std::max(0.1, compute_jitter_dist(rng));
          double controller_0_flops = RECOVERY_DECISION_FLOPS * compute_jitter * CPU_SCALE;
          s4u::this_actor::execute(controller_0_flops);

          // Survivors = active and not failed in this all-reduce
          std::vector<int> survivors;
          survivors.reserve(world_size);
          for (int i = 0; i < world_size; i++) {
            if (active_flags[i] && fail_agg[i] == 0) survivors.push_back(i);
          }

          // Phase 0: if current S_A is still OK with fixed stacks, keep it
          bool phase0_ok = fixed_valid_global(worker_data, survivors, stack_depth, world_size);

          if (phase0_ok) {
            XBT_INFO("[worker%*d %.6f] step %03d/%d -- %-12s active %d/%d",
                     util::worker_id_width(), id, s4u::Engine::get_clock(), step + 1, STEPS,
                     "Control: Phase#0 Go-ahead", (int)survivors.size(), world_size);
            do_partial_recovery = true;
            step_done = true;
          } else {
            XBT_INFO("[worker%*d %.6f] step %03d/%d -- %-12s",
                     util::worker_id_width(), id, s4u::Engine::get_clock(), step + 1, STEPS,
                     "Control: Phase#0 Reorder");

            double controller_1_flops = RECOVERY_DECISION_FLOPS * compute_jitter * CPU_SCALE;
            s4u::this_actor::execute(controller_1_flops);

            int old_SA = stack_depth;
            int new_SA = find_minimal_SA(data_workers, survivors, stack_depth, replicate_level, world_size);

            if (new_SA < 0) {
              do_full_recovery = true;
              XBT_INFO("[worker%*d %.6f] step %03d/%d -- %-12s Full recovery (wipe-out)",
                       util::worker_id_width(), id, s4u::Engine::get_clock(), step + 1, STEPS,
                       "Control: Phase#1");
            } else {
              XBT_INFO("[worker%*d %.6f] step %03d/%d -- %-12s %d -> %d",
                       util::worker_id_width(), id, s4u::Engine::get_clock(), step + 1, STEPS,
                       "Control: Phase#1 Depth", old_SA, new_SA);

              std::vector<std::pair<int, int>> assign_by_type;
              std::vector<std::vector<int>> before_seq(world_size);
              for (int w : survivors) {
                before_seq[w] = worker_data[w];
              }

              bool assigned = min_cost_assignment_global(data_workers, worker_data, survivors, new_SA, world_size, assign_by_type);
              if (!assigned) {
                do_full_recovery = true;
                XBT_INFO("[worker%*d %.6f] step %03d/%d -- %-12s Full recovery (assignment failed)",
                         util::worker_id_width(), id, s4u::Engine::get_clock(), step + 1, STEPS,
                         "Control: Phase#2");
              } else {
                double controller_2_flops = RECOVERY_DECISION_FLOPS * compute_jitter * CPU_SCALE;
                s4u::this_actor::execute(controller_2_flops);

                apply_assignment_global(survivors, assign_by_type, new_SA, replicate_level, world_size, worker_data);

                stack_depth = new_SA;
                do_partial_recovery = true;
                seq_updated = true;
                recompute_updated = true;
                recomp_start.assign(world_size, -1);
                recomp_end.assign(world_size, -1);

                bool any_change = false;
                for (int w : survivors) {
                  bool changed = (before_seq[w] != worker_data[w]);
                  if (!changed) continue;
                  any_change = true;

                  int earliest_change = stack_depth;
                  for (int t = 0; t < stack_depth; ++t) {
                    if (before_seq[w][t] != worker_data[w][t]) {
                      earliest_change = t;
                      break;
                    }
                  }
                  recomp_start[w] = earliest_change;
                  recomp_end[w] = std::min(stack_depth, earliest_change + 1);

                  std::string before = "[";
                  std::string after = "[";
                  for (int t = 0; t < (int)worker_data[w].size(); t++) {
                    if (t > 0) { before += " "; after += " "; }
                    before += "d" + std::to_string(before_seq[w][t]);
                    after += "d" + std::to_string(worker_data[w][t]);
                  }
                  before += "]";
                  after += "]";
                  XBT_INFO("[worker%*d %.6f] step %03d/%d -- %-12s w%d %s -> %s",
                           util::worker_id_width(), id, s4u::Engine::get_clock(), step + 1, STEPS,
                           "Control: Phase#2 Change", w, before.c_str(), after.c_str());
                }
                if (!any_change) {
                  XBT_INFO("[worker%*d %.6f] step %03d/%d -- %-12s",
                           util::worker_id_width(), id, s4u::Engine::get_clock(), step + 1, STEPS,
                           "Control: Phase#2 No-op");
                }

                if (recomp_start.empty()) {
                  recomp_start.assign(world_size, 0);
                  recomp_end.assign(world_size, 0);
                } else if (recomp_start.size() == (size_t)world_size) {
                  for (int w : survivors) {
                    if (recomp_start[w] < 0) {
                      recomp_start[w] = (stack_depth > old_SA) ? old_SA : stack_depth;
                      recomp_end[w] = recomp_start[w] + 1;
                    }
                  }
                  for (int w = 0; w < world_size; ++w) {
                    if (recomp_start[w] < 0) recomp_start[w] = 0;
                    if (recomp_end[w] < 0) recomp_end[w] = 0;
                  }
                }
              }
            }
          }

          if (do_partial_recovery) {
            std::string total_affected = "total affected:";
            std::string round_affected = "this round appended:";
            int round_appended = 0;
            for (int i = 0; i < world_size; i++) {
              if (active_flags[i] && fail_agg[i] != 0) {
                active_flags[i] = 0;
                removed_flags[i] = 1;
                round_affected += " w" + std::to_string(i);
                round_appended++;
              }
            }
            if (round_appended == 0) round_affected += " none";
            for (int i = 0; i < world_size; i++) {
              if (removed_flags[i]) total_affected += " w" + std::to_string(i);
            }
            if (total_affected == "total affected:") total_affected += " none";
            XBT_INFO("[worker%*d %.6f] step %03d/%d -- %-12s %s %s",
                     util::worker_id_width(), id, s4u::Engine::get_clock(), step + 1, STEPS,
                     "Control: Starting PARTIAL system recovery due to",
                     total_affected.c_str(),
                     round_affected.c_str());
          } else if (do_full_recovery) {
            std::fill(full_recovery_failed.begin(), full_recovery_failed.end(), 0);
            for (int i = 0; i < world_size; i++) {
              if (fail_agg[i] != 0 || removed_flags[i] != 0) {
                full_recovery_failed[i] = 1;
              }
            }
            std::fill(active_flags.begin(), active_flags.end(), 1);
          }

          if (!step_done) step_done = false; // on failure, this step attempt did not commit
        }

        if (do_partial_recovery && !seq_updated && stack_depth > old_stack_depth) {
          recomp_start.assign(world_size, 0);
          recomp_end.assign(world_size, 0);
          for (int w = 0; w < world_size; ++w) {
            if (active_flags[w]) {
              recomp_start[w] = old_stack_depth;
              recomp_end[w] = std::min(stack_depth, old_stack_depth + 1);
            }
          }
          recompute_updated = true;
        }

        // Leader-only: log recompute plan when any recompute range is defined.
        if (id == leader_id && (seq_updated || recompute_updated) && stack_depth > 0) {
          std::vector<unsigned char> stack_recompute(stack_depth, 0);
          int n = std::min((int)recomp_start.size(), (int)recomp_end.size());
          for (int w = 0; w < n; ++w) {
            int rs = recomp_start[w];
            int re = recomp_end[w];
            if (re <= 0) continue;
            rs = std::max(0, std::min(rs, stack_depth));
            re = std::max(0, std::min(re, stack_depth));
            for (int s = rs; s < re; ++s) stack_recompute[s] = 1;
          }
          for (int s = 0; s < stack_depth; ++s) {
            if (!stack_recompute[s]) continue;
            if (s >= old_stack_depth) continue; // first-time stacks are not recompute
            std::string msg = "[Stack " + std::to_string(s + 1) + "/" +
                              std::to_string(stack_depth) + " Recompute]";
            util::log_event(id, step, STEPS, msg.c_str());
          }
        }

        // Build communicator ring among active workers
        std::vector<int> active_workers;
        active_workers.reserve(world_size);
        for (int i = 0; i < world_size; i++) if (active_flags[i]) active_workers.push_back(i);
        int active_count = (int)active_workers.size();
        int new_leader_id = active_workers.empty() ? leader_id : active_workers.front();
        if (do_full_recovery) new_leader_id = leader_id;

        for (int i = 0; i < world_size; i++) {
          CommConfig cfg{};
          cfg.active = active_flags[i] ? 1 : 0;
          cfg.active_count = active_count;
          cfg.full_recovery = do_full_recovery ? 1 : 0;
          cfg.partial_recovery = do_partial_recovery ? 1 : 0;
          cfg.leader_id = new_leader_id;
          cfg.stack_depth = stack_depth;
          cfg.seq_updated = seq_updated ? 1 : 0;
          cfg.recompute_updated = recompute_updated ? 1 : 0;
          cfg.step_done = step_done ? 1 : 0;

          if (cfg.active && active_count > 0) {
            auto it = std::find(active_workers.begin(), active_workers.end(), i);
            int rank = (int)std::distance(active_workers.begin(), it);
            int prev_index = (rank - 1 + active_count) % active_count;
            int next_index = (rank + 1) % active_count;
            cfg.prev = active_workers[prev_index];
            cfg.next = active_workers[next_index];
          } else {
            cfg.prev = -1;
            cfg.next = -1;
          }

          SHARED_COMM_CONFIG[i] = cfg;
          if (i == leader_id) comm_state = cfg;
        }

        if (seq_updated) {
          for (int w = 0; w < world_size; w++) {
            for (int t = 0; t < replicate_level; t++) {
              SHARED_SEQ_SNAPSHOT[w * replicate_level + t] = worker_data[w][t];
            }
          }
          if (recomp_start.empty()) recomp_start.assign(world_size, 0);
          if (recomp_end.empty()) recomp_end.assign(world_size, 0);
          SHARED_RECOMPUTE_START = recomp_start;
          SHARED_RECOMPUTE_END = recomp_end;
        } else if (recompute_updated) {
          if (recomp_start.empty()) recomp_start.assign(world_size, 0);
          if (recomp_end.empty()) recomp_end.assign(world_size, 0);
          SHARED_RECOMPUTE_START = recomp_start;
          SHARED_RECOMPUTE_END = recomp_end;
        }
        SHARED_ACTIVE_FLAGS = active_flags;
      }

      allreduce_barrier->wait();

      comm_state = SHARED_COMM_CONFIG[id];
      if (comm_state.seq_updated && comm_state.active) {
        for (int w = 0; w < world_size; w++) {
          for (int t = 0; t < replicate_level; t++) {
            worker_data[w][t] = SHARED_SEQ_SNAPSHOT[w * replicate_level + t];
          }
        }
      }
      if ((comm_state.seq_updated || comm_state.recompute_updated) && comm_state.active) {
        recompute_start = 0;
        if (id >= 0 && id < (int)SHARED_RECOMPUTE_START.size()) {
          recompute_start = SHARED_RECOMPUTE_START[id];
        }
        recompute_end = 0;
        if (id >= 0 && id < (int)SHARED_RECOMPUTE_END.size()) {
          recompute_end = SHARED_RECOMPUTE_END[id];
        }
      }
      if (comm_state.leader_id == id && leader_id != id) {
        active_flags = SHARED_ACTIVE_FLAGS;
        for (int i = 0; i < world_size; i++) {
          removed_flags[i] = active_flags[i] ? 0 : 1;
        }
      }

      stack_depth = comm_state.stack_depth;

      // Partial recovery delay (before retry)
      if (comm_state.partial_recovery && comm_state.active && PARTIAL_RECOVER_TIME > 0.0) {
        double rec_start = s4u::Engine::get_clock();
        s4u::this_actor::sleep_for(PARTIAL_RECOVER_TIME);
        partial_recovery_time += s4u::Engine::get_clock() - rec_start;
        if (id == comm_state.leader_id) {
          XBT_INFO("[worker%*d %.6f] step %03d/%d -- %-12s done",
                   util::worker_id_width(), id, s4u::Engine::get_clock(), step + 1, STEPS,
                   "Control: Partial Recovery");
        }
      } else if (id == comm_state.leader_id && !system_failed) {
        XBT_INFO("[worker%*d %.6f] step %03d/%d -- [Go-ahead active %d/%d]",
                 util::worker_id_width(), id, s4u::Engine::get_clock(), step + 1, STEPS,
                 comm_state.active_count, world_size);
      }

      allreduce_barrier->wait();

      // Full recovery (restart)
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
        comm_state.stack_depth = 1;
        comm_state.seq_updated = 0;
        comm_state.recompute_updated = 0;
        comm_state.step_done = 0;

        std::fill(active_flags.begin(), active_flags.end(), 1);
        worker_data = base_worker_data;
        stack_depth = 1;
        std::fill(removed_flags.begin(), removed_flags.end(), 0);

        step = last_success_ckpt_step - 1;
        restart_step = true;
        full_recovery_applied = true;
        break;
      }

      // If this attempt succeeded, we commit the step.
      if (comm_state.step_done) {
        pending_fail = false;
        break; // exit retry loop; proceed to checkpoint logic
      }

      // Otherwise this attempt failed; retry the SAME step after partial recovery.
      pending_fail = false;
      prev_attempt_stack_depth = attempt_stack_depth;
      attempt++;
      // Continue retry loop
    } // end retry loop

    if (restart_step) {
      if (!full_recovery_applied) step--;
      continue;
    }

    // Checkpointing (unchanged)
    if ((CHECKPOINT_INTERVAL > 0 && ((step + 1) % CHECKPOINT_INTERVAL == 0)) || (step + 1) == STEPS) {
      bool checkpoint_failed = false;
      if (id == comm_state.leader_id) {
        current_time = s4u::Engine::get_clock();
        checkpoint_failed = util::should_fail(rng, dist, current_time, next_failure_time, failure_cfg);
        if (checkpoint_failed) {
          util::log_event(id, step, STEPS, "I/O: Checkpoint-fail");
        }
      }

      if (checkpoint_failed) {
        pending_fail = true;
      } else if (id == comm_state.leader_id) {
        disk->write(MODEL_SIZE);
        util::log_event(id, step, STEPS, "I/O: Checkpoint-saving");
        last_success_ckpt_step = step;
      }

      checkpoint_barrier->wait();

      if (step + 1 == STEPS && id == comm_state.leader_id) {
        util::log_event(id, step, STEPS, "Communication: Final-sync");
      }
    }
  }

  double runtime = s4u::Engine::get_clock() - start_time;
  StatsPayload local_stats{runtime, compute_time, full_recovery_time, partial_recovery_time, compute_steps};
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
      RECOVERY_DECISION_FLOPS = std::max(0.0, std::atof(arg.c_str() + 11));
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
      xbt_assert(REPLICATE_LEVEL < 29, "replicate_level must be < 29");
    } else if (arg.rfind("--platform=", 0) == 0) {
      PLATFORM_PATH = arg.substr(11);
    } else {
      XBT_WARN("Ignoring unknown argument: %s", arg.c_str());
    }
  }

  xbt_assert(REPLICATE_LEVEL < 29, "replicate_level must be < 29");
  xbt_assert(WORLD_SIZE > 2 * RULER_TABLE[REPLICATE_LEVEL - 1].back() - 1,
             "WORLD_SIZE must be > 2 * RULER_TABLE[REPLICATE_LEVEL-1][-1] - 1");

  if (RNG_SEED == 0) RNG_SEED = util::seed_now();

  s4u::Engine e(&argc, argv);
  e.load_platform(PLATFORM_PATH);

  auto hosts = e.get_all_hosts();
  std::sort(hosts.begin(), hosts.end(), [](const s4u::Host* a, const s4u::Host* b) {
    std::string key_a = util::host_locality_key(a->get_name());
    std::string key_b = util::host_locality_key(b->get_name());
    if (key_a == key_b) return a->get_name() < b->get_name();
    return key_a < key_b;
  });

  int world_size = std::min(WORLD_SIZE, (int)hosts.size());
  if (world_size < WORLD_SIZE) {
    XBT_WARN("Requested workers=%d but only %d hosts found", WORLD_SIZE, world_size);
  }
  util::set_worker_id_width(world_size);

  int replicate_level = std::max(1, (int)REPLICATE_LEVEL);

  SHARED_FAIL_FLAGS.assign(world_size, 0);
  SHARED_COMM_CONFIG.assign(world_size, CommConfig{});
  SHARED_ACTIVE_FLAGS.assign(world_size, 1);
  SHARED_SEQ_SNAPSHOT.assign(world_size * replicate_level, 0);
  SHARED_RECOMPUTE_START.assign(world_size, 0);
  SHARED_RECOMPUTE_END.assign(world_size, 0);
  SHARED_CKPT_STEPS.assign(world_size, 0);
  SHARED_STATS.resize(world_size);

  auto allreduce_barrier = s4u::Barrier::create(world_size);
  auto recovery_barrier = s4u::Barrier::create(world_size);
  auto checkpoint_barrier = s4u::Barrier::create(world_size);

  for (int i = 0; i < world_size; i++) {
    s4u::Actor::create("worker" + std::to_string(i),
                       hosts[i],
                       worker,
                       i,
                       world_size,
                       allreduce_barrier,
                       recovery_barrier,
                       checkpoint_barrier);
  }

  e.run();
  return 0;
}
