#pragma once

struct PAGRunConfig {
  const char *base_file;
  const char *query_file;
  const char *truth_file;
  const char *index_dir;

  int base_count;
  int query_count;
  int dim;
  int result_k;

  int ef_construction;
  int target_degree;
  int projection_levels;

  const char *metric_name = "l2";
  const char *build_order = "dataset_order";
  int max_search_k = 0;
};

void RunPAG(const PAGRunConfig &config);
