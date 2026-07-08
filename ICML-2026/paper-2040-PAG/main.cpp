#include "pag_config.h"

#include <cstdlib>
#include <cstring>
#include <cctype>
#include <iostream>
#include <exception>

namespace {

constexpr int kMinArgc = 12;
constexpr int kMaxArgc = 15;

int ParseIntArg(const char *value) { return std::atoi(value); }

bool IsIntegerArg(const char *value) {
  if (value == nullptr || *value == '\0') {
    return false;
  }
  for (const char *cursor = value; *cursor != '\0'; ++cursor) {
    if (!std::isdigit(static_cast<unsigned char>(*cursor))) {
      return false;
    }
  }
  return true;
}

const char *ParseMetricArg(const char *value) {
  if (std::strcmp(value, "cosine") == 0 || std::strcmp(value, "COSINE") == 0 ||
      std::strcmp(value, "l2") == 0 || std::strcmp(value, "L2") == 0 ||
      std::strcmp(value, "mips") == 0 || std::strcmp(value, "MIPS") == 0 ||
      std::strcmp(value, "ip") == 0 || std::strcmp(value, "IP") == 0) {
    return value;
  }
  std::cerr << "Unknown metric: " << value << "\n";
  std::exit(1);
}

const char *ParseBuildOrderArg(const char *value) {
  if (std::strcmp(value, "dataset_order") == 0 ||
      std::strcmp(value, "default") == 0) {
    return value;
  }
  std::cerr << "Unknown build order: " << value << "\n";
  std::exit(1);
}

void PrintUsage(const char *program) {
  std::cerr << "Usage: " << program
            << " <base.fbin> <query.fbin> <truth.ibin> <index_dir>"
            << " <base_count> <query_count> <dim> <topk>"
            << " <ef_construction> <target_degree> <projection_levels>"
            << " [l2|cosine|mips] [max_search_k]\n";
}

} // namespace

int main(int argc, char **argv) {
  if (argc < kMinArgc || argc > kMaxArgc) {
    PrintUsage(argv[0]);
    return 1;
  }

  PAGRunConfig config{argv[1],
                      argv[2],
                      argv[3],
                      argv[4],
                      ParseIntArg(argv[5]),
                      ParseIntArg(argv[6]),
                      ParseIntArg(argv[7]),
                      ParseIntArg(argv[8]),
                      ParseIntArg(argv[9]),
                      ParseIntArg(argv[10]),
                      ParseIntArg(argv[11])};

  if (argc >= 13) {
    config.metric_name = ParseMetricArg(argv[12]);
  }
  if (argc >= 14) {
    if (IsIntegerArg(argv[13])) {
      config.max_search_k = ParseIntArg(argv[13]);
    } else {
      config.build_order = ParseBuildOrderArg(argv[13]);
    }
  }
  if (argc >= 15) {
    config.max_search_k = ParseIntArg(argv[14]);
  }

  try {
    RunPAG(config);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "PAG error: " << error.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "PAG error: unknown failure\n";
    return 1;
  }
}
