#include "pag_index.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <random>
#include <vector>

namespace {

std::vector<float> make_vectors(size_t count, size_t dim) {
  std::vector<float> vectors(count * dim);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  for (size_t i = 0; i < count; ++i) {
    for (size_t d = 0; d < dim; ++d) {
      vectors[i * dim + d] = uniform(rng);
    }
  }
  return vectors;
}

void print_first_result(const char *name,
                        const std::vector<pag::SearchResult> &results) {
  if (results.empty()) {
    std::cout << name << ": no result\n";
    return;
  }
  std::cout << name << ": id=" << results.front().id << "\n";
}

void print_first_batch_result(
    const char *name,
    const std::vector<std::vector<pag::SearchResult>> &results) {
  if (results.empty() || results.front().empty()) {
    std::cout << name << ": no result\n";
    return;
  }
  std::cout << name << ": id=" << results.front().front().id << "\n";
}

} // namespace

int main() {
  const size_t count = 1024;
  const size_t dim = 32;
  const std::string static_path = "./example_pag_static";
  const std::string online_path = "./example_pag_online";

  std::filesystem::remove_all(static_path);
  std::filesystem::remove_all(online_path);

  std::vector<float> base = make_vectors(count, dim);

  pag::BuildOptions build;
  build.index_path = static_path;
  build.metric = pag::Metric::L2;
  build.max_search_k = 10;
  build.ef_construction = 100;
  build.target_degree = 16;
  build.projection_levels = 8;

  pag::Index index;
  index.build(base.data(), count, dim, build);
  index.save();

  pag::SearchOptions search;
  search.top_k = 10;
  search.ef_search = 100;

  print_first_result("static search", index.search(base.data(), search));
  print_first_batch_result("static batch search",
                           index.search_batch(base.data(), 4, search));

  pag::Index loaded;
  pag::LoadOptions load;
  load.index_path = static_path;
  load.metric = pag::Metric::L2;
  loaded.load(load);
  print_first_result("loaded search", loaded.search(base.data(), search));

  pag::BuildOptions online_build = build;
  online_build.index_path = online_path;
  online_build.mode = pag::IndexMode::Online;
  online_build.max_elements = count + 4;

  pag::Index online;
  online.build(base.data(), count, dim, online_build);

  std::vector<float> inserted(dim, 2.0f);
  online.insert(inserted.data(), 50000);
  std::vector<float> batch(3 * dim);
  for (size_t i = 0; i < 3; ++i) {
    std::fill(batch.begin() + i * dim, batch.begin() + (i + 1) * dim,
              3.0f + static_cast<float>(i));
  }
  std::vector<pag::Label> labels = {50001, 50002};
  auto assigned = online.add_batch(batch.data(), 1);
  (void)assigned;
  online.insert_batch(batch.data() + dim, labels.data(), labels.size());
  print_first_result("online search", online.search(inserted.data(), search));
  print_first_batch_result("online batch search",
                           online.search_batch(batch.data(), 3, search));

  std::filesystem::remove_all(static_path);
  std::filesystem::remove_all(online_path);
  return 0;
}
