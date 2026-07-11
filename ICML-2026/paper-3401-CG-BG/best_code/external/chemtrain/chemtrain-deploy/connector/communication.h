#ifndef CHEMTRAIN_DEPLOY_COMMUNICATION_H_
#define CHEMTRAIN_DEPLOY_COMMUNICATION_H_

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "connector/libconnector.h"

#include "absl/status/status.h"
#include "xla/pjrt/pjrt_executable.h"

struct PJRT_Api;

namespace stream_executor {
class MemoryAllocation;
class StreamExecutor;
}  // namespace stream_executor

namespace jcn {

// Owns one persistent pinned host buffer and a worker that executes ordered
// communication jobs. A Runner has one workspace, so forward and reverse
// calls reuse storage safely across timesteps.
class CommunicationWorkspace {
 public:
  CommunicationWorkspace();
  ~CommunicationWorkspace();

  CommunicationWorkspace(const CommunicationWorkspace&) = delete;
  CommunicationWorkspace& operator=(const CommunicationWorkspace&) = delete;

  using Task = std::function<absl::Status(void*)>;
  using Completion = std::function<void(absl::Status)>;

  void Schedule(stream_executor::StreamExecutor* executor, std::size_t bytes,
                Task task, Completion completion);

 private:
  struct Job {
    stream_executor::StreamExecutor* executor;
    std::size_t bytes;
    Task task;
    Completion completion;
  };

  void WorkerLoop();

  std::mutex mutex_;
  std::condition_variable ready_;
  std::queue<Job> jobs_;
  bool stopping_ = false;
  std::thread worker_;

  stream_executor::StreamExecutor* buffer_executor_ = nullptr;
  std::size_t buffer_capacity_ = 0;
  std::unique_ptr<stream_executor::MemoryAllocation> buffer_;
};

// Per-execution rendezvous between PJRT's asynchronous FFI worker and the
// caller thread that owns LAMMPS/MPI. It also checks that runtime calls still
// match the static communication structure stored by the exporter.
class CommunicationContext {
 public:
  CommunicationContext(CommunicationCallbacks callbacks, bool enabled,
                       CommunicationWorkspace* workspace,
                       std::int64_t owned_rows,
                       std::int64_t active_rows,
                       int expected_forward_sites = 0,
                       std::vector<int> expected_widths = {});

  absl::Status Exchange(void* data, std::int64_t rows, std::int64_t cols,
                        CommunicationScalarType type, bool reverse);
  bool ServiceOne();
  bool HasPending() const;
  void BeginExecution();
  void NotifyExecutionComplete();
  void ServiceUntilExecutionComplete();
  absl::Status ValidateExecution() const;
  bool enabled() const { return enabled_; }
  std::int64_t OwnedRows(std::int64_t capacity) const;
  std::int64_t ActiveRows(std::int64_t capacity) const;
  CommunicationWorkspace* workspace() const { return workspace_; }

 private:
  CommunicationCallbacks callbacks_;
  bool enabled_;
  CommunicationWorkspace* workspace_;
  std::int64_t owned_rows_;
  std::int64_t active_rows_;

  mutable std::mutex mutex_;
  std::condition_variable request_ready_;
  std::condition_variable request_done_;

  bool pending_ = false;
  bool servicing_ = false;
  bool completed_ = false;
  bool execution_complete_ = false;
  int expected_forward_sites_ = 0;
  std::vector<int> expected_widths_;
  int maximum_expected_width_ = 0;
  bool validate_communication_sites_ = false;
  int forward_sites_ = 0;
  int reverse_sites_ = 0;

  void* data_ = nullptr;
  std::int64_t rows_ = 0;
  std::int64_t cols_ = 0;
  CommunicationScalarType type_ = CommunicationScalarType::F32;
  bool reverse_ = false;
  std::string error_;
};

int RegisterCommunicationFfi(const PJRT_Api* api, const char* platform_name);

absl::Status AddCommunicationContextToExecuteContext(
    xla::ExecuteContext* execute_context,
    CommunicationContext* communication_context);

}  // namespace jcn

#endif  // CHEMTRAIN_DEPLOY_COMMUNICATION_H_
