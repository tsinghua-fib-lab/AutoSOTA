#include "connector/communication.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <mutex>

#include "absl/status/status.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/type_registry.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_extension.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace jcn {

// XLA's internal C++ FFI and TSL async types are intentionally confined to
// this translation unit. The rest of the connector sees only the execution
// context adapter declared in communication.h, which limits coupling to the
// pinned XLA revision used to build libconnector.

CommunicationWorkspace::CommunicationWorkspace()
    : worker_(&CommunicationWorkspace::WorkerLoop, this) {}

CommunicationWorkspace::~CommunicationWorkspace() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  ready_.notify_one();
  if (worker_.joinable()) worker_.join();
}

void CommunicationWorkspace::Schedule(
    stream_executor::StreamExecutor* executor, std::size_t bytes, Task task,
    Completion completion) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    jobs_.push(Job{executor, bytes, std::move(task), std::move(completion)});
  }
  ready_.notify_one();
}

void CommunicationWorkspace::WorkerLoop() {
  while (true) {
    Job job;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      ready_.wait(lock, [this] { return stopping_ || !jobs_.empty(); });
      if (stopping_ && jobs_.empty()) return;
      job = std::move(jobs_.front());
      jobs_.pop();
    }

    absl::Status status;
    try {
      if (job.executor == nullptr) {
        status = absl::InternalError(
            "communication job has no StreamExecutor");
      } else {
        const std::size_t allocation_bytes = std::max<std::size_t>(job.bytes, 1);
        if (buffer_ == nullptr || buffer_executor_ != job.executor ||
            buffer_capacity_ < allocation_bytes) {
          // Jobs are serialized by this one worker, so a pinned allocation can
          // be reused safely. Keep a non-null one-byte allocation for empty
          // MPI ranks, which still participate in LAMMPS communication.
          auto allocation = job.executor->HostMemoryAllocate(allocation_bytes);
          if (!allocation.ok()) {
            status = allocation.status();
          } else {
            buffer_ = std::move(allocation).value();
            buffer_executor_ = job.executor;
            buffer_capacity_ = allocation_bytes;
          }
        }
        if (status.ok()) status = job.task(buffer_->address().opaque());
      }
    } catch (const std::exception& error) {
      status = absl::InternalError(error.what());
    } catch (...) {
      status = absl::InternalError("unknown communication worker failure");
    }
    job.completion(std::move(status));
  }
}

CommunicationContext::CommunicationContext(CommunicationCallbacks callbacks,
                                           bool enabled,
                                           CommunicationWorkspace* workspace,
                                           std::int64_t owned_rows,
                                           std::int64_t active_rows,
                                           int expected_forward_sites,
                                           std::vector<int> expected_widths)
    : callbacks_(callbacks),
      enabled_(enabled),
      workspace_(workspace),
      owned_rows_(owned_rows),
      active_rows_(active_rows),
      expected_forward_sites_(expected_forward_sites),
      expected_widths_(std::move(expected_widths)),
      validate_communication_sites_(
          std::getenv("JCN_VALIDATE_COMMUNICATION") != nullptr) {
  if (!expected_widths_.empty()) {
    maximum_expected_width_ = *std::max_element(
        expected_widths_.begin(), expected_widths_.end());
  }
}

std::int64_t CommunicationContext::ActiveRows(std::int64_t capacity) const {
  if (!enabled_) return capacity;
  // A/B switch for measuring whether active-prefix staging beats transferring
  // the complete static-capacity buffer on a specific system.
  if (std::getenv("JCN_COMM_STAGE_FULL_BUFFER") != nullptr) return capacity;
  const std::int64_t rows = active_rows_;
  return rows >= 0 && rows <= capacity ? rows : capacity;
}

std::int64_t CommunicationContext::OwnedRows(std::int64_t capacity) const {
  if (!enabled_) return capacity;
  // Preserve the old symmetric staging path as a runtime A/B fallback.
  if (std::getenv("JCN_COMM_STAGE_FULL_BUFFER") != nullptr) return capacity;
  // LAMMPS stores owned atoms first and ghosts directly afterward. Runner
  // supplies the owned prefix from its existing `lnum` argument, allowing the
  // FFI to reduce host transfers without adding another runtime callback.
  return owned_rows_ >= 0 && owned_rows_ <= capacity ? owned_rows_ : capacity;
}

absl::Status CommunicationContext::Exchange(
    void* data, std::int64_t rows, std::int64_t cols,
    CommunicationScalarType type, bool reverse) {
  if (!enabled_) return absl::OkStatus();

  if (callbacks_.exchange == nullptr) {
    return absl::FailedPreconditionError(
        "communicating model executed without LAMMPS communication callbacks");
  }

  std::unique_lock<std::mutex> lock(mutex_);

  int& sites = reverse ? reverse_sites_ : forward_sites_;

  if (cols <= 0) {
    return absl::InvalidArgumentError(
        "communication site width must be positive");
  }

  if (validate_communication_sites_ && !expected_widths_.empty()) {
    if (!reverse &&
        (sites >= static_cast<int>(expected_widths_.size()) ||
        expected_widths_[sites] != cols)) {
      return absl::InvalidArgumentError(
          "communication site width does not match exported metadata");
    }

    if (reverse &&
        (sites >= static_cast<int>(expected_widths_.size()) ||
        cols > maximum_expected_width_)) {
      return absl::InvalidArgumentError(
          "reverse communication exceeds exported communication bounds");
    }
  }

  ++sites;

  request_done_.wait(lock, [this] { return !pending_; });

  data_ = data;
  rows_ = rows;
  cols_ = cols;
  type_ = type;
  reverse_ = reverse;
  error_.clear();
  completed_ = false;
  pending_ = true;

  request_ready_.notify_one();
  request_done_.wait(lock, [this] { return completed_; });

  std::string error = error_;

  pending_ = false;
  completed_ = false;
  request_done_.notify_all();

  if (!error.empty()) return absl::InternalError(error);
  return absl::OkStatus();
}

bool CommunicationContext::ServiceOne() {
  void* data;
  std::int64_t rows;
  std::int64_t cols;
  CommunicationScalarType type;
  bool reverse;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pending_ || servicing_ || completed_) return false;

    servicing_ = true;
    data = data_;
    rows = rows_;
    cols = cols_;
    type = type_;
    reverse = reverse_;
  }

  const char* callback_error = nullptr;
  int rc = callbacks_.exchange(callbacks_.context, data, rows, cols, type,
                               reverse, &callback_error);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    error_.clear();

    if (rc != 0) {
      error_ = callback_error == nullptr
                   ? "LAMMPS communication callback failed"
                   : callback_error;
    }

    servicing_ = false;
    completed_ = true;
  }

  request_done_.notify_all();
  return true;
}

bool CommunicationContext::HasPending() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return pending_ && !completed_;
}

void CommunicationContext::BeginExecution() {
  std::lock_guard<std::mutex> lock(mutex_);
  execution_complete_ = false;
  forward_sites_ = 0;
  reverse_sites_ = 0;
}

absl::Status CommunicationContext::ValidateExecution() const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (enabled_ && expected_forward_sites_ > 0 &&
      forward_sites_ != expected_forward_sites_) {
    return absl::FailedPreconditionError(
        "executed " + std::to_string(forward_sites_) +
        " forward communication sites; exported metadata requires " +
        std::to_string(expected_forward_sites_));
  }
  return absl::OkStatus();
}

void CommunicationContext::NotifyExecutionComplete() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    execution_complete_ = true;
  }
  request_ready_.notify_all();
}

void CommunicationContext::ServiceUntilExecutionComplete() {
  while (true) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      // PJRT executes on a worker, but LAMMPS and MPI must remain on the
      // caller thread. Wake this thread only for one of those callbacks or
      // when all device work has completed; no periodic polling is needed.
      request_ready_.wait(lock, [this] {
        return execution_complete_ ||
               (pending_ && !servicing_ && !completed_);
      });

      // A communication request takes priority over execution completion.
      // In practice PJRT cannot finish while an async FFI result is pending,
      // but this ordering also makes shutdown robust to future FFI changes.
      if (!(pending_ && !servicing_ && !completed_) &&
          execution_complete_) {
        return;
      }
    }
    ServiceOne();
  }
}

namespace {

namespace ffi = xla::ffi;
namespace se = stream_executor;

bool CommunicationDebugEnabled() {
  static const bool enabled = std::getenv("JCN_COMM_DEBUG") != nullptr;
  return enabled;
}

class ScopedCommunicationProfileRange {
 public:
  explicit ScopedCommunicationProfileRange(const char* name) {
    static const bool enabled = std::getenv("JCN_COMM_PROFILE") != nullptr;
    if (enabled) active_ = PushCommunicationProfileRange(name);
  }

  ~ScopedCommunicationProfileRange() {
    if (active_) PopCommunicationProfileRange();
  }

 private:
  bool active_ = false;
};

xla::ffi::TypeRegistry::TypeId g_communication_context_type_id =
    xla::ffi::TypeRegistry::kUnknownTypeId;
std::recursive_mutex g_communication_registration_mutex;

tsl::AsyncValueRef<tsl::Chain> RunExchange(
    ffi::AnyBuffer input, ffi::AnyBuffer token_input,
    ffi::Result<ffi::AnyBuffer> output,
    ffi::Result<ffi::AnyBuffer> token_output,
    se::Stream* stream, CommunicationContext* context, bool reverse) {
  auto done = tsl::MakeConstructedAsyncValueRef<tsl::Chain>();

  auto fail = [&done](std::string message) {
    done.SetError(absl::InternalError(std::move(message)));
  };

  if (stream == nullptr) {
    fail("chemtrain communication FFI called without stream");
    return done;
  }

  if (context == nullptr) {
    fail("chemtrain communication FFI called without CommunicationContext");
    return done;
  }

  if (context->workspace() == nullptr) {
    fail("chemtrain communication FFI called without workspace");
    return done;
  }

  if (input.dimensions().size() != 2 || output->dimensions().size() != 2 ||
      input.dimensions()[0] != output->dimensions()[0] ||
      input.dimensions()[1] != output->dimensions()[1] ||
      input.element_type() != output->element_type()) {
    fail("chemtrain gather expects matching rank-2 input and output buffers");
    return done;
  }

  if (token_input.element_type() != xla::F32 ||
      token_output->element_type() != xla::F32 ||
      token_input.dimensions().size() != 1 ||
      token_output->dimensions().size() != 1 ||
      token_input.dimensions()[0] != 1 ||
      token_output->dimensions()[0] != 1) {
    fail("chemtrain gather expects a matching float32[1] ordering token");
    return done;
  }

  CommunicationScalarType scalar_type;
  if (input.element_type() == xla::F32) {
    scalar_type = CommunicationScalarType::F32;
  } else if (input.element_type() == xla::F64) {
    scalar_type = CommunicationScalarType::F64;
  } else {
    fail("chemtrain gather supports only f32 and f64 buffers");
    return done;
  }

  const std::int64_t rows = input.dimensions()[0];
  const std::int64_t cols = input.dimensions()[1];
  const std::size_t bytes = input.size_bytes();
  const std::int64_t active_rows = context->ActiveRows(rows);
  const std::int64_t owned_rows = context->OwnedRows(active_rows);
  const std::size_t element_bytes =
      scalar_type == CommunicationScalarType::F32 ? sizeof(float)
                                                   : sizeof(double);
  const std::size_t owned_bytes =
      static_cast<std::size_t>(owned_rows) *
      static_cast<std::size_t>(cols) * element_bytes;
  const std::size_t active_bytes =
      static_cast<std::size_t>(active_rows) *
      static_cast<std::size_t>(cols) * element_bytes;
  const std::size_t ghost_bytes = active_bytes - owned_bytes;
  const bool compact_staging = owned_rows < active_rows;

  void* input_data = input.untyped_data();
  void* output_data = output->untyped_data();
  void* token_input_data = token_input.untyped_data();
  void* token_output_data = token_output->untyped_data();
  CommunicationWorkspace* workspace = context->workspace();
  workspace->Schedule(
      stream->parent(), active_bytes,
      [=](void* host) -> absl::Status {
        se::DeviceAddressBase src(input_data, bytes);
        se::DeviceAddressBase dst(output_data, bytes);
        se::DeviceAddressBase token_src(token_input_data, sizeof(float));
        se::DeviceAddressBase token_dst(token_output_data, sizeof(float));

        absl::Status status;

        // Identity/no-communication path: preserve exact identity semantics.
        // This path is not the performance-critical communicating path.
        if (!context->enabled()) {
          status = stream->Memcpy(&dst, src, bytes);
          if (!status.ok()) return status;

          status = stream->Memcpy(&token_dst, token_src, sizeof(float));
          if (!status.ok()) return status;

          return absl::OkStatus();
        }

        // Compact staging preserves the complete device result with a cheap
        // D2D identity copy, then crosses the host boundary only for the row
        // ranges needed by the direction-specific LAMMPS exchange.
        if (compact_staging) {
          status = stream->Memcpy(&dst, src, bytes);
          if (!status.ok()) return status;
        }

        const std::size_t inbound_bytes =
            compact_staging ? (reverse ? active_bytes : owned_bytes)
                            : active_bytes;
        se::DeviceAddressBase inbound_src(input_data, inbound_bytes);
        status = stream->Memcpy(host, inbound_src, inbound_bytes);
        if (!status.ok()) return status;

        // Synchronize an event recorded immediately after the D2H transfer,
        // rather than draining the whole XLA stream with BlockHostUntilDone.
        // Only this workspace worker waits; the FFI remains asynchronous.
        // Keep the old behavior as an explicit profiling fallback.
        {
          ScopedCommunicationProfileRange range(
              "chemtrain_comm.device_to_host_wait");
          if (std::getenv("JCN_COMM_BLOCK_STREAM") != nullptr) {
            status = stream->BlockHostUntilDone();
          } else {
            auto copy_ready = stream->parent()->CreateEvent();
            if (!copy_ready.ok()) return copy_ready.status();
            std::unique_ptr<se::Event> event = std::move(copy_ready).value();
            status = stream->RecordEvent(event.get());
            if (!status.ok()) return status;
            status = event->Synchronize();
          }
        }
        if (!status.ok()) return status;

        {
          // This range covers the worker-to-caller rendezvous as well as the
          // LAMMPS callback. Main-thread LAMMPS and pack/unpack ranges nested
          // in the same interval identify how much of the wait is useful work.
          ScopedCommunicationProfileRange range(
              "chemtrain_comm.host_exchange_wait");
          status = context->Exchange(host, active_rows, cols, scalar_type,
                                     reverse);
        }
        if (!status.ok()) return status;

        if (compact_staging) {
          if (reverse) {
            // Reverse communication changes owned cotangents and defines
            // ghost cotangents as zero after their contribution was returned.
            se::DeviceAddressBase owned_dst(output_data, owned_bytes);
            status = stream->Memcpy(&owned_dst, host, owned_bytes);
            if (!status.ok()) return status;
            if (ghost_bytes > 0) {
              se::DeviceAddressBase ghost_dst(
                  static_cast<char*>(output_data) + owned_bytes, ghost_bytes);
              status = stream->MemZero(&ghost_dst, ghost_bytes);
              if (!status.ok()) return status;
            }
          } else if (ghost_bytes > 0) {
            // Owned rows already came from the D2D identity copy. LAMMPS has
            // overwritten every active ghost row during forward exchange.
            se::DeviceAddressBase ghost_dst(
                static_cast<char*>(output_data) + owned_bytes, ghost_bytes);
            status = stream->Memcpy(
                &ghost_dst, static_cast<char*>(host) + owned_bytes,
                ghost_bytes);
            if (!status.ok()) return status;
          }
        } else {
          status = stream->Memcpy(&dst, host, active_bytes);
          if (!status.ok()) return status;
          if (reverse && ghost_bytes > 0) {
            // The LAMMPS callback deliberately leaves staged ghost rows
            // untouched. Define the transpose result on the device in both
            // compact and full-staging modes without an extra host traversal.
            se::DeviceAddressBase ghost_dst(
                static_cast<char*>(output_data) + owned_bytes, ghost_bytes);
            status = stream->MemZero(&ghost_dst, ghost_bytes);
            if (!status.ok()) return status;
          }
        }
        // The token copy is deliberately enqueued last. Its output cannot
        // become ready until LAMMPS communication and the feature copy-back
        // finish, so the next FFI site has an ordinary XLA data dependency.
        status = stream->Memcpy(&token_dst, token_src, sizeof(float));
        if (!status.ok()) return status;
        return absl::OkStatus();
      },
      [done](absl::Status status) mutable {
        // Completing this async value tells XLA that both the host callback
        // and the final host-to-device copy are finished. Dependent kernels
        // must not consume the output before this point.
        if (status.ok()) {
          done.SetStateConcrete();
        } else {
          done.SetError(std::move(status));
        }
      });
  return done;
}

tsl::AsyncValueRef<tsl::Chain> GatherForward(
    ffi::AnyBuffer input, ffi::AnyBuffer token_input,
    ffi::Result<ffi::AnyBuffer> output,
    ffi::Result<ffi::AnyBuffer> token_output,
    se::Stream* stream, CommunicationContext* context) {
  return RunExchange(input, token_input, output, token_output, stream, context,
                     false);
}

tsl::AsyncValueRef<tsl::Chain> GatherReverse(
    ffi::AnyBuffer input, ffi::AnyBuffer token_input,
    ffi::Result<ffi::AnyBuffer> output,
    ffi::Result<ffi::AnyBuffer> token_output,
    se::Stream* stream, CommunicationContext* context) {
  return RunExchange(input, token_input, output, token_output, stream, context,
                     true);
}

XLA_FFI_DEFINE_HANDLER(
    kGatherForward, GatherForward,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::UserData<CommunicationContext>>());

XLA_FFI_DEFINE_HANDLER(
    kGatherReverse, GatherReverse,
    ffi::Ffi::Bind()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ctx<ffi::Stream>()
        .Ctx<ffi::UserData<CommunicationContext>>());

const PJRT_FFI* FindFfiExtension(const PJRT_Api* api) {
  if (api == nullptr) return nullptr;

  for (PJRT_Extension_Base* ext = api->extension_start; ext != nullptr;
       ext = ext->next) {
    if (ext->type == PJRT_Extension_Type_FFI) {
      return reinterpret_cast<const PJRT_FFI*>(ext);
    }
  }

  return nullptr;
}

absl::Status RegisterCommunicationContextType(const PJRT_FFI* ffi) {
  std::lock_guard<std::recursive_mutex> lock(g_communication_registration_mutex);
  if (g_communication_context_type_id !=
      xla::ffi::TypeRegistry::kUnknownTypeId) {
    return absl::OkStatus();
  }

  if (ffi == nullptr || ffi->type_register == nullptr) {
    return absl::InternalError(
        "PJRT FFI extension does not provide type_register");
  }

  // Critical:
  // This must match what ffi::UserData<CommunicationContext> asks for.
  // In this XLA revision, TypeRegistry::GetTypeName<T>() is typeid(T).name(),
  // not a demangled string such as "jcn::CommunicationContext".
  absl::string_view type_name =
      xla::ffi::TypeRegistry::GetTypeName<CommunicationContext>();

  // Match the C++ FFI type info used by internal::GetTypeId<T>(api).
  // If this does not match, XLA may assign or expect a different type id.
  xla::ffi::TypeRegistry::TypeInfo cpp_type_info =
      xla::ffi::TypeRegistry::GetTypeInfo<CommunicationContext>();

  PJRT_FFI_Type_Info type_info;
  std::memset(&type_info, 0, sizeof(type_info));
  type_info.deleter = cpp_type_info.deleter;
  type_info.serialize = nullptr;
  type_info.deserialize = nullptr;

  PJRT_FFI_Type_Register_Args args;
  std::memset(&args, 0, sizeof(args));

  args.struct_size = PJRT_FFI_Type_Register_Args_STRUCT_SIZE;
  args.extension_start = nullptr;  // Keep this only if your struct has it.
  args.type_name = type_name.data();
  args.type_name_size = type_name.size();
  args.type_id = 0;
  args.type_info = &type_info;

  PJRT_Error* error = ffi->type_register(&args);
  if (error != nullptr) {
    return absl::InternalError(
        "Failed to register CommunicationContext FFI type");
  }

  g_communication_context_type_id =
      xla::ffi::TypeRegistry::TypeId(args.type_id);

  if (g_communication_context_type_id ==
      xla::ffi::TypeRegistry::kUnknownTypeId) {
    return absl::InternalError(
        "PJRT FFI registered CommunicationContext with unknown type id");
  }

  if (CommunicationDebugEnabled()) {
    std::cerr << "Registered CommunicationContext FFI type name='"
              << std::string(type_name)
              << "' id="
              << g_communication_context_type_id.value()
              << std::endl;
  }

  return absl::OkStatus();
}

int RegisterOne(const PJRT_FFI* ffi, const char* name,
                XLA_FFI_Handler* handler,
                const char* platform_name) {
  PJRT_FFI_Register_Handler_Args args;
  std::memset(&args, 0, sizeof(args));

  args.struct_size = PJRT_FFI_Register_Handler_Args_STRUCT_SIZE;
  args.target_name = name;
  args.target_name_size = std::strlen(name);
  args.handler = reinterpret_cast<void*>(handler);
  args.platform_name = platform_name;
  args.platform_name_size = std::strlen(platform_name);
  args.traits = static_cast<PJRT_FFI_Handler_TraitsBits>(0);

  return ffi->register_handler(&args) == nullptr ? 0 : 1;
}

}  // namespace

absl::Status AddCommunicationContextToExecuteContext(
    xla::ExecuteContext* execute_context,
    CommunicationContext* communication_context) {
  if (execute_context == nullptr) {
    return absl::InvalidArgumentError("execute_context must not be null");
  }

  if (communication_context == nullptr) {
    return absl::InvalidArgumentError(
        "communication_context must not be null");
  }

  if (g_communication_context_type_id ==
      xla::ffi::TypeRegistry::kUnknownTypeId) {
    return absl::FailedPreconditionError(
        "CommunicationContext FFI type was not registered");
  }

  return execute_context->ffi_context().Insert(
      g_communication_context_type_id,
      communication_context);
}

int RegisterCommunicationFfi(const PJRT_Api* api, const char* platform_name) {
  std::lock_guard<std::recursive_mutex> lock(g_communication_registration_mutex);
  if (CommunicationDebugEnabled()) {
    std::cerr << "RegisterCommunicationFfi called for " << platform_name
              << std::endl;
  }

  const PJRT_FFI* ffi = FindFfiExtension(api);
  if (ffi == nullptr || ffi->register_handler == nullptr) {
    std::cerr << "PJRT FFI extension/register_handler unavailable"
              << std::endl;
    return 1;
  }

  absl::Status type_status = RegisterCommunicationContextType(ffi);
  if (!type_status.ok()) {
    std::cerr << "RegisterCommunicationContextType failed: "
              << type_status.ToString() << std::endl;
    return 1;
  }

  int rc = 0;
  rc |= RegisterOne(ffi, "chemtrain_deploy.gather_forward", kGatherForward,
                    platform_name);
  rc |= RegisterOne(ffi, "chemtrain_deploy.gather_reverse", kGatherReverse,
                    platform_name);

  if (CommunicationDebugEnabled()) {
    std::cerr << "RegisterCommunicationFfi rc=" << rc << std::endl;
  }
  return rc;
}

}  // namespace jcn
