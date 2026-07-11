/*
Copyright 2025 Multiscale Modeling of Fluid Materials, TU Munich

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "buffer.h"

#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/tfrt_cpu_pjrt_client.h"

namespace jcn {

    /*
     * Creates Buffer from a literal with zero copy but preserves literal.
     *
     * Adapted from https://github.com/openxla/xla/blob/ee9ee727b533dbd14698c9eda979a8c83ed86e11/xla/pjrt/pjrt_stream_executor_client.cc#L1699
     */
    std::unique_ptr<xla::PjRtBuffer> create_buffer(xla::PjRtClient* client, int device_id, xla::Literal* literal) {
        absl::StatusOr<xla::PjRtMemorySpace *> memory_space = client->addressable_devices()[device_id]->default_memory_space();
        
        if (!memory_space.ok()) {
            throw std::runtime_error("Failed to get memory space: " + memory_space.status().ToString());
        }

        absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> input_buffer = client->BufferFromHostBuffer(
            literal->untyped_data(),
            literal->shape().element_type(),
            literal->shape().dimensions(),
            std::optional<absl::Span<int64_t const>>{},
            xla::PjRtClient::HostBufferSemantics::kImmutableZeroCopy,
            []() { /* frees literal */ },
            memory_space.value(),
            nullptr
        );

        if (!input_buffer.ok()) {
            throw std::runtime_error("Failed to create buffer: " + input_buffer.status().ToString());
        }

        return std::move(input_buffer).value();
    }

} // namespace jcn
