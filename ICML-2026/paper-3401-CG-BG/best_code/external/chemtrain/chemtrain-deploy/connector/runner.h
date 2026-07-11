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

#include "connector/compiler.h"
#include "connector/graph_builder.h"
#include "connector/libconnector.h"
#include "connector/domain.h"
#include "connector/communication.h"
#include "connector/model.pb.h"

#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/tfrt_cpu_pjrt_client.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/service/dump.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/env.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"

#include <vector>

#ifndef RUNNER_H
#define RUNNER_H

namespace jcn {

    // Loads plugins and initializes the runtime
    void initialize();

    class Runner {
    public:
        Runner(ConnectorConfig connector_config, bool initialize);
        ~Runner() = default;

        ModelProperties load_model(ModelConfig config);

        // Computes the forces and writes them directly to the force array
        Results compute_forces(
            int lnum, int gnum, double **x, double** f, int *type, int inum,
            int *ilist, int *numneigh, int **firstneigh, bool list_changed,
            bool allow_recompile);

        static void initialize();

    private:
        ModelProperties get_model_properties();

        std::unique_ptr<jcn::Model> model;

        std::unique_ptr<xla::PjRtClient> client;
        std::unique_ptr<xla::PjRtLoadedExecutable> executable;

        std::unique_ptr<GraphBuilder> neighbor_list;

        std::unique_ptr<AtomBuilder> atom_builder;
        std::unique_ptr<Compiler> compiler;

        xla::CompileOptions compile_options;

        ConnectorConfig config;

        // Index into client->addressable_devices() used for buffer allocation.
        int pjrt_device_index_ = 0;

        /*
         * Saves the recompilation request until the exectuable is actually
         * recompiled.
         */
        bool recompilation_required = false;
        bool has_successful_execution_ = false;

        // Shape discovery can request a collective retry before compilation is
        // allowed. Preserve its causes until the retry actually compiles them.
        bool atom_recompilation_required_ = false;
        bool edge_recompilation_required_ = false;

        float flops_;

        bool newton;

        CommunicationCallbacks communication_callbacks;
        CommunicationWorkspace communication_workspace_;
        int communication_forward_sites_ = 0;
        std::vector<int> communication_widths_;

    };

} // namespace jcn

#endif //RUNNER_H
