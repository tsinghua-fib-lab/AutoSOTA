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

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <regex>
#include <future>
#include <cstdlib>

#include "connector/runner.h"
#include "connector/compiler.h"
#include "connector/libconnector.h"
#include "connector/domain.h"
#include "connector/buffer.h"
#include "connector/model.pb.h"
#include "connector/utils.h"
#include "connector/openequivariance.h"
#include "connector/communication.h"

#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/log/initialize.h"
#include "xla/service/dump.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/env.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"


namespace jcn {

    void Runner::initialize() {

        absl::InitializeLog();

        Logger logger = Logger::getlogger();


        const char* raw_env = std::getenv("JCN_PJRT_PATH");
        if (raw_env == nullptr) {
            throw std::runtime_error(
                "Set JCN_PJRT_PATH to discover PJRT plugins");
        }

        std::string raw_path = std::string(raw_env) + "/pjrt";
        const PJRT_Api* cuda_pjrt_api = nullptr;

        try {
            struct stat st;
            if (stat(raw_path.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
                throw std::runtime_error(
                    "Invalid PJRT plugin directory: " + raw_path);
            }

            DIR* dir = opendir(raw_path.c_str());
            if (!dir) {
                throw std::runtime_error(
                    "Failed to open PJRT plugin directory: " + raw_path);
            }

            struct dirent* entry;
            while ((entry = readdir(dir)) != nullptr) {
                if (entry->d_name[0] == '.') continue;

                std::string backend(entry->d_name);
                std::string backend_dir = raw_path + "/" + backend;

                struct stat backend_st;
                if (stat(backend_dir.c_str(), &backend_st) != 0 || !S_ISDIR(backend_st.st_mode)) {
                    continue;
                }

                std::string plugin_path = backend_dir + "/pjrt_plugin.so";
                if (access(plugin_path.c_str(), R_OK) != 0) {
                    continue;
                }

                absl::StatusOr<const PJRT_Api*> status_or_api =
                    pjrt::LoadPjrtPlugin(backend, plugin_path);

                if (status_or_api.ok()) {
                    logger.log(LogLevel::INFO, "Loaded PJRT plugin " + backend);
                    if (backend == "cuda") {
                        cuda_pjrt_api = status_or_api.value();
                    }
                } else {
                    std::cerr << "Failed to load PJRT plugin " << backend
                            << ": " << status_or_api.status().ToString() << std::endl;
                }
            }

            closedir(dir);
        } catch (const std::exception& e) {
            throw std::runtime_error(
                std::string("Failed to load PJRT plugins: ") + e.what());
        }

        if (cuda_pjrt_api != nullptr) {
            int oeq_rc = chemtrain_register_openequivariance_xla_ffi(
                cuda_pjrt_api, "CUDA");
            if (oeq_rc != 0) {
                throw std::runtime_error(
                    "Failed to register OpenEquivariance XLA FFI handlers for CUDA");
            }
            if (RegisterCommunicationFfi(cuda_pjrt_api, "CUDA") != 0) {
                throw std::runtime_error(
                    "Failed to register chemtrain communication XLA FFI handlers for CUDA");
            }
        }

    }

    Runner::Runner(ConnectorConfig connector_config, bool initialize)
        : config(std::move(connector_config)) {

        if (initialize) {
            Runner::initialize();
        }

        // Singleton
        Logger logger = Logger::getlogger();

        absl::StatusOr<std::unique_ptr<xla::PjRtClient>> client_or_status;

        if (this->config.backend == "cpu") {

          xla::CpuClientOptions create_options;
          create_options.asynchronous = true;

          client_or_status = xla::GetPjRtCpuClient(create_options);

        } else {

            logger.log(LogLevel::INFO, "Initializing PjRtClient for backend '" + this->config.backend + "' with options:");
            logger.log(LogLevel::INFO, "  - Device: " + std::to_string(this->config.device));
            logger.log(LogLevel::INFO, "  - Memory fraction: " + std::to_string(this->config.memory_fraction));


            absl::flat_hash_map<std::string, xla::PjRtValueType> create_options = {
                {"memory_fraction", static_cast<float>(this->config.memory_fraction)},
                {"visible_devices", std::vector<int64_t>({this->config.device})},
            };

            // Initialize the possible backends in the libconnector file
            absl::StatusOr<bool> status_or_success = pjrt::IsPjrtPluginInitialized(this->config.backend);
            if (!status_or_success.ok()) {
                throw std::runtime_error("Failed to initialize PjRtClient: " + status_or_success.status().ToString());
            }

            if (!status_or_success.value()) {
                absl::Status status = pjrt::InitializePjrtPlugin(this->config.backend);
                if (!status.ok()) {
                    throw std::runtime_error("Failed to initialize PjRtClient: " + status.ToString());
                }
            }

            // Get the client
            client_or_status = xla::GetCApiClient(this->config.backend, create_options);

        }

        if (!client_or_status.ok()) {
            throw std::runtime_error("Failed to initialize PjRtClient: " + client_or_status.status().ToString());
        }

        client = std::move(client_or_status).value();

        // Determine the index into addressable_devices() to use for buffer allocation.
        // When visible_devices filtering is applied, addressable_devices() may be remapped.
        pjrt_device_index_ = 0;
        absl::Span<xla::PjRtDevice* const> addressable = client->addressable_devices();
        if (addressable.empty()) {
            throw std::runtime_error("PjRtClient has no addressable devices");
        }

        for (int i = 0; i < addressable.size(); ++i) {
            if (addressable[i]->id() == this->config.device) {
                pjrt_device_index_ = i;
                break;
            }
        }

        logger.log(
            LogLevel::INFO,
            "Using addressable device index " + std::to_string(pjrt_device_index_) +
                " (requested id=" + std::to_string(this->config.device) +
                ", actual id=" + std::to_string(addressable[pjrt_device_index_]->id()) + ")"
        );

        // Print devices
        absl::Span<xla::PjRtDevice* const> devices = client->devices();
        std::string device_list = "";
        for (int i = 0; i < devices.size(); i++) {
            device_list += std::string(devices[i]->ToString()) + ",";
	    }
	    logger.log(LogLevel::INFO, "Found devices [" + device_list + "]");

    }


    ModelProperties Runner::load_model(ModelConfig config) {
        // Singleton
        Logger logger = Logger::getlogger();

        newton = config.newton;
        communication_callbacks = config.communication;
        communication_forward_sites_ = 0;
        communication_widths_.clear();

        // The communication variant uses explicit reverse halo exchange and
        // currently implements only LAMMPS's Newton-on force convention.
        // Never silently substitute the default executable for an explicit
        // `comm on` request.
        if (config.use_communication && !config.newton) {
            throw std::runtime_error(
                "Communication requires Newton pair forces. Use 'newton on' "
                "or select 'comm off' for the standard Newton-off model.");
        }

        model = std::make_unique<jcn::Model>();

        // Deserialize the protobuffer
        if (config.model.empty()) {
            throw std::runtime_error("Cannot load model: Model file is empty.");
        }

        if (!model->ParseFromString(config.model)) {
            throw std::runtime_error("Cannot load model: Model file is invalid or corrupted.");
        }

        // Resolve the executable before constructing the compiler and graph.
        // Legacy top-level fields remain the compatibility view for comm off.
        if (model->variants_size() == 0) {
            if (config.use_communication) {
                throw std::runtime_error(
                    "Communication was requested, but this is a legacy model "
                    "without a named 'comm' variant. Re-export the model.");
            }
            if (model->uses_communication()) {
                throw std::runtime_error(
                    "Legacy communication model cannot be used with 'comm "
                    "off'. Re-export it with named variants.");
            }
        } else {
            const char* requested_name =
                config.use_communication ? "comm" : "default";
            const jcn::Model::ModelVariant* selected = nullptr;
            for (const auto& variant : model->variants()) {
                if (variant.name() == requested_name) {
                    if (selected != nullptr) {
                        throw std::runtime_error(
                            "Model contains duplicate '" +
                            std::string(requested_name) + "' variants.");
                    }
                    selected = &variant;
                }
            }
            if (selected == nullptr) {
                throw std::runtime_error(
                    "Model does not contain the requested '" +
                    std::string(requested_name) + "' variant.");
            }
            if (selected->uses_communication() != config.use_communication) {
                throw std::runtime_error(
                    "Model variant '" + std::string(requested_name) +
                    "' has inconsistent communication metadata.");
            }
            if (selected->uses_communication()) {
                if (selected->communication_forward_sites() <= 0 ||
                    selected->communication_widths_size() !=
                        selected->communication_forward_sites()) {
                    throw std::runtime_error(
                        "Communication variant has inconsistent site metadata.");
                }
                for (int width : selected->communication_widths()) {
                    if (width <= 0 ||
                        width > selected->communication_buffer_width()) {
                        throw std::runtime_error(
                            "Communication site width exceeds the exported "
                            "buffer capacity.");
                    }
                }
            }

            // The runtime below consumes the historical top-level view. Copy
            // the selected variant into it so all compiler and graph setup is
            // driven by one coherent set of metadata.
            model->set_mlir_module(selected->mlir_module());
            model->mutable_neighbor_list()->CopyFrom(selected->neighbor_list());
            model->set_uses_communication(selected->uses_communication());
            model->set_communication_buffer_width(
                selected->communication_buffer_width());
            communication_forward_sites_ =
                selected->communication_forward_sites();
            communication_widths_.assign(selected->communication_widths().begin(),
                                         selected->communication_widths().end());
        }

        if (model->uses_communication() &&
            model->communication_buffer_width() <= 0) {
            throw std::runtime_error(
                "Communication model has no positive buffer width.");
        }
        if (model->mlir_module().empty()) {
            throw std::runtime_error("Selected model variant has no MLIR module.");
        }
        if (model->neighbor_list().nbr_order_size() < 2) {
            throw std::runtime_error(
                "Selected model variant must provide Newton on/off neighbor orders.");
        }

        logger.log(
            LogLevel::DEBUG,
            "Model communication: enabled=" +
                std::to_string(model->uses_communication()) +
                ", width=" +
                std::to_string(model->communication_buffer_width()) +
                ", newton=" + std::to_string(newton) +
                ", neighbor_orders=[" +
                (model->neighbor_list().nbr_order_size() > 0
                     ? std::to_string(model->neighbor_list().nbr_order(0))
                     : "missing") +
                ", " +
                (model->neighbor_list().nbr_order_size() > 1
                     ? std::to_string(model->neighbor_list().nbr_order(1))
                     : "missing") +
                "]");

        // Pass the mlir module to the compiler
        compiler = std::make_unique<Compiler>(model->mlir_module());

        // Extract exported quantity keys from the model proto and pass to atom_builder
        std::vector<std::string> quantities;
        for (int i = 0; i < model->quantities_size(); ++i) {
            quantities.push_back(model->quantities(i));
        }
        atom_builder = std::make_unique<AtomBuilder>(config.atom_multiplier, config.newton, quantities);

        // Read out statistics required for the neighbor lists
        std::vector<std::string> statistics_keys;
        for (int i = 0; i < model->neighbor_list().statistics_keys_size(); i++) {
            statistics_keys.push_back(model->neighbor_list().statistics_keys(i));
        }

        // Select from the available neighbor list types
        switch (model->neighbor_list().type()) {
            case jcn::Model::SIMPLE_SPARSE:
                neighbor_list = std::make_unique<SimpleSparseNeighborList>(
                    statistics_keys
                );
                neighbor_list->initialize(config.neighbor_list_multipliers);

                logger.log(LogLevel::INFO, "Initialize SimpleSparseNeighborList");
                break;
            case jcn::Model::SIMPLE_DENSE:
                neighbor_list = std::make_unique<SimpleDenseNeighborList>(
                    statistics_keys
                );
                neighbor_list->initialize(config.neighbor_list_multipliers);

                logger.log(LogLevel::INFO, "Initialize SimpleDenseNeighborList");
                break;
            case jcn::Model::DEVICE_SPARSE:
                throw std::runtime_error(
                    "DeviceSparseNeighborList is not supported by the current "
                    "LAMMPS connector. Re-export the model with a host "
                    "neighbor-list variant.");
            default:
                throw std::runtime_error(
                    "Unknown neighbor list type: "
                    + std::to_string(model->neighbor_list().type())
                );
        }

        return get_model_properties();

    }

    Results Runner::compute_forces(
        int lnum, int gnum, double **x, double **f, int *type, int inum,
        int *ilist, int *numneigh, int **firstneigh, bool list_changed,
        bool allow_recompile
    ) {

        // Singleton
        Logger logger = Logger::getlogger();

        int max_trials = 10;
        int compilations = 0;
        int initial_compilations = 0;
        int atom_recompilations = 0;
        int edge_recompilations = 0;

        for (int i = 0; i < max_trials; i++) {

            auto trial_start = std::chrono::high_resolution_clock::now();

            // First we build the domain and the neighbor list, then we can
            // determine the input shapes to the program

            AtomShapes atoms = atom_builder->get_shapes(lnum, gnum, allow_recompile);

            NeighborListShapes neighbors = neighbor_list->get_neighbor_list_shapes(
                atoms.n_atoms, inum, ilist, numneigh, allow_recompile);

            atom_recompilation_required_ |= atoms.reallocate;
            edge_recompilation_required_ |= neighbors.reallocate;

            // Now we have all shapes setup to build the module if required.
            // If the module tried to recompile but failed due to disabled
            // recompilation, it will try again in the next call due to the
            // flag recompilation_required.
            // If recompilation is not necessary but allowed, it will depend
            // on how much the buffers are filled.
            recompilation_required |= !executable || atoms.reallocate || neighbors.reallocate;
            if (recompilation_required && allow_recompile) {
                // Shape discovery can need more than one compile before the
                // first successful evaluation (for example, after observing
                // the valid-edge mask). Treat that whole bootstrap as initial
                // compilation rather than reporting a runtime regression.
                const bool compiling_initial_shapes =
                    !has_successful_execution_;
                ++compilations;
                if (compiling_initial_shapes) {
                    ++initial_compilations;
                } else {
                    atom_recompilations +=
                        static_cast<int>(atom_recompilation_required_);
                    edge_recompilations +=
                        static_cast<int>(edge_recompilation_required_);
                }

                logger.log(LogLevel::INFO, "Recompilation necessary");

                compiler->compile(
                    atoms.n_atoms, neighbors.graph_shapes, neighbors.graph_types);

                absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> executable_or_status = client->CompileAndLoad(
                    compiler->module(), compile_options);

                if (!executable_or_status.ok()) {
                    throw std::runtime_error("Failed to compile: " + executable_or_status.status().ToString());
                }

                executable = std::move(executable_or_status).value();

                // Print a cost analysis of the exectuable
                absl::StatusOr<absl::flat_hash_map<std::string, xla::PjRtValueType>> cost_analysis;
                cost_analysis = executable->GetCostAnalysis();
                if (cost_analysis.ok()) {
                    const absl::flat_hash_map<std::string, xla::PjRtValueType>& cost_map = cost_analysis.value();

                    auto it = cost_map.find("flops");
                    if (it != cost_map.end()) {
                        if (const float* flops = std::get_if<float>(&it->second)) {
                            flops_ = *flops;
                            logger.log(LogLevel::INFO,
                                "Cost analysis: " + std::to_string(*flops) + " flops."
                            );
                        } else {
                            std::cerr << "Error: 'flops' is not a float type" << std::endl;
                        }
                    } else {
                        std::cerr << "Error: 'flops' key not found in cost_map" << std::endl;
                    }
                } else {
                    std::cerr << "Failed to get cost analysis: " << cost_analysis.status().ToString() << std::endl;
                }

                recompilation_required = false; // Reset the recompilation flag
                atom_recompilation_required_ = false;
                edge_recompilation_required_ = false;

            } else if (recompilation_required) {
                throw jcn::RecompilationRequired(
                    "Recompilation required but not allowed. Please set allow_recompile to true.");
            }

            auto start = std::chrono::high_resolution_clock::now();

            // Only transfer new data to the GPU if necessary
            bool update = (compilations > 0 || list_changed);

            // Now we have to create the buffers, i.e., copy the data onto
            // the device
            std::vector<xla::PjRtBuffer*> buffer_ptrs = atom_builder->build_domain(client.get(), pjrt_device_index_, lnum, gnum, x, type);

            // TODO: We have to add the gnum option to the neighbor list.
            //       This is only a workaround for the sparse neighbor list
            //       which includes the ghost atoms as senders.
            std::vector<xla::PjRtBuffer*> graph_buffers = neighbor_list->build_graph(
                client.get(), pjrt_device_index_, inum, ilist, numneigh, firstneigh, update);
            buffer_ptrs.insert(buffer_ptrs.end(), graph_buffers.begin(), graph_buffers.end());

            std::vector<std::vector<xla::PjRtBuffer*>> arg_handles = {buffer_ptrs};

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;

            logger.log(LogLevel::DEBUG, "Time taken for buffer creation: " + std::to_string(duration.count()) + " seconds");

            // Check if arg_handles is correctly populated
            if (arg_handles.empty() || arg_handles[0].empty()) {
                throw std::runtime_error("arg_handles is empty or not properly populated");
            }

            const bool uses_communication = model->uses_communication();

            xla::ExecuteContext execute_context;
            xla::ExecuteOptions execute_options;

            // AtomBuilder already receives the owned and ghost counts from
            // LAMMPS. Store those values in the FFI context before execution so
            // PJRT worker threads never need to read LAMMPS atom state.
            CommunicationContext communication_context(
                communication_callbacks, uses_communication,
                &communication_workspace_, lnum,
                static_cast<std::int64_t>(lnum) + gnum,
                communication_forward_sites_, communication_widths_);

            if (uses_communication) {
                absl::Status context_status =
                    AddCommunicationContextToExecuteContext(
                        &execute_context, &communication_context);

                if (!context_status.ok()) {
                    throw std::runtime_error(
                        "Failed to initialize communication execution context: " +
                        context_status.ToString());
                }
                execute_options.context = &execute_context;
            }

            start = std::chrono::high_resolution_clock::now();

            if (uses_communication) {
                communication_context.BeginExecution();
            }

            // PJRT execution runs separately because this caller thread is
            // the only thread allowed to enter the LAMMPS/MPI callbacks.
            std::future<absl::StatusOr<std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>>>> future_results =
                std::async(std::launch::async, [&]() {
                    try {
                        auto results = executable->Execute(
                            absl::Span<const std::vector<xla::PjRtBuffer*>>(arg_handles),
                            execute_options
                        );
                        if (results.ok()) {
                            // Execute only enqueues GPU work. Awaiting output
                            // readiness keeps the rendezvous alive through all
                            // async FFI calls and their dependent device work.
                            for (const auto& replica : results.value()) {
                                for (const auto& buffer : replica) {
                                    absl::Status ready =
                                        buffer->GetReadyFuture().Await();
                                    if (!ready.ok()) {
                                        results = decltype(results)(ready);
                                        break;
                                    }
                                }
                                if (!results.ok()) break;
                            }
                        }
                        if (uses_communication) {
                            communication_context.NotifyExecutionComplete();
                        }
                        return results;
                    } catch (...) {
                        // Ensure the main thread cannot remain asleep if PJRT
                        // reports an exception instead of an error Status.
                        if (uses_communication) {
                            communication_context.NotifyExecutionComplete();
                        }
                        throw;
                    }
                });

            if (uses_communication) {
                communication_context.ServiceUntilExecutionComplete();
            }

            // Wait for the results before validating communication metadata so
            // real PJRT execution failures are not hidden by a secondary
            // "missing communication sites" message.
            absl::StatusOr<std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>>> results = future_results.get();

            if (!results.ok()) {
                throw std::runtime_error("Failed to execute: " + results.status().ToString());
            }

            if (uses_communication) {
                absl::Status communication_status =
                    communication_context.ValidateExecution();
                if (!communication_status.ok()) {
                    throw std::runtime_error(
                        "Communication execution validation failed: " +
                        communication_status.ToString());
                }
            }

            // Now we have to copy the results back to the host
            std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> results_buffers = std::move(results).value();

            // Sort out the results buffers. Map statistics after the exported quantities.
            // The atom_builder carries the exported `quantities` keys so we can
            // determine the offset into the returned results.
            std::map<std::string, std::unique_ptr<xla::PjRtBuffer>> statistics;
            int offset = static_cast<int>(atom_builder->get_quantities().size());
            for (int i = 0; i < neighbor_list->statistics_keys.size(); i++) {
                statistics.emplace(
                    neighbor_list->statistics_keys[i],
                    std::move(results_buffers[0][i + offset])
                );
            }

            bool success = neighbor_list->evaluate_statistics(
                std::move(statistics), allow_recompile);
            if (!success) {
                // Runtime graph statistics can discover that the compiled
                // valid-edge mask is too small. The next trial recompiles it.
                edge_recompilation_required_ = true;
            }

            end = std::chrono::high_resolution_clock::now();
            duration = end - start;

            logger.log(LogLevel::DEBUG, "Time taken for computation: " + std::to_string(duration.count()) + " seconds");

            // Write back the results
            std::vector<double> per_atom_potential;
            double potential = atom_builder->evaluate_domain(
                success, lnum, gnum, f, results_buffers,
                per_atom_potential);

            auto trial_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> trial_duration = trial_end - trial_start;

            logger.log(LogLevel::DEBUG, "Time taken for trial: " + std::to_string(trial_duration.count()) + " seconds");

            results_buffers.clear();

            Results compute_results;
            compute_results.potential = potential;
            compute_results.per_atom_potential = std::move(per_atom_potential);
            compute_results.stats.flops = flops_;
            compute_results.stats.compilations = compilations;
            compute_results.stats.initial_compilations = initial_compilations;
            compute_results.stats.atom_recompilations = atom_recompilations;
            compute_results.stats.edge_recompilations = edge_recompilations;

            // Finished
            if (success) {
                has_successful_execution_ = true;
                return compute_results;
            }

        }

        throw std::runtime_error("Failed to compute forces after " + std::to_string(max_trials) + " trials");

    }

    ModelProperties Runner::get_model_properties() {
        // Singleton
        Logger logger = Logger::getlogger();

        ModelProperties properties;

        if (!model) {
            throw std::runtime_error("Model is not initialized");
        }

        // Sufficient number of ghost atomus must be communicated.
        // The communication cutoff depends on the number of message passing
        // steps which effectively increase the cutoff distance.
        int multiplier;
        if (newton) {
            multiplier = model->neighbor_list().nbr_order()[0];
        } else {
            multiplier = model->neighbor_list().nbr_order()[1];
        }
        properties.comm_dist = multiplier * model->neighbor_list().cutoff();
        properties.communication_buffer_width =
            model->communication_buffer_width();

        if (model->has_unit_style()) {
            properties.unit_style = model->unit_style().c_str();
        } else {
            properties.unit_style = "real"; // Define this as default
        }

        switch (model->neighbor_list().type()) {
            case jcn::Model::SIMPLE_SPARSE:
            case jcn::Model::SIMPLE_DENSE:
                // Neighbor list cutoff must be larger than the model cutoff
                properties.cutoff = model->neighbor_list().cutoff();

                if (multiplier > 1) {
                    // Ghost atoms only required if more than the next neighbor
                    // can affect the local energy of a particle
                    properties.neighbor_list.include_ghosts = true;
                    logger.log(LogLevel::INFO,
                        "Include ghosts: " + std::to_string(properties.neighbor_list.include_ghosts)
                    );
                };
                if (model->neighbor_list().has_half_list()) {
                    properties.neighbor_list.half_list = model->neighbor_list().half_list();
                    logger.log(LogLevel::INFO,
                        "Use half list only " + std::to_string(properties.neighbor_list.half_list)
                    );
                };

                break;
            case jcn::Model::DEVICE_SPARSE:
                throw std::runtime_error(
                    "DeviceSparseNeighborList is not supported by the current "
                    "LAMMPS connector.");
        }

        logger.log(LogLevel::INFO,
            std::string("Model properties:") +
            "\n\t-Cutoff: " + std::to_string(properties.cutoff) +
            "\n\t-Com. distance: " + std::to_string(properties.comm_dist) +
            "\n\t-Unit style: " + properties.unit_style
        );

        return properties;
    }

} // namespace jcn
