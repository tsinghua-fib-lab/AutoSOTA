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

#ifndef main_h
#define main_h

#include <vector>
#include <memory>
#include <string>
#include <exception>
#include <cstdint>

#define EXPORT __attribute__((visibility("default")))

namespace jcn {

    /**
     * Add an optional host range to the profiler domain already used by XLA.
     * These functions are a small bridge for consumers such as the LAMMPS
     * plugin, which should not depend directly on XLA or NVTX headers.
     */
    EXPORT bool PushCommunicationProfileRange(const char* name);
    EXPORT void PopCommunicationProfileRange();

    enum class CommunicationScalarType {
        F32,
        F64,
    };

    struct CommunicationCallbacks {
        void* context = nullptr;
        int (*exchange)(void* context, void* data, std::int64_t rows,
                        std::int64_t cols, CommunicationScalarType type,
                        bool reverse, const char** error) = nullptr;
    };

    /**
     * Exception to indicate that the model must be recompiled du to a change
     * of shapes.
     */
    class RecompilationRequired : public std::exception {
    public:
        explicit RecompilationRequired(const std::string& message)
            : message_(message) {}
        const char* what() const noexcept override {
            return message_.c_str();
        }
    private:
        std::string message_;

    };

    /**
     * Configurations for the evaluation of the potential
     */
    struct ConnectorConfig{
        /**
         * String identifying the backend on which to evaluate the model.
         * The backend name is inferred from the name of the PjRt plugin file.
         */
        std::string backend;

        /**
         * Integer specifying the local device ID on which to exectute the model.
         */
        int device = 0;

        /**
         * Fraction of device memory to allocate to execute the model.
         */
        float memory_fraction = 0.75f;

    };

    /**
     * Configurations for the evaluation of the potential
     */
    struct ModelConfig{
        /** Protobuffer string containing the exported model */
        std::string model;

        /**
         * Vector with multipliers for the neighbor list. The required
         * multipliers and their effect depend on the type of used neighbor list.
         */
        std::vector<float> neighbor_list_multipliers = {1.5};

        /**
         * Multiplier for reserving additional capacities for ghost atoms
         * (padding).
         */
        float atom_multiplier = 1.1;

        /**
         * Flag to indicate whether the forces should be computed according to
         * the LAMMPS newton setting.
         */
        bool newton;

        /** Select the exported distributed-communication model variant. */
        bool use_communication = false;

        /** Optional host communication implementation for embedded gathers. */
        CommunicationCallbacks communication;

    };

    /**
     * Properties of the model that can be queried by the consumer of the
     * interface, e.g., LAMMPS.
     */
    struct ModelProperties {

        /** The cutoff distance for the potential. */
        double cutoff;

        /** Minimum distance to local atoms for which ghost atoms must be
        communicated. */
        double comm_dist = 0.0;

        /** Maximum number of packed scalars communicated for each atom. */
        int communication_buffer_width = 0;

        /** The unit style of the model. */
        const char* unit_style;

        /** The neighbor list settings of the model. */
        struct {
            bool include_ghosts = false;
            bool half_list = true;
        } neighbor_list;

    };


    /**
     * Statistics of the computation, e.g., the expected FLOPs
     */
    struct Statistics {

        /** The estimated number of floating point operations. */
        double flops;

        /** Number of executable compilations during this evaluation. */
        int compilations;

        /** Compilations performed before the first successful evaluation. */
        int initial_compilations;

        /** Runtime compilations caused by atom-buffer growth. */
        int atom_recompilations;

        /** Runtime compilations caused by neighbor-buffer growth. */
        int edge_recompilations;

    };


    /**
     * Results of the evaluation of the potential.
     */
    struct Results {

        /** The potential energy of the system. */
        double potential;

        /** Potential-energy contribution of each local atom. */
        std::vector<double> per_atom_potential;

        /** The statistics of the computation. */
        Statistics stats;

    };

    /**
     * Pointer to implementation class.
     */
    class EXPORT Connector {
    public:
        Connector(ConnectorConfig config);
        ~Connector();

        /**
         * Loads the model and initializes the interface.
         * @params config: The configuration of the model and interface.
         * @returns The properties of the model.
         */
        ModelProperties load_model(ModelConfig config);

        /**
        * Computes the forces for a system given the atom positions and
        * (if required) the neighbor list.
        * @param lnum: Number of local atoms in the domain.
        * @param gnum: Number of ghost atoms associated with the domain.
        * @param x: Pointer to the atom positions for local and ghost atoms.
        * @param f: Pointer to the force array for local and ghost atoms.
        *     Array is overwritten with computed forces by the interface.
        * @param type: Pointer to the atom types (species) for local and ghost
        *     atoms.
        * @param inum: Number of sender atoms in the domain, corresponding to
        *     the rows of the neighbor list.
        * @param ilist: Array holding the local indices of the senders.
        * @param numneigh: Array holding the number of neighbors for each
        *     sender atom.
        * @param firstneigh: Array holding the local indices of the neighbor
        *     atoms.
        * @param list_changed: Whether the neighbor list has changed since the
        *     the last call to the function.
        * @param allow_recompile: Allows recompilation of the executable.
        *     If set to false while a recompilation is required, will
        *     throw a RecompilationRequired exception.
        * @returns The potential energy of the system and statistics of the
        *     computation.
        */
        Results compute_force(int lnum, int gnum, double **x, double** f,
            int *type, int inum, int *ilist, int *numneigh, int **firstneigh,
            bool list_changed, bool allow_recompile);

        /**
         * Initializes the PjRt backends.
         */
        static void initialize();

    protected:
        static bool initialized;

    private:
        class Impl; // Forward declaration of the implementation class
        std::unique_ptr<Impl> impl_; // Use unique_ptr to manage the implementation
    };
}

#endif
