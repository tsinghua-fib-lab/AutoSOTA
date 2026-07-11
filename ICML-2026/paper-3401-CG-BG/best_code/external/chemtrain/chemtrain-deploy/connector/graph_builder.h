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

#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/literal_util.h"

#include "connector/utils.h"

#ifndef GRAPH_BUILDER_H
#define GRAPH_BUILDER_H

namespace jcn {

    /**
     * Contains shapes and types of the graph input buffers.
     */
    struct NeighborListShapes {

        /**
         * Shapes of the neighborlist inputs.
         */
        std::vector<std::vector<int64_t>> graph_shapes;

        /**
         * Types of the neighborlist inputs.
         */
        std::vector<xla::PrimitiveType> graph_types;

        /**
         * Indicates whether a shape change requires recompilation.
         */
        bool reallocate;
    };

    /**
     * Abstract baseclass for all graph / neighborlist builders.
     */
    class GraphBuilder {
    public:
        GraphBuilder(std::vector<std::string> statistics);
        virtual ~GraphBuilder() = default;

        virtual void initialize(std::vector<float> multiplier) = 0;

        virtual NeighborListShapes get_neighbor_list_shapes(
            int max_atoms, int inum, int* ilist, int* numneigh,
            bool check_buffers) = 0;

        virtual std::vector<xla::PjRtBuffer*> build_graph(
            xla::PjRtClient* client, int device_id, int inum, int *ilist,
                int *numneigh, int **firstneigh, bool update) = 0;

        virtual bool evaluate_statistics(
            std::map<std::string, std::unique_ptr<xla::PjRtBuffer>> statistics,
            bool check_buffers
        ) { return true; }; // Returns success if not required

        std::vector<std::string> statistics_keys;

    protected:
        Logger logger = Logger::getlogger();

    };

    /**
     * Class that interfaces neighborlists, e.g., from LAMMPS, to a sparse
     * neighbor list in chemtrain.
     */
    class SimpleSparseNeighborList : public GraphBuilder {

        public:
            SimpleSparseNeighborList(std::vector<std::string> statistics)
                : GraphBuilder(statistics) {};

            /**
             * Initializes the interface.
             *
             * @param multipliers: A vector of multipliers specifying the
             *     relative increase of the neighborlist buffers. Only the
             *     first element is used.
             */
            void initialize(std::vector<float> multipliers) override;

            /**
             * Returns the required shapes and types of the sparse neighborlist
             * based on the reference neighborlist.
             *
             * @param max_atoms: The maximum number of atoms in the system.
             * @param inum: The number of local atoms.
             * @param ilist: Array holding the atom indices included in the
             *     neighbor list.
             * @param numneigh: Array holding the number of neighbors for each
             *     atom.
             *
             * @returns Returns ``NeighborListShapes`` struct with necessary
             *     dimensions.
             */
            NeighborListShapes get_neighbor_list_shapes(
                int max_atoms, int inum, int* ilist, int* numneigh,
                bool check_buffers) override;

            /**
              * Builds the sparse neighborlist from the reference neighborlist.
              *
              * @param client: The PjRt client to allocate buffers.
              * @param device_id: The device ID on which to allocate buffers.
              * @param inum: The number of local atoms.
              * @param ilist: Array holding the indices of the senders of
              *     each neighborlist entry.
              * @param numneigh: Array holding the number of neighbors for each
              *     atom.
              * @param firstneigh: Array holding the index of the neighbors
              *     (receivers) for each sender.
              * @param update: Whether the neighbor list data must be updated.
              *
              * @returns A vector holding references to the buffers.
              */
            std::vector<xla::PjRtBuffer*> build_graph(
                xla::PjRtClient* client, int device_id, int inum, int *ilist,
                int *numneigh, int **firstneigh, bool update) override;

            bool evaluate_statistics(
                std::map<std::string, std::unique_ptr<xla::PjRtBuffer>> statistics,
                bool check_buffers
            ) override;

        private:
            float edge_multiplier;

            std::unique_ptr<xla::Literal> senders_literal;
            std::unique_ptr<xla::Literal> receivers_literal;
            std::unique_ptr<xla::Literal> neighbors_literal;

            std::unique_ptr<xla::PjRtBuffer> senders_buffer;
            std::unique_ptr<xla::PjRtBuffer> receivers_buffer;
            std::unique_ptr<xla::PjRtBuffer> neighbors_buffer;

            int edge_buffer_size = 0;
            int n_valid_edges = 1; // At least one edge should be present
            int fill_value = 0;

    };


    /**
     * Class that interfaces neighborlists, e.g., from LAMMPS, to a dense
     * neighbor list in chemtrain.
     */
    class SimpleDenseNeighborList : public GraphBuilder {

        public:
            SimpleDenseNeighborList(std::vector<std::string> statistics)
                : GraphBuilder(statistics) {};

            /**
             * Initializes the interface.
             *
             * @param multipliers: A vector of multipliers specifying the
             *     relative increase of the neighborlist buffers. Only the
             *     first element is used.
             */
            void initialize(std::vector<float> multipliers) override;

            /**
             * Returns the required shapes and types of the sparse neighborlist
             * based on the reference neighborlist.
             *
             * @param max_atoms: The maximum number of atoms in the system.
             * @param inum: The number of local atoms.
             * @param ilist: Array holding the atom indices included in the
             *     neighbor list.
             * @param numneigh: Array holding the number of neighbors for each
             *     atom.
             *
             * @returns Returns ``NeighborListShapes`` struct with necessary
             *     dimensions.
             */
            NeighborListShapes get_neighbor_list_shapes(
                int max_atoms, int inum, int* ilist, int* numneigh,
                bool check_buffers) override;

            /**
              * Builds the sparse neighborlist from the reference neighborlist.
              *
              * @param client: The PjRt client to allocate buffers.
              * @param device_id: The device ID on which to allocate buffers.
              * @param inum: The number of local atoms.
              * @param ilist: Array holding the indices of the senders of
              *     each neighborlist entry.
              * @param numneigh: Array holding the number of neighbors for each
              *     atom.
              * @param firstneigh: Array holding the index of the neighbors
              *     (receivers) for each sender.
              * @param update: Whether the neighbor list data must be updated.
              *
              * @returns A vector holding references to the buffers.
              */
            std::vector<xla::PjRtBuffer*> build_graph(
                xla::PjRtClient* client, int device_id, int inum, int *ilist,
                int *numneigh, int **firstneigh, bool update) override;

            bool evaluate_statistics(
                std::map<std::string, std::unique_ptr<xla::PjRtBuffer>> statistics,
                bool check_buffers
            ) override;

        private:
            float buffer_multiplier;

            std::unique_ptr<xla::Literal> neighbor_literal;
            std::unique_ptr<xla::Literal> edge_literal;
            std::unique_ptr<xla::Literal> triplet_literal;

            std::unique_ptr<xla::PjRtBuffer> neighbor_buffer;
            std::unique_ptr<xla::PjRtBuffer> edge_buffer;
            std::unique_ptr<xla::PjRtBuffer> triplet_buffer;

            int neighbor_buffer_size_ = 0;

            int n_valid_edges_ = 2; // At least one edge should be present
            int n_valid_triplets_ = 1;

            int max_atoms_ = 0;

    };


    class DeviceSparseNeighborList : public GraphBuilder {

        public:
            DeviceSparseNeighborList(std::vector<std::string> statistics)
                : GraphBuilder(statistics) {};

            /**
             * Initializes the interface.
             *
             * @params multipliers: A vector of multipliers specifying the
             *     relative increase of the neighborlist buffers.
             *     TODO: Document the required multipliers.
             */
            void initialize(std::vector<float> multipliers) override;

            /**
             * Returns the required shapes and types of the sparse neighborlist
             * based on the reference neighborlist.
             *
             * @param max_atoms: The maximum number of atoms in the system.
             * @param inum: The number of local atoms.
             * @param ilist: Array holding the atom indices included in the
             *     neighbor list.
             * @param numneigh: Array holding the number of neighbors for each
             *     atom.
             *
             * @returns Returns ``NeighborListShapes`` struct with necessary
             *     dimensions.
             */
            NeighborListShapes get_neighbor_list_shapes(
                int max_atoms, int inum, int* ilist, int* numneigh,
                bool check_buffers) override;

            /**
              * Builds the sparse neighborlist from the reference neighborlist.
              *
              * @param client: The PjRt client to allocate buffers.
              * @param device_id: The device ID on which to allocate buffers.
              * @param inum: The number of local atoms.
              * @param ilist: Array holding the indices of the senders of
              *     each neighborlist entry.
              * @param numneigh: Array holding the number of neighbors for each
              *     atom.
              * @param firstneigh: Array holding the index of the neighbors
              *     (receivers) for each sender.
              * @param update: Whether the neighbor list should be regenerated,
              *     e.g., due to newly communicated atoms.
              *
              * @returns A vector holding references to the buffers.
              *
              */
            std::vector<xla::PjRtBuffer*> build_graph(
                xla::PjRtClient* client, int device_id, int inum, int *ilist,
                int *numneigh, int **firstneigh, bool update) override;

            /**
             * Evaluates the statistics of the neighborlist computation and
             * increase buffer dimensions if necessary.
             *
             * This class generates the neighbor list completely on the device
             * using a cell list. Therefore, it can be necessary to increase
             * the buffers of the cell list and neighborlist or the dimension
             * of the cell grid. Cells must be at least as large as the
             * cutoff radius, but their size might change in NPT simulations.
             *
             * @param results: A vector of vectors holding references to all
             *     result buffers.
             *
             * @returns True if the neighborlist generation was sucessful and
             *     no overflow occured.
             */
            bool evaluate_statistics(
                std::map<std::string, std::unique_ptr<xla::PjRtBuffer>> statistics,
                bool check_buffers
            ) override;

        private:

            float edge_multiplier;
            float capacity_multiplier;

            // This will trigger an overflow during the first execution
            // and return better estimates

            // The number of computations (tested neighbors) is
            // 27 * capacity^2 * nx * ny * nz. We should find a
            // tradeoff or avoid computing the neighbor list too often.

            int n_edges = 729;
            int n_cells_x = 20;
            int n_cells_y = 20;
            int n_cells_z = 20;
            int capacity = 10;

            std::unique_ptr<xla::Literal> update_lit;
            std::unique_ptr<xla::Literal> xcells_lit;
            std::unique_ptr<xla::Literal> ycells_lit;
            std::unique_ptr<xla::Literal> zcells_lit;
            std::unique_ptr<xla::Literal> capacity_lit;
            std::unique_ptr<xla::Literal> senders_lit;

            std::unique_ptr<xla::PjRtBuffer> update_buffer;
            std::unique_ptr<xla::PjRtBuffer> xcells_buffer;
            std::unique_ptr<xla::PjRtBuffer> ycells_buffer;
            std::unique_ptr<xla::PjRtBuffer> zcells_buffer;
            std::unique_ptr<xla::PjRtBuffer> capacity_buffer;
            std::unique_ptr<xla::PjRtBuffer> senders_buffer;
            std::unique_ptr<xla::PjRtBuffer> receivers_buffer;

            bool adjust_dimension(std::unique_ptr<xla::Literal>& cells, int size, xla::PrimitiveType type);

    };

} // namespace jcn

#endif //GRAPH_BUILDER_H
