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

#include <chrono>
#include <bitset>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <limits>

#include "connector/graph_builder.h"
#include "connector/buffer.h"
#include "connector/utils.h"

namespace jcn {

    namespace {
        constexpr int kLammpsNeighborMask = 0x3FFFFFFF;
    }

    GraphBuilder::GraphBuilder(
        std::vector<std::string> statistics
        ) : statistics_keys(statistics) {}

    /** Simple sparse neighbor list ******************************************/

    void SimpleSparseNeighborList::initialize(std::vector<float> multipliers) {
        edge_multiplier = multipliers[0];
    }

    NeighborListShapes SimpleSparseNeighborList::get_neighbor_list_shapes(
        int max_atoms, int inum, int* ilist, int* numneigh, bool check_buffers) {
        // We pass the neighbor list of LAMMPS

        Logger logger = Logger::getlogger();

        fill_value = max_atoms;

        // No reallocation necessary if buffers sufficiently large
        bool reallocate = false;
        if (!senders_literal || !receivers_literal || !neighbors_literal) {
            reallocate = true;
        }

        // Count the number of edges
        int current_edges = 0;
        for (int ii = 0; ii < inum; ii++) {
            const int i = ilist[ii];
            if (std::numeric_limits<int>::max() - current_edges < numneigh[i]) {
                throw std::runtime_error("Exceeded maximum number of edges");
            }

            if (numneigh[i] < 0) {
                throw std::runtime_error("Number of neighbors must be positive");
            }

            current_edges += numneigh[i];
        }

        // Check whether buffer overflowed (always recompile) or whether
        // buffer is close to beeing full when asked to check buffers
        bool buffer_overflow = current_edges > edge_buffer_size;
        bool buffer_filled = current_edges > static_cast<int>(
            std::ceil(edge_buffer_size / std::sqrt(edge_multiplier))
        );
        buffer_filled &= check_buffers;

        if (buffer_filled) {
            logger.log(LogLevel::INFO, "SimpleSparseNeighborList: Recompiled neighborlist buffer due to diminished capacity.");
        }

        if (buffer_overflow || buffer_filled) {
            logger.log(LogLevel::INFO, "Reallocation necessary, current edges are " + std::to_string(current_edges));
            logger.log(LogLevel::INFO, "Increasing edge count by multiplier " + std::to_string(edge_multiplier));
            edge_buffer_size = static_cast<int>(std::ceil(current_edges * edge_multiplier));
            reallocate = true;
        }

        // Check whether more edges are valid
        if (neighbors_literal) {
            reallocate |= n_valid_edges > neighbors_literal->shape().dimensions(0);
        }

        if (reallocate) {
            logger.log(LogLevel::INFO, "Reallocating to " + std::to_string(edge_buffer_size) + " edges");

            xla::Shape shape = xla::ShapeUtil::MakeShape(
                xla::S32, absl::Span<const int64_t>{edge_buffer_size});
            xla::Shape max_nbrs_shape = xla::ShapeUtil::MakeShape(
                xla::PRED, absl::Span<const int64_t>{n_valid_edges});

            senders_literal = std::make_unique<xla::Literal>(xla::Literal::CreateFromShape(shape));
            receivers_literal = std::make_unique<xla::Literal>(xla::Literal::CreateFromShape(shape));
            neighbors_literal = std::make_unique<xla::Literal>(xla::Literal::CreateFromShape(max_nbrs_shape));
        }

        std::vector<std::vector<int64_t>> graph_shapes = {{edge_buffer_size}, {edge_buffer_size}, {n_valid_edges}};
        std::vector<xla::PrimitiveType> graph_types = {xla::S32, xla::S32, xla::PRED};

        return NeighborListShapes{graph_shapes, graph_types, reallocate};

    }


    std::vector<xla::PjRtBuffer*> SimpleSparseNeighborList::build_graph(
            xla::PjRtClient* client, int device_id, int inum, int *ilist, int *numneigh, int **firstneigh, bool update) {

        if (update) {

            Logger logger = Logger::getlogger();

            // Clear old buffers
            if (senders_buffer) {
                senders_buffer->Delete();
                receivers_buffer->Delete();
                neighbors_buffer->Delete();
            }

            auto start = std::chrono::high_resolution_clock::now();

            // Only update the values if the shape or content of the neighbor list changed
            int* senders_data = senders_literal->data<int>().data();
            int* receivers_data = receivers_literal->data<int>().data();

            int max_senders = senders_literal->shape().dimensions(0);
            int max_receivers = receivers_literal->shape().dimensions(0);

            if (max_receivers != max_senders) {
                throw std::runtime_error("Senders and receivers must have the same size");
            }
            if (max_receivers != edge_buffer_size) {
                throw std::runtime_error("Senders and receivers must have the same size as the edge buffer");
            }
            if (max_senders != edge_buffer_size) {
                throw std::runtime_error("Senders and receivers must have the same size as the edge buffer");
            }

            // Fill in the sender and receiver values
            int edge_counter = 0;
            for (int ii = 0; ii < inum; ii++) {
                const int i = ilist[ii];
                int num_neighbors = numneigh[i];
                int* firstneigh_ptr = firstneigh[i];

                if (edge_counter + num_neighbors > max_senders) {
                    throw std::runtime_error("Exceeded maximum number of senders");
                }
                if (edge_counter + num_neighbors > max_receivers) {
                    throw std::runtime_error("Exceeded maximum number of receivers");
                }

                // LAMMPS stores a neighbor list as central atom i followed by
                // its neighbors j. MACE uses edge_index[0] as the message
                // source and edge_index[1] as the receiving central atom, so
                // the deployed edge must be j -> i (not i -> j).
                std::fill(receivers_data + edge_counter,
                          receivers_data + edge_counter + num_neighbors, i);

                for (int jj = 0; jj < num_neighbors; jj++) {
                    senders_data[edge_counter + jj] =
                        firstneigh_ptr[jj] & kLammpsNeighborMask;
                }

                edge_counter += num_neighbors;
            }

            // Fill in the invalid values
            std::fill(senders_data + edge_counter, senders_data + edge_buffer_size, fill_value);
            std::fill(receivers_data + edge_counter, receivers_data + edge_buffer_size, fill_value);

            if (std::getenv("JCN_COMM_DEBUG") != nullptr && edge_counter > 0) {
                int ghost_sources = 0;
                int ghost_receivers = 0;
                for (int edge = 0; edge < edge_counter; ++edge) {
                    ghost_sources += senders_data[edge] >= inum;
                    ghost_receivers += receivers_data[edge] >= inum;
                }
                std::cerr << "[COMM GRAPH] edges=" << edge_counter
                          << " first=" << senders_data[0] << "->"
                          << receivers_data[0]
                          << " ghost_sources=" << ghost_sources
                          << " ghost_receivers=" << ghost_receivers
                          << std::endl;
            }

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;
            logger.log(LogLevel::DEBUG, "Time taken for neighborlist array creation: " + std::to_string(duration.count()) + " seconds");

            if (logger.log(LogLevel::DEBUG)) {
                logger.log(LogLevel::DEBUG, "Senders: " + senders_literal->ToString());
                logger.log(LogLevel::DEBUG, "Receivers: " + receivers_literal->ToString());
            }

            // Create buffers
            senders_buffer = create_buffer(client, device_id, senders_literal.get());
            receivers_buffer = create_buffer(client, device_id, receivers_literal.get());
            neighbors_buffer = create_buffer(client, device_id, neighbors_literal.get());
        }

        // Return pointers to the buffers.
        std::vector<xla::PjRtBuffer*> buffer_ptrs;
        buffer_ptrs.push_back(senders_buffer.get());
        buffer_ptrs.push_back(receivers_buffer.get());
        buffer_ptrs.push_back(neighbors_buffer.get());

        return buffer_ptrs;

    };


    bool SimpleSparseNeighborList::evaluate_statistics(
        std::map<std::string, std::unique_ptr<xla::PjRtBuffer>> statistics,
        bool check_buffers) {

        bool success = true;

	    // Check if more valid edges are necessary
        absl::StatusOr<std::shared_ptr<xla::Literal>> valid_edges = statistics["max_neighbors"]->ToLiteralSync();
        absl::StatusOr<std::shared_ptr<xla::Literal>> overlong = statistics["overlong"]->ToLiteralSync();

    	if (!valid_edges.ok() || !overlong.ok()) {
        	std::cerr << "Error: Failed to convert PjRtBuffer to Literal" << std::endl;
       		return false;
    	}

    	if (valid_edges.value() == nullptr || overlong.value() == nullptr) {
        	std::cerr << "Error: Literal is null" << std::endl;
        	return false;
    	}

        int req_valid_edges = valid_edges.value()->data<int>().data()[0];

        // Check whether buffer overflowed (always recompile) or whether
        // buffer is close to beeing full when asked to check buffers
        bool buffer_overflow = req_valid_edges > n_valid_edges;
        bool buffer_filled = req_valid_edges > static_cast<int>(
            std::ceil(n_valid_edges / std::sqrt(edge_multiplier))
        );
        buffer_filled &= check_buffers;

        if (buffer_filled) {
            logger.log(LogLevel::INFO,
                "SimpleSparseNeighborList: Recompile edge buffer due to diminished capacity."
            );
        }

        if (buffer_overflow || buffer_filled) {
            logger.log(LogLevel::INFO,
                "SimpleSparseNeighborList: Increasing valid edges from " +
                std::to_string(n_valid_edges) + " to " +
                std::to_string(req_valid_edges)
            );
            n_valid_edges = static_cast<int>(std::ceil(req_valid_edges * edge_multiplier));

            // Ensure that not greater that maximum possible amount.
            // Provided edges are from undirected graph
            if (n_valid_edges > 2 * edge_buffer_size) {
                n_valid_edges = 2 * edge_buffer_size;
            }

            success = false;
        }

        return success;

    }

    /** Simple Dense Neighbor List *******************************************/

    void SimpleDenseNeighborList::initialize(std::vector<float> multipliers) {
        buffer_multiplier = multipliers[0];
    }


    NeighborListShapes SimpleDenseNeighborList::get_neighbor_list_shapes(
        int max_atoms, int inum, int* ilist, int* numneigh, bool check_buffers) {
        // We pass the neighbor list of LAMMPS

        Logger logger = Logger::getlogger();

        // No reallocation necessary if buffers sufficiently large
        bool reallocate = false;
        if (!neighbor_literal || !edge_literal || !triplet_literal) {
            reallocate = true;
        }

        // Check whether the number of atoms (including padding atoms) increased
        if (max_atoms > max_atoms_) {
            max_atoms_ = max_atoms;
            reallocate = true;
        }

        // Determine the maximum number of neighbors
        int max_neighbors = 0;
        for (int ii = 0; ii < inum; ii++) {
            const int i = ilist[ii];
            max_neighbors = std::max(max_neighbors, numneigh[i]);
        }

        // Check whether buffer overflowed (always recompile) or whether
        // buffer is close to beeing full when asked to check buffers
        bool buffer_overflow = max_neighbors > neighbor_buffer_size_;
        bool buffer_filled = max_neighbors > static_cast<int>(
            std::ceil(neighbor_buffer_size_ / std::sqrt(buffer_multiplier))
        );
        buffer_filled &= check_buffers;

        if (buffer_filled) {
            logger.log(LogLevel::INFO, "Recompiled buffer due to diminished capacity.");
        }

        if (buffer_overflow || buffer_filled) {
            logger.log(LogLevel::INFO, "Reallocation necessary, current neighbor buffer is " + std::to_string(neighbor_buffer_size_));
            neighbor_buffer_size_ = static_cast<int>(std::ceil(max_neighbors * buffer_multiplier));
            reallocate = true;
        }

        // Check whether more edges or triplets are valid
        if (neighbor_literal) {
            reallocate |= n_valid_edges_ > edge_literal->shape().dimensions(0);
            reallocate |= n_valid_triplets_ > triplet_literal->shape().dimensions(0);
        }

        if (reallocate) {
            logger.log(LogLevel::INFO, "Reallocating to "
                + std::to_string(n_valid_edges_) + " edges and "
                + std::to_string(n_valid_triplets_) + " triplets");

            xla::Shape neighbor_shape = xla::ShapeUtil::MakeShape(
                xla::S32, absl::Span<const int64_t>{max_atoms_, neighbor_buffer_size_});
            xla::Shape edge_shape = xla::ShapeUtil::MakeShape(
                xla::PRED, absl::Span<const int64_t>{n_valid_edges_});
            xla::Shape triplet_shape = xla::ShapeUtil::MakeShape(
                xla::PRED, absl::Span<const int64_t>{n_valid_triplets_});

            neighbor_literal = std::make_unique<xla::Literal>(xla::Literal::CreateFromShape(neighbor_shape));
            edge_literal = std::make_unique<xla::Literal>(xla::Literal::CreateFromShape(edge_shape));
            triplet_literal = std::make_unique<xla::Literal>(xla::Literal::CreateFromShape(triplet_shape));
        }

        std::vector<std::vector<int64_t>> graph_shapes = {
            {max_atoms_, neighbor_buffer_size_},
            {n_valid_edges_},
            {n_valid_triplets_}
        };
        std::vector<xla::PrimitiveType> graph_types = {
            xla::S32, xla::PRED, xla::PRED};

        return NeighborListShapes{graph_shapes, graph_types, reallocate};

    }


    std::vector<xla::PjRtBuffer*> SimpleDenseNeighborList::build_graph(
            xla::PjRtClient* client, int device_id, int inum, int *ilist,
            int *numneigh, int **firstneigh, bool update) {

        if (update) {

            Logger logger = Logger::getlogger();

            // Clear old buffers
            if (neighbor_buffer) {
                neighbor_buffer->Delete();
                edge_buffer->Delete();
                triplet_buffer->Delete();
            }

            auto start = std::chrono::high_resolution_clock::now();

            // Only update the values if the shape or content of the neighbor list changed
            int* neighbor_data = neighbor_literal->data<int>().data();

            // Fill in the sender and receiver values
            for (int ii = 0; ii < inum; ii++) {
                const int i = ilist[ii];
                int num_neighbors = numneigh[i];
                int* firstneigh_ptr = firstneigh[i];

                // Copy row by row
                for (int jj = 0; jj < num_neighbors; jj++) {
                    neighbor_data[i * neighbor_buffer_size_ + jj] =
                        firstneigh_ptr[jj] & kLammpsNeighborMask;
                }

                // Fill in the remainder
                std::fill(
                    neighbor_data + i * neighbor_buffer_size_ + num_neighbors,
                    neighbor_data + (i + 1) * neighbor_buffer_size_,
                    max_atoms_
                );
            }

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;
            logger.log(LogLevel::DEBUG, "Time taken for neighborlist array creation: " + std::to_string(duration.count()) + " seconds");

            if (logger.log(LogLevel::DEBUG)) {
                logger.log(LogLevel::DEBUG, "Neighborlist: " + neighbor_literal->ToString());
            }

            // Create buffers
            neighbor_buffer = create_buffer(client, device_id, neighbor_literal.get());
            edge_buffer = create_buffer(client, device_id, edge_literal.get());
            triplet_buffer = create_buffer(client, device_id, triplet_literal.get());
        }

        // Return pointers to the buffers.
        std::vector<xla::PjRtBuffer*> buffer_ptrs;
        buffer_ptrs.push_back(neighbor_buffer.get());
        buffer_ptrs.push_back(edge_buffer.get());
        buffer_ptrs.push_back(triplet_buffer.get());

        return buffer_ptrs;

    };


    bool SimpleDenseNeighborList::evaluate_statistics(
        std::map<std::string, std::unique_ptr<xla::PjRtBuffer>> statistics,
        bool check_buffers) {

        bool success = true;

        // Check if more valid edges or triplets are necessary
        absl::StatusOr<std::shared_ptr<xla::Literal>> valid_edges = statistics["max_neighbors"]->ToLiteralSync();
        absl::StatusOr<std::shared_ptr<xla::Literal>> valid_triplets = statistics["overlong"]->ToLiteralSync();

        int req_valid_edges = valid_edges.value()->data<int>().data()[0];
        int req_valid_triplets = valid_triplets.value()->data<int>().data()[0];

        // Check whether buffer overflowed (always recompile) or whether
        // buffer is close to beeing full when asked to check buffers
        bool edge_buffer_overflow = req_valid_edges > n_valid_edges_;
        bool triplet_buffer_overflow = req_valid_triplets > n_valid_triplets_;
        bool edge_buffer_filled = req_valid_edges > static_cast<int>(
            std::ceil(n_valid_edges_ / std::sqrt(buffer_multiplier))
        );
        bool triplet_buffer_filled = req_valid_triplets > static_cast<int>(
            std::ceil(n_valid_triplets_ / buffer_multiplier)
        );
        edge_buffer_filled &= check_buffers;
        triplet_buffer_filled &= check_buffers;

        if (edge_buffer_filled || triplet_buffer_filled) {
            logger.log(LogLevel::INFO, "Recompiled buffer due to diminished capacity.");
        }

        if (edge_buffer_overflow || edge_buffer_filled) {
            logger.log(LogLevel::INFO,
                "SimpleDenseNeighborList: Increasing valid edges"
            );
            n_valid_edges_ = static_cast<int>(std::ceil(req_valid_edges * buffer_multiplier));

            // Ensure that not greater that maximum possible amount.
            // Provided edges are from undirected graph
            if (n_valid_edges_ > neighbor_buffer_size_ * max_atoms_) {
                n_valid_edges_ = neighbor_buffer_size_ * max_atoms_;
            }

            success = false;
        }
        if (triplet_buffer_overflow || triplet_buffer_filled) {
            logger.log(LogLevel::INFO,
                "SimpleDenseNeighborList: Increasing valid triplets."
            );
            n_valid_triplets_ = static_cast<int>(std::ceil(req_valid_triplets * buffer_multiplier * buffer_multiplier));

            // Ensure that not greater that maximum possible amount.
            // Provided edges are from undirected graph
            if (n_valid_triplets_ > neighbor_buffer_size_ * neighbor_buffer_size_ * max_atoms_) {
                n_valid_triplets_ = neighbor_buffer_size_ * neighbor_buffer_size_ * max_atoms_;
            }

            success = false;
        }

        return success;

    }

    /** Device Sparse Neighbor List ******************************************/


    void DeviceSparseNeighborList::initialize(std::vector<float> multipliers) {
        edge_multiplier = multipliers[0];
        capacity_multiplier = multipliers[1];
    }


    bool DeviceSparseNeighborList::adjust_dimension(std::unique_ptr<xla::Literal>& cells, int size, xla::PrimitiveType type) {

        // Check if evaluation of the statistics increased the number of cells
        if (cells && cells->shape().dimensions(0) >= size) return false;

        // Reallocate the cells
        xla::Shape shape = xla::ShapeUtil::MakeShape(
            type, absl::Span<const int64_t>{size});

        cells = std::make_unique<xla::Literal>(xla::Literal::CreateFromShape(shape));
        // cells->PopulateWithValue(0); // Value is not important

        return true;
    }


    NeighborListShapes DeviceSparseNeighborList::get_neighbor_list_shapes(
         int max_atoms, int inum, int* ilist, int* numneigh, bool check_buffers) {
         // We pass the neighbor list of LAMMPS
         throw std::runtime_error("Not yet implemented.");

         // No reallocation necessary if buffers sufficiently large
         bool reallocate = false;

         // Will not change shape (only a predicate)
         if (!update_lit) {
            xla::Shape shape = xla::ShapeUtil::MakeShape(
                xla::PRED, absl::Span<const int64_t>{1});

            update_lit = std::make_unique<xla::Literal>(xla::Literal::CreateFromShape(shape));
         }

         // Check the cell dimensions (works also for the capacity and senders)
         reallocate |= adjust_dimension(xcells_lit, n_cells_x, xla::PRED);
         reallocate |= adjust_dimension(ycells_lit, n_cells_y, xla::PRED);
         reallocate |= adjust_dimension(zcells_lit, n_cells_z, xla::PRED);
         reallocate |= adjust_dimension(capacity_lit, capacity, xla::PRED);
         reallocate |= adjust_dimension(senders_lit, n_edges, xla::S32);

         std::vector<std::vector<int64_t>> graph_shapes = {
             {1}, {n_cells_x}, {n_cells_y}, {n_cells_z}, {capacity}, {n_edges}, {n_edges}
         };
         std::vector<xla::PrimitiveType> graph_types = {
             xla::PRED, xla::PRED, xla::PRED, xla::PRED, xla::PRED, xla::S32, xla::S32
         };

         return NeighborListShapes{graph_shapes, graph_types, reallocate};

    }


    std::vector<xla::PjRtBuffer*> DeviceSparseNeighborList::build_graph(
        xla::PjRtClient* client, int device_id, int inum, int *ilist,
        int *numneigh, int **firstneigh, bool update) {

        // The update predicate can change
        if (update_buffer) {
            update_buffer->Delete();
        }
        update_lit->PopulateWithValue(update);
        update_buffer = create_buffer(client, device_id, update_lit.get());

        if (update) {
            // Clear old buffers
            if (senders_buffer) {
                xcells_buffer->Delete();
                ycells_buffer->Delete();
                zcells_buffer->Delete();
                capacity_buffer->Delete();
                senders_buffer->Delete();
                receivers_buffer->Delete();
            }

            // Create buffers (value is not important)
            xcells_buffer = create_buffer(client, device_id, xcells_lit.get());
            ycells_buffer = create_buffer(client, device_id, ycells_lit.get());
            zcells_buffer = create_buffer(client, device_id, zcells_lit.get());
            capacity_buffer = create_buffer(client, device_id, capacity_lit.get());

            // When reallocated, these are only placeholders
            senders_buffer = create_buffer(client, device_id, senders_lit.get());
            receivers_buffer = create_buffer(client, device_id, senders_lit.get());
        }

        std::vector<xla::PjRtBuffer*> buffer_ptrs;

        buffer_ptrs.push_back(update_buffer.get());
        buffer_ptrs.push_back(xcells_buffer.get());
        buffer_ptrs.push_back(ycells_buffer.get());
        buffer_ptrs.push_back(zcells_buffer.get());
        buffer_ptrs.push_back(capacity_buffer.get());
        buffer_ptrs.push_back(senders_buffer.get());
        buffer_ptrs.push_back(receivers_buffer.get());

        return buffer_ptrs;
    }


    bool DeviceSparseNeighborList::evaluate_statistics(
        std::map<std::string, std::unique_ptr<xla::PjRtBuffer>> statistics,
        bool check_buffers) {

        throw std::runtime_error("Not yet implemented.");

        bool success = true;

        absl::StatusOr<std::shared_ptr<xla::Literal>> min_cell_capacity = statistics["min_cell_capacity"]->ToLiteralSync();
        absl::StatusOr<std::shared_ptr<xla::Literal>> cell_too_small = statistics["cell_too_small"]->ToLiteralSync();
        absl::StatusOr<std::shared_ptr<xla::Literal>> min_neighbors = statistics["max_neighbors"]->ToLiteralSync();

        if (!min_cell_capacity.ok() || !cell_too_small.ok() || !min_neighbors.ok()) {
            throw std::runtime_error("Failed to convert buffer to literal");
        }

//        std::cout << "Returned statistics:" << std::endl \
//                  << "- Min cell capacity: " << min_cell_capacity.value()->data<int>().data()[0] << std::endl \
//                  << "- Cell too small: " << cell_too_small.value()->data<int>().data()[0] << std::endl \
//                  << "- Min neighbors: " << min_neighbors.value()->data<int>().data()[0] << std::endl;

        // TODO
        // Get back the directions with too small cell sizes
        // std::bitset<3> binary(number);

        // Adjust the capcities of cells and neighborlist
        int req_cell_capacity = min_cell_capacity.value()->data<int>().data()[0];
        if (req_cell_capacity > capacity) {
            capacity = static_cast<int>(std::ceil(req_cell_capacity * capacity_multiplier));
            success = false;
        }

        int req_nbrs_capacity = min_neighbors.value()->data<int>().data()[0];
        if (req_nbrs_capacity > n_edges) {
            n_edges = static_cast<int>(std::ceil(req_nbrs_capacity * edge_multiplier));
            success = false;
        }

        // Store the result buffers and reuse them
        if (receivers_buffer) {
            receivers_buffer->Delete();
        }
        if (senders_buffer) {
            senders_buffer->Delete();
        }

        // We must remove these buffers from the results vector or they
        // will be deleted after the evaluation function finished
        // TODO: Fix
//        receivers_buffer = std::move(results[0].back());
//        results[0].pop_back();
//        senders_buffer = std::move(results[0].back());
//        results[0].pop_back();

        // Returns whether rerun with bigger capacities is necessary
        return success;

    }

} // namespace jcn
