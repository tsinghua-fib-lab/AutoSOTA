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

#include <chrono>
#include <cmath>
#include <numeric>

#include "connector/domain.h"
#include "connector/buffer.h"
#include "connector/utils.h"

#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/literal_util.h"



namespace jcn {

    AtomBuilder::AtomBuilder(float atom_multiplier, bool newton, const std::vector<std::string>& quantities)
        : max_atoms(0), atom_multiplier(atom_multiplier), quantities(quantities) {
            newton_literal = std::make_unique<xla::Literal>(
                xla::LiteralUtil::CreateR0<bool>(newton)
            );
    }

	AtomShapes AtomBuilder::get_shapes(int inum, int gnum, bool check_buffers) {

        bool reallocate = false;

        // Check whether buffer overflowed (always recompile) or whether
        // buffer is close to beeing full when asked to check buffers
        bool buffer_overflow = (inum + gnum) > max_atoms;
        bool buffer_filled = (inum + gnum) > static_cast<int>(
            std::ceil(max_atoms / std::sqrt(atom_multiplier))
        );
        buffer_filled &= check_buffers;

        if (buffer_filled) {
            logger.log(LogLevel::INFO, "Domain: Recompile atom buffer due to diminished capacity.");
        }

        if (buffer_overflow || buffer_filled) {
            max_atoms = static_cast<int>(std::ceil(atom_multiplier * (inum + gnum)));
            reallocate = true;
        }

        // Only reallocate new memory if required
        if (!position_literal || reallocate) {
            xla::Shape position_shape = xla::ShapeUtil::MakeShape(
               xla::F32, absl::Span<const int64_t>{max_atoms, 3});
            xla::Shape species_shape = xla::ShapeUtil::MakeShape(
               xla::S32, absl::Span<const int64_t>{max_atoms,});

            xla::Shape count_shape = xla::ShapeUtil::MakeShape(
               xla::S32, absl::Span<const int64_t>{});

            position_literal = std::make_unique<xla::Literal>(xla::Literal::CreateFromShape(position_shape));
            species_literal = std::make_unique<xla::Literal>(xla::Literal::CreateFromShape(species_shape));
            locals_literal = std::make_unique<xla::Literal>(xla::Literal::CreateFromShape(count_shape));
            ghosts_literal = std::make_unique<xla::Literal>(xla::Literal::CreateFromShape(count_shape));
        }

        return AtomShapes{max_atoms, reallocate};

	}

    std::vector<xla::PjRtBuffer*> AtomBuilder::build_domain(xla::PjRtClient* client, int device_id, int inum, int gnum, double **x, int *type) {

        // If number of atoms in domain (including ghost) exceeds the allocated
        // buffers

        Logger logger = Logger::getlogger();

        auto start = std::chrono::high_resolution_clock::now();

        if (!position_literal || (inum + gnum) > max_atoms) {
            throw std::runtime_error("Domain not initialized or too many atoms");
        }

        if (logger.log(LogLevel::DEBUG)) {
            logger.log(LogLevel::DEBUG, "Positions: " + position_literal->ToString());
            logger.log(LogLevel::DEBUG, "Species: " + species_literal->ToString());
        }

        float *position_data = position_literal->data<float>().data();
        int *species_data = species_literal->data<int>().data();

        // Collect data for all local atoms and ghost atoms
        for (int i = 0; i < inum + gnum; i++) {
            std::transform(x[i], x[i] + 3, position_data + i * 3, [](double t) { return static_cast<float>(t); });
        }
        std::fill(position_data + (inum + gnum) * 3, position_data + max_atoms * 3, 0.0f);

        // Adjust species values
        std::transform(type, type + inum + gnum, species_data, [](int t) { return t - 1; });
        std::fill(species_data + inum + gnum, species_data + max_atoms, 0);

        // Provide information about number of ghost and number of local atoms
        locals_literal->data<int>().data()[0] = inum;
        ghosts_literal->data<int>().data()[0] = gnum;


        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        logger.log(LogLevel::DEBUG, "Time taken for atom literal creation: " + std::to_string(duration.count()) + " seconds");

        // Create the buffers
     	// TODO: Maybe explicit deallocation is required

        buffers.clear();
        buffers.push_back(create_buffer(client, device_id, position_literal.get()));
        buffers.push_back(create_buffer(client, device_id, species_literal.get()));
        buffers.push_back(create_buffer(client, device_id, locals_literal.get()));
        buffers.push_back(create_buffer(client, device_id, ghosts_literal.get()));
        buffers.push_back(create_buffer(client, device_id, newton_literal.get()));

        std::vector<xla::PjRtBuffer*> buffer_ptrs;
        for (int i = 0; i < buffers.size(); i++) {
            buffer_ptrs.push_back(buffers[i].get());
        }

        return buffer_ptrs;

    }

    double AtomBuilder::evaluate_domain(bool success, int inum, int gnum, double **f,
        std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>>& results,
        std::vector<double>& per_atom_potential) {

        double potential = 0.0;
        per_atom_potential.clear();

        Logger logger = Logger::getlogger();

        if (success) {
            auto start = std::chrono::high_resolution_clock::now();

            // Try to map returned buffers to quantities using stored proto keys.
            int energy_idx = -1;
            int force_idx = -1;
            for (int i = 0; i < static_cast<int>(quantities.size()); ++i) {
                if (quantities[i] == "U") energy_idx = i;
                if (quantities[i] == "F") force_idx = i;
            }

            // Require explicit mapping from the exported model proto quantities.
            // The `quantities` vector must contain entries 'F' and 'U' and the
            // connector will strictly map those keys to the returned result
            // buffers.
            if (force_idx < 0 || energy_idx < 0) {
                throw std::runtime_error("Model proto does not list required quantities 'F' and 'U'.");
            }

            absl::StatusOr<std::shared_ptr<xla::Literal>> force_literal = results[0][force_idx]->ToLiteralSync();
            absl::StatusOr<std::shared_ptr<xla::Literal>> energy_literal = results[0][energy_idx]->ToLiteralSync();

            if (!force_literal.ok() || !energy_literal.ok()) {
                throw std::runtime_error("Failed to convert buffer to literal for F/U using proto indices");
            }

            float *force_data = force_literal.value()->data<float>().data();
            float *potential_data = energy_literal.value()->data<float>().data();

            // If newton is set, the forces are computed for all atoms and
            // communicated later between domains. Otherwise, we write back the
            // complete forces for all atoms
            int max_atoms = inum;
            if (newton_literal.get()->data<bool>().data()[0]) {
                max_atoms += gnum;
            }

            for (int i = 0; i < max_atoms; i++) {
                std::transform(force_data + 3 * i, force_data + 3 * (i + 1),
                    f[i], [](float t) { return static_cast<double>(t); });
            }

            if (logger.log(LogLevel::DEBUG)) {
                logger.log(LogLevel::DEBUG, "Force literal: " + force_literal.value()->ToString());
            }

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;

            logger.log(
                LogLevel::DEBUG, "Time taken for force backtransfer: " 
                + std::to_string(duration.count()) + " seconds"
            );

            // The exported energy buffer already contains one contribution per
            // atom. Preserve the local entries for LAMMPS compute pe/atom and
            // sum the same values for the domain's global-energy contribution.
            per_atom_potential.assign(potential_data, potential_data + inum);
            potential = std::accumulate(
                per_atom_potential.begin(), per_atom_potential.end(), 0.0,
                [](double sum, float v){ return sum + static_cast<double>(v); }
            );

            // Remove the buffers after computation
            buffers.clear();

        }

        return potential;

    }

} // namespace jcn
