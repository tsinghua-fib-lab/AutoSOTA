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

#include <vector>
#include <string>
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/literal_util.h"

#include "connector/utils.h"

#ifndef DOMAIN_H
#define DOMAIN_H

namespace jcn {

	/**
     * Contains all per-atom data of a domnain
     */
    struct AtomShapes {
        /**
  		 * Maximum possible number of atoms in the domain, including local, ghost,
         * and padded atoms.
         */
        int n_atoms;

        /**
  		 * Tracks change of maxmimum number of atoms in the domain. If true,
         * must re-compile the mlir module with new shapes.
         */
        bool reallocate;

    };

    /**
     * Transforms atom data from the local domain into a padded XLA-copmpatible format
     */
    class AtomBuilder {
    public:
         /**
          * Constructor
          * @param atom_multiplier Fraction of extra atoms to consider when re-allocate
    	  *     the arrays
    	  * @params newton True if the forces should be computed according to
    	  *     LAMMPS newton setting
          */
        /**
         * Constructor
         * @param atom_multiplier Fraction of extra atoms to consider when re-allocate
         *     the arrays
         * @param newton True if the forces should be computed according to
         *     LAMMPS newton setting
         * @param quantities Vector of quantity keys exported in the model proto
         *     (e.g., ["F","U"]). The connector uses these keys to map
         *     returned result buffers to named quantities. Mapping is strict;
         *     if required keys are missing, evaluation will fail.
         */
        AtomBuilder(float atom_multiplier, bool newton, const std::vector<std::string>& quantities = {});
        ~AtomBuilder() = default;

        AtomShapes get_shapes(int inum, int gnum, bool check_buffers);

        /**
         * Writes atom positions into device buffers. Padds the atom data to
         * reduce the number of recompilations.
         *
         * @param client PjRt client to allocate buffers
         * @param device_id Device ID to allocate buffers
         * @param inum Number of local atoms
         * @param gnum Number of ghost atoms
         * @param x Pointer to atom positions
         * @param type Pointer to atom types (one-based species)
         *
         * @return A vector holding references to the buffers
         */
        std::vector<xla::PjRtBuffer*> build_domain(xla::PjRtClient* client, int device_id, int inum, int gnum, double **x, int *type);

        /**
         * Writes back the force to the original array and returns the potential.
         *
         * Notes:
         * - This function strictly maps the returned result buffers to the
         *   exported `quantities` provided at construction. The proto must
         *   contain entries for 'F' (forces) and 'U' (per-atom energies).
         * - No heuristic fallback is performed; missing or mismatched
         *   quantities will cause an error to be raised.
         *
         * @param success True if the computation was successful, i.e., the
         *     neighbor list did not overflow.
         * @param inum Number of local atoms
         * @param gnum Number of ghost atoms
         * @param f Pointer to target force array
         * @param results Vector of vector of pointers to the result buffers.
         * @param per_atom_potential Receives the local per-atom energies.
         *
         * @return The potential energy of the system
         */
        double evaluate_domain(bool success, int inum, int gnum, double **f,
            std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>>& results,
            std::vector<double>& per_atom_potential);

        const std::vector<std::string>& get_quantities() const { return quantities; }

    private:
        int max_atoms;

        std::unique_ptr<xla::Literal> newton_literal;

        std::unique_ptr<xla::Literal> position_literal;
        std::unique_ptr<xla::Literal> species_literal;
        std::unique_ptr<xla::Literal> locals_literal;
        std::unique_ptr<xla::Literal> ghosts_literal;

        std::vector<std::unique_ptr<xla::PjRtBuffer>> buffers;

        float atom_multiplier;
        std::vector<std::string> quantities;

        Logger logger = Logger::getlogger();

    };

} // namespace jcn

#endif //DOMAIN_H
