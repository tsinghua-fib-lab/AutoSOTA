/* ----------------------------------------------------------------------------
    chemtrain-deploy - LAMMPS plugin
    Copyright (C) 2025  Multiscale Modeling of Fluid Materials, TU Munich

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    See the LICENSE file in the directory of this file.
---------------------------------------------------------------------------- */
#include "pair_chemtrain_deploy.h"

#include "libconnector.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace LAMMPS_NS;

namespace {

class ProfileRange {
 public:
  explicit ProfileRange(const char *name) {
    // XLA already initializes the NVTX domain used by Nsight. Route optional
    // LAMMPS ranges through the connector so this plugin does not need XLA or
    // NVTX headers and production callbacks retain only one cached branch.
    static const bool enabled = std::getenv("JCN_COMM_PROFILE") != nullptr;
    if (!enabled) return;
    active_ = jcn::PushCommunicationProfileRange(name);
  }

  ~ProfileRange() {
    if (active_) jcn::PopCommunicationProfileRange();
  }

 private:
  bool active_ = false;
};

bool communication_debug_enabled() {
  static const bool enabled = std::getenv("JCN_COMM_DEBUG") != nullptr;
  return enabled;
}

double debug_first_value(void *data,
                         jcn::CommunicationScalarType type,
                         std::int64_t rows,
                         std::int64_t cols) {
  if (data == nullptr || rows <= 0 || cols <= 0) return 0.0;

  if (type == jcn::CommunicationScalarType::F32) {
    return static_cast<double>(static_cast<float *>(data)[0]);
  }

  return static_cast<double *>(data)[0];
}

int checked_communication_count(int n, std::int64_t cols) {
  if (n < 0 || cols <= 0 ||
      static_cast<std::int64_t>(n) >
          std::numeric_limits<int>::max() / cols) {
    throw std::runtime_error(
        "LAMMPS communication buffer count exceeds the integer ABI limit");
  }
  return static_cast<int>(static_cast<std::int64_t>(n) * cols);
}

struct CountSummary {
  int minimum;
  double average;
  int maximum;
  int total;
};

}  // namespace

/* ---------------------------------------------------------------------- */

ChemtrainDeploy::ChemtrainDeploy(LAMMPS *lmp) : Pair(lmp)
{
  writedata = 0;
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
}

/* ---------------------------------------------------------------------- */

ChemtrainDeploy::~ChemtrainDeploy()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut);
  }
}

/* ---------------------------------------------------------------------- */

void ChemtrainDeploy::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  auto start = std::chrono::high_resolution_clock::now();

  // LAMMPS owns neighbor rebuild decisions for the supported host-neighbor-list
  // path. The old explicit displacement scan existed for the removed
  // DeviceSparseNeighborList path and would add redundant work here.
  bool update_list = (neighbor->ago == 0);

  // Number of sender atoms can change depending on the ghost setting of the neighbor list.
  int inum = list->inum + list->gnum;

  int retry_flag = 0;
  jcn::Results results;

  try {
    results = connector->compute_force(
      atom->nlocal, atom->nghost, atom->x, atom->f, atom->type,
      inum, list->ilist, list->numneigh, list->firstneigh, update_list, false
    );
  } catch (const jcn::RecompilationRequired& e) {
    retry_flag = 1;
  }

  int retry_flag_all;
  MPI_Allreduce(&retry_flag, &retry_flag_all, 1, MPI_INT, MPI_MAX, world);

  // If one device must recompile, give all other devices the possibility to recompile too.
  if (retry_flag_all > 0) {
    results = connector->compute_force(
      atom->nlocal, atom->nghost, atom->x, atom->f, atom->type,
      inum, list->ilist, list->numneigh, list->firstneigh, update_list, true
    );
  }

  // Scale the forces.
  if (scale != 1.0) {
    double **f = atom->f;
    for (int i = 0; i < inum; i++) {
      f[i][0] *= scale;
      f[i][1] *= scale;
      f[i][2] *= scale;
    }
  }

  // Pass the evaluated potential energy to LAMMPS.
  if (eflag) {
    eng_vdwl = scale * results.potential;
  }
  if (eflag_atom) {
    if (results.per_atom_potential.size() !=
        static_cast<std::size_t>(atom->nlocal)) {
      error->all(
          FLERR, "Chemtrain per-atom energy count does not match local atoms");
    }
    // ev_init() allocated and cleared eatom for this force evaluation. Add the
    // model's local atomic contributions so standard LAMMPS consumers such as
    // compute pe/atom and custom dumps can inspect the unsummed predictions.
    for (int i = 0; i < atom->nlocal; ++i) {
      eatom[i] += scale * results.per_atom_potential[i];
    }
  }

  flops += static_cast<double>(results.stats.flops);
  compilations += results.stats.compilations;
  initial_compilations += results.stats.initial_compilations;
  atom_recompilations += results.stats.atom_recompilations;
  edge_recompilations += results.stats.edge_recompilations;

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  (void)duration;

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void ChemtrainDeploy::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(cut, n + 1, n + 1, "pair:cut");

  for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= n; j++) setflag[i][j] = 0;
  }
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void ChemtrainDeploy::settings(int narg, char **arg)
{
  if (narg < 1) error->all(FLERR, "Illegal jax_connect command");

  // pair_style chemtrain_deploy <backend> [memory_fraction] [comm on|off]
  //
  // `backend` selects the PJRT implementation (normally `cuda`) and the
  // optional fraction limits PJRT's device-memory pool. `comm on` selects the
  // exported graph variant containing intermediate feature gathers; `comm
  // off` selects the ordinary graph that evaluates the complete local/ghost
  // environment without those gathers. Communication is off by default.
  jcn::ConnectorConfig config;
  communication_enabled = false;

  // Assign devices based on local rank.
  int device_id = 0;
  char* local_rank;

  if ((local_rank = getenv("SLURM_LOCALID"))) {
    device_id = std::stoi(local_rank);
    utils::logmesg(lmp, "Assign device based on SLURM_LOCALID");
  }
  if ((local_rank = getenv("OMPI_COMM_WORLD_LOCAL_RANK"))) {
    utils::logmesg(lmp, "Assign device based on OMPI_COMM_WORLD_LOCAL_RANK");
    device_id = std::stoi(local_rank);
  }
  if ((local_rank = getenv("MV2_COMM_WORLD_LOCAL_RANK"))) {
    utils::logmesg(lmp, "Assign device based on MV2_COMM_WORLD_LOCAL_RANK");
    device_id = std::stoi(local_rank);
  }
  if ((local_rank = getenv("FLUX_TASK_LOCAL_ID"))) {
    utils::logmesg(lmp, "Assign device based on FLUX_TASK_LOCAL_ID");
    device_id = std::stoi(local_rank);
  }
  if ((local_rank = getenv("PMI_LOCAL_RANK"))) {
    utils::logmesg(lmp, "Assign device based on PMI_LOCAL_RANK");
    device_id = std::stoi(local_rank);
  }

  config.backend = std::string(arg[0]);
  config.device = device_id;

  // Record which pre-exported model variant pair_coeff should load. The
  // executable is selected later, after the model file has been read.
  int option = 1;
  if (option < narg && std::string(arg[option]) != "comm") {
    config.memory_fraction = std::stof(arg[option++]);
  }
  while (option < narg) {
    if (std::string(arg[option]) != "comm" || option + 1 >= narg) {
      error->all(FLERR,
                 "Expected 'comm on' or 'comm off' in pair_style "
                 "chemtrain_deploy settings");
    }
    const std::string value = arg[option + 1];
    if (value == "on") {
      communication_enabled = true;
    } else if (value == "off") {
      communication_enabled = false;
    } else {
      error->all(FLERR,
                 "The chemtrain/deploy comm setting must be on or off");
    }
    option += 2;
  }

  try {
    connector = std::make_unique<jcn::Connector>(config);
  } catch (const std::exception& e) {
    std::string msg =
        std::string("chemtrain_deploy: failed to initialize connector: ") +
        e.what();
    error->all(FLERR, msg.c_str());
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void ChemtrainDeploy::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  if (narg < 4) error->all(FLERR, "Illegal jax_connect command");

  std::string exported_model_path = arg[2];

  const float atom_multiplier = std::stof(arg[3]);

  std::vector<float> neighbor_list_multipliers;
  for (int i = 4; i < narg; i++) {
    neighbor_list_multipliers.push_back(std::stof(arg[i]));
  }

  std::ifstream file(exported_model_path);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + exported_model_path);
  }

  std::string exported_model((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());

  jcn::ModelConfig config;

  config.model = exported_model;
  config.neighbor_list_multipliers = neighbor_list_multipliers;
  config.atom_multiplier = atom_multiplier;
  config.newton = force->newton_pair;
  config.use_communication = communication_enabled;
  if (communication_enabled) {
    // The selected graph contains FFI gathers; bind them to LAMMPS's normal
    // forward/reverse pair communication for this pair instance.
    config.communication.context = this;
    config.communication.exchange = &ChemtrainDeploy::exchange_callback;
  }

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      setflag[i][j] = 1;
    }
  }

  model_properties = connector->load_model(config);

  // LAMMPS sizes pair communication buffers during initialization.
  // Reserve the exported maximum up front; changing these fields inside
  // the communication callback is too late and can overrun Comm buffers.
  comm_forward = model_properties.communication_buffer_width;
  comm_reverse = model_properties.communication_buffer_width;
  comm_reverse_off = model_properties.communication_buffer_width;

  std::string req_style = update->unit_style;
  std::string set_style = model_properties.unit_style;
  if (set_style != req_style) {
    error->all(
        FLERR,
        "The units of the model do not match the unit style {:s}. "
        "Please use the units from {:s} to {:s}.",
        req_style, set_style);
  }
}

/* ---------------------------------------------------------------------- */

int ChemtrainDeploy::exchange_callback(
    void *context, void *data, std::int64_t rows, std::int64_t cols,
    jcn::CommunicationScalarType type, bool reverse, const char **error_msg) {
  // Call order for one exported gather site:
  //
  //   XLA FFI -> connector worker stages the active device rows on the host
  //           -> CommunicationContext wakes the LAMMPS caller thread
  //           -> this callback enters exchange()
  //           -> LAMMPS Comm performs its ordered domain swaps
  //           -> pack_* / MPI / unpack_* runs once per swap
  //           -> the connector copies the changed row range back to XLA
  //
  // MPI must run on the thread that called LAMMPS, rather than PJRT's worker.
  // The CommunicationContext rendezvous provides that hand-off while the FFI
  // result remains asynchronous from XLA's point of view.
  auto *self = static_cast<ChemtrainDeploy *>(context);
  try {
    return self->exchange(data, rows, cols, type, reverse);
  } catch (const std::exception &e) {
    self->communication_error = e.what();
    if (error_msg != nullptr) *error_msg = self->communication_error.c_str();
    return 1;
  }
}

int ChemtrainDeploy::exchange(void *data, std::int64_t rows,
                              std::int64_t cols,
                              jcn::CommunicationScalarType type,
                              bool reverse) {
  const std::int64_t required =
      static_cast<std::int64_t>(atom->nlocal) + atom->nghost;

  if (communication_debug_enabled()) {
    std::cerr << "[COMM] " << (reverse ? "REV" : "FWD")
              << " rows=" << rows
              << " cols=" << cols
              << " required=" << required
              << " first_before=" << debug_first_value(data, type, rows, cols)
              << std::endl;
  }

  if (data == nullptr || cols <= 0 || rows < required) {
    throw std::runtime_error("Invalid packed feature buffer from PJRT");
  }

  if (cols > std::numeric_limits<int>::max()) {
    throw std::runtime_error("Packed feature width exceeds LAMMPS limits");
  }

  if (cols > model_properties.communication_buffer_width) {
    throw std::runtime_error(
        "Packed feature width exceeds exported communication buffer width");
  }

  communication_data = data;
  communication_rows = rows;
  communication_cols = cols;
  communication_type = type;

  if (reverse) {
    ProfileRange range("chemtrain_comm.lammps_reverse");
    // This is the transpose of the forward ghost overwrite: LAMMPS sends
    // ghost cotangents back to their owning ranks, where unpack_reverse adds
    // them to the local feature gradient.
    comm->reverse_comm(this);

    if (communication_debug_enabled()) {
      std::cerr << "[COMM] REV after_comm first="
                << debug_first_value(data, type, rows, cols)
                << std::endl;
    }

    // The connector copies back only owner rows and zeros ghost rows directly
    // on the device. Clearing the staged host ghosts here would traverse the
    // same large buffer without producing data that is subsequently consumed.
  } else {
    ProfileRange range("chemtrain_comm.lammps_forward");
    // LAMMPS owns the domain decomposition, so its standard pair exchange is
    // the source of truth for replacing each ghost feature with its owner's
    // current value at this message-passing boundary.
    comm->forward_comm(this);

    if (communication_debug_enabled()) {
      std::cerr << "[COMM] FWD after_comm first="
                << debug_first_value(data, type, rows, cols)
                << std::endl;
    }
  }

  communication_data = nullptr;
  communication_rows = 0;
  communication_cols = 0;

  return 0;
}

/* ---------------------------------------------------------------------- */

double ChemtrainDeploy::communication_get(std::int64_t row,
                                          std::int64_t col) const {
  const std::int64_t index = row * communication_cols + col;
  if (communication_type == jcn::CommunicationScalarType::F64) {
    return static_cast<double *>(communication_data)[index];
  }
  return static_cast<float *>(communication_data)[index];
}

void ChemtrainDeploy::communication_set(std::int64_t row, std::int64_t col,
                                        double value) {
  const std::int64_t index = row * communication_cols + col;
  if (communication_type == jcn::CommunicationScalarType::F64) {
    static_cast<double *>(communication_data)[index] = value;
  } else {
    static_cast<float *>(communication_data)[index] = static_cast<float>(value);
  }
}

void ChemtrainDeploy::communication_add(std::int64_t row, std::int64_t col,
                                        double value) {
  communication_set(row, col, communication_get(row, col) + value);
}

/* ---------------------------------------------------------------------- */

int ChemtrainDeploy::pack_forward_comm(int n, int *list, double *buf,
                                       int, int *) {
  ProfileRange range("chemtrain_comm.pack_forward");
  if (communication_debug_enabled()) {
    std::cerr << "[COMM] pack_forward n=" << n
              << " cols=" << communication_cols << std::endl;
  }

  const int count = checked_communication_count(n, communication_cols);
  // For a forward swap LAMMPS supplies the possibly non-contiguous owner/ghost
  // indices that the neighboring domain needs. Gather each complete feature
  // row into LAMMPS's contiguous double buffer. F64 rows use bulk copies;
  // model-native F32 rows are converted only at this final MPI boundary.
  const int width = static_cast<int>(communication_cols);
  const int nthreads = comm->nthreads;
  if (communication_type == jcn::CommunicationScalarType::F64) {
    const auto *data = static_cast<const double *>(communication_data);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nthreads) \
    if(nthreads > 1 && count >= 65536)
#endif
    for (int i = 0; i < n; ++i) {
      const double *row = data +
          static_cast<std::int64_t>(list[i]) * communication_cols;
      std::copy_n(row, width, buf + static_cast<std::int64_t>(i) * width);
    }
  } else {
    const auto *data = static_cast<const float *>(communication_data);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nthreads) \
    if(nthreads > 1 && count >= 65536)
#endif
    for (int i = 0; i < n; ++i) {
      const float *row = data +
          static_cast<std::int64_t>(list[i]) * communication_cols;
      double *packed = buf + static_cast<std::int64_t>(i) * width;
      for (int j = 0; j < width; ++j) {
        packed[j] = static_cast<double>(row[j]);
      }
    }
  }
  return count;
}

void ChemtrainDeploy::unpack_forward_comm(int n, int first, double *buf) {
  ProfileRange range("chemtrain_comm.unpack_forward");
  // Forward receives occupy the contiguous ghost range [first, first + n).
  // Overwrite those staged feature rows with the neighboring owners' values;
  // later swaps may use these newly received ghosts to construct corner halos.
  const bool debug = communication_debug_enabled();
  if (debug) {
    std::cerr << "[COMM] unpack_forward n=" << n
              << " first=" << first
              << " cols=" << communication_cols;
  }

  if (debug && n > 0 && communication_cols > 0) {
    std::cerr << " before=" << communication_get(first, 0)
              << " incoming=" << buf[0];
  }

  const int count = checked_communication_count(n, communication_cols);
  const std::int64_t offset =
      static_cast<std::int64_t>(first) * communication_cols;
  const int nthreads = comm->nthreads;
  if (communication_type == jcn::CommunicationScalarType::F64) {
    auto *destination = static_cast<double *>(communication_data) + offset;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nthreads) \
    if(nthreads > 1 && count >= 65536)
#endif
    for (int j = 0; j < count; ++j) destination[j] = buf[j];
  } else {
    float *destination = static_cast<float *>(communication_data) + offset;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nthreads) \
    if(nthreads > 1 && count >= 65536)
#endif
    for (int j = 0; j < count; ++j)
      destination[j] = static_cast<float>(buf[j]);
  }

  if (debug && n > 0 && communication_cols > 0) {
    std::cerr << " after=" << communication_get(first, 0);
  }

  if (debug) std::cerr << std::endl;
}

static double max_abs_double_buf(const double *buf, int n) {
  double m = 0.0;
  for (int i = 0; i < n; ++i) {
    m = std::max(m, std::abs(buf[i]));
  }
  return m;
}


int ChemtrainDeploy::pack_reverse_comm(int n, int first, double *buf) {
  ProfileRange range("chemtrain_comm.pack_reverse");
  const int count = checked_communication_count(n, communication_cols);
  // Reverse communication traverses the forward swaps in reverse order. Its
  // send rows are therefore the contiguous ghost block populated by the
  // corresponding forward receive; pack their cotangents for their owners.
  const std::int64_t offset =
      static_cast<std::int64_t>(first) * communication_cols;
  const int nthreads = comm->nthreads;
  if (communication_type == jcn::CommunicationScalarType::F64) {
    const auto *source = static_cast<const double *>(communication_data) + offset;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nthreads) \
    if(nthreads > 1 && count >= 65536)
#endif
    for (int j = 0; j < count; ++j) buf[j] = source[j];
  } else {
    const float *source = static_cast<const float *>(communication_data) + offset;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nthreads) \
    if(nthreads > 1 && count >= 65536)
#endif
    for (int j = 0; j < count; ++j)
      buf[j] = static_cast<double>(source[j]);
  }

  if (communication_debug_enabled()) {
    std::cerr << "[COMM] pack_reverse n=" << n
              << " first=" << first
              << " cols=" << communication_cols
              << " first_val=" << (count > 0 ? buf[0] : 0.0)
              << " maxabs=" << max_abs_double_buf(buf, count)
              << std::endl;
  }

  return count;
}


void ChemtrainDeploy::unpack_reverse_comm(int n, int *list, double *buf) {
  ProfileRange range("chemtrain_comm.unpack_reverse");
  const int count = checked_communication_count(n, communication_cols);
  // The transpose of a forward overwrite is an indexed accumulation. Scatter
  // each returned ghost cotangent into the owner row selected by LAMMPS. The
  // connector subsequently copies owners back and defines ghost cotangents as
  // zero, so this function deliberately does not clear staged ghost rows.
  int first_atom = (n > 0 ? list[0] : -1);
  const bool debug = communication_debug_enabled();
  double before = 0.0;
  if (debug && n > 0 && communication_cols > 0) {
    before = communication_get(first_atom, 0);
  }

  const int width = static_cast<int>(communication_cols);
  const int nthreads = comm->nthreads;
  if (communication_type == jcn::CommunicationScalarType::F64) {
    auto *data = static_cast<double *>(communication_data);
#ifdef _OPENMP
#pragma omp parallel num_threads(nthreads) \
    if(nthreads > 1 && count >= 65536)
#endif
    {
      int begin = 0;
      int end = width;
#ifdef _OPENMP
      const int team_size = omp_get_num_threads();
      const int tid = omp_get_thread_num();
      begin = width * tid / team_size;
      end = width * (tid + 1) / team_size;
#endif
      for (int i = 0; i < n; ++i) {
        double *row = data +
            static_cast<std::int64_t>(list[i]) * width;
        const double *packed = buf + static_cast<std::int64_t>(i) * width;
        for (int j = begin; j < end; ++j) row[j] += packed[j];
      }
    }
  } else {
    auto *data = static_cast<float *>(communication_data);
#ifdef _OPENMP
#pragma omp parallel num_threads(nthreads) \
    if(nthreads > 1 && count >= 65536)
#endif
    {
      int begin = 0;
      int end = width;
#ifdef _OPENMP
      const int team_size = omp_get_num_threads();
      const int tid = omp_get_thread_num();
      begin = width * tid / team_size;
      end = width * (tid + 1) / team_size;
#endif
      for (int i = 0; i < n; ++i) {
        float *row = data +
            static_cast<std::int64_t>(list[i]) * width;
        const double *packed = buf + static_cast<std::int64_t>(i) * width;
        for (int j = begin; j < end; ++j)
          row[j] += static_cast<float>(packed[j]);
      }
    }
  }

  double after = 0.0;
  if (debug && n > 0 && communication_cols > 0) {
    after = communication_get(first_atom, 0);
  }

  if (debug) {
    std::cerr << "[COMM] unpack_reverse n=" << n
              << " cols=" << communication_cols
              << " target=" << first_atom
              << " first_in=" << (count > 0 ? buf[0] : 0.0)
              << " maxabs_in=" << max_abs_double_buf(buf, count)
              << " before=" << before
              << " after=" << after
              << std::endl;
  }
}

/* ---------------------------------------------------------------------- */

void ChemtrainDeploy::init_style()
{
  compilations = 0;
  initial_compilations = 0;
  atom_recompilations = 0;
  edge_recompilations = 0;
  flops = 0.0;

  // The exported model is authoritative for the halo depth. Include the
  // LAMMPS neighbor skin so users do not need a matching comm_modify command.
  comm->cutghostuser = model_properties.comm_dist + neighbor->skin;

  int request = NeighConst::REQ_DEFAULT;

  if (model_properties.neighbor_list.include_ghosts) {
    request |= NeighConst::REQ_GHOST;
  }

  if (!model_properties.neighbor_list.half_list || force->newton) {
    // It seems like setting newton to true requires a full list.
    request |= NeighConst::REQ_FULL;
  }

  neighbor->add_request(this, request);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double ChemtrainDeploy::init_one(int i, int j)
{
  if (!allocated) allocate();

  if (setflag[i][j] == 0) error->all(FLERR, "Not all pair coeffs are set");

  double min_comm_dist = model_properties.comm_dist + neighbor->skin;
  if (min_comm_dist > comm->get_comm_cutoff()) {
    error->all(
      FLERR, "Communication cutoff is too small for the model. Increase "
      "the communication cutoff to at least {:.4f}.", min_comm_dist
    );
  }

  return model_properties.cutoff;
}

/* ---------------------------------------------------------------------- */

void ChemtrainDeploy::finish()
{
  int num_procs;
  double min_flops, max_flops, sum_flops;
  MPI_Allreduce(&flops, &min_flops, 1, MPI_DOUBLE, MPI_MIN, world);
  MPI_Allreduce(&flops, &max_flops, 1, MPI_DOUBLE, MPI_MAX, world);
  MPI_Allreduce(&flops, &sum_flops, 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Comm_size(world, &num_procs);

  double avg_flops = sum_flops / static_cast<double>(num_procs);

  auto reduce_counts = [this, num_procs](int local) {
    int minimum, maximum, total;
    MPI_Allreduce(&local, &minimum, 1, MPI_INT, MPI_MIN, world);
    MPI_Allreduce(&local, &maximum, 1, MPI_INT, MPI_MAX, world);
    MPI_Allreduce(&local, &total, 1, MPI_INT, MPI_SUM, world);
    return CountSummary{
        minimum,
        static_cast<double>(total) / num_procs,
        maximum,
        total,
    };
  };

  const auto compile_stats = reduce_counts(compilations);
  const auto initial_stats = reduce_counts(initial_compilations);
  const auto atom_stats = reduce_counts(atom_recompilations);
  const auto edge_stats = reduce_counts(edge_recompilations);

  utils::logmesg(
      lmp, "\n==== JaxConnect Summary =========.\n"
           "- Compilations: {:d} min / {:.2f} avg / {:d} max / {:d} total\n"
           "- Initial compilations: {:d} min / {:.2f} avg / {:d} max / {:d} total\n"
           "- Atom recompilations: {:d} min / {:.2f} avg / {:d} max / {:d} total\n"
           "- Edge recompilations: {:d} min / {:.2f} avg / {:d} max / {:d} total\n"
           "- Estimated FLOP: {:.2e} min / {:.2e} avg / {:.2e} max / {:.2e} total\n\n",
      compile_stats.minimum, compile_stats.average, compile_stats.maximum,
      compile_stats.total,
      initial_stats.minimum, initial_stats.average, initial_stats.maximum,
      initial_stats.total,
      atom_stats.minimum, atom_stats.average, atom_stats.maximum,
      atom_stats.total,
      edge_stats.minimum, edge_stats.average, edge_stats.maximum,
      edge_stats.total,
      min_flops, avg_flops, max_flops, sum_flops);

  // This stable record is consumed by regression tooling; the readable table
  // above may evolve without turning wording changes into test failures.
  utils::logmesg(
      lmp,
      "JCN_STATS compilations_total={:d} initial_total={:d} "
      "atom_total={:d} edge_total={:d}\n",
      compile_stats.total, initial_stats.total, atom_stats.total,
      edge_stats.total);
}

/* ---------------------------------------------------------------------- */

void *ChemtrainDeploy::extract(const char *str, int &dim)
{
  dim = 0;
  if (strcmp(str, "scale") == 0) return (void *) &scale;
  return nullptr;
}
