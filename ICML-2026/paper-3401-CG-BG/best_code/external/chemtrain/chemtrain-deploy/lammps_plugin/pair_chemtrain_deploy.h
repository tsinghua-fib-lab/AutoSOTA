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

#ifndef LMP_PAIR_MORSE2_H
#define LMP_PAIR_MORSE2_H

#include "pair.h"
#include "libconnector.h"


namespace LAMMPS_NS {

class ChemtrainDeploy : public Pair {
 public:
  ChemtrainDeploy(class LAMMPS *);
  ~ChemtrainDeploy() override;
  void compute(int, int) override;

  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void finish() override;
  void *extract(const char *, int &) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

 protected:
//  bool allocated;

  // double cut_global;
  double **cut;
  double scale = 1.0;

  // Statistics
  int compilations = 0;
  int initial_compilations = 0;
  int atom_recompilations = 0;
  int edge_recompilations = 0;
  double flops = 0.0;

  jcn::ModelProperties model_properties;

  std::unique_ptr<jcn::Connector> connector;

  void *communication_data = nullptr;
  std::int64_t communication_rows = 0;
  std::int64_t communication_cols = 0;
  jcn::CommunicationScalarType communication_type =
      jcn::CommunicationScalarType::F32;
  std::string communication_error;
  bool communication_enabled = false;

  static int exchange_callback(void *, void *, std::int64_t, std::int64_t,
                               jcn::CommunicationScalarType, bool,
                               const char **);
  int exchange(void *, std::int64_t, std::int64_t,
               jcn::CommunicationScalarType, bool);
  double communication_get(std::int64_t, std::int64_t) const;
  void communication_set(std::int64_t, std::int64_t, double);
  void communication_add(std::int64_t, std::int64_t, double);

  virtual void allocate();
};

}    // namespace LAMMPS_NS

#endif
