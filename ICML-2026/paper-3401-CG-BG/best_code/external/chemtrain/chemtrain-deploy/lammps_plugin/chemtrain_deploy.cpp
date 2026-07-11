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

#include "lammpsplugin.h"

#include "version.h"

#include <cstring>

#include "pair_chemtrain_deploy.h"

using namespace LAMMPS_NS;

static Pair *ChemtrainDeployCreator(LAMMPS *lmp)
{
  return new ChemtrainDeploy(lmp);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_t plugin;
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;

  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "chemtrain_deploy";
  plugin.info = "JAX-Based GNN Potential Models";
  plugin.author = "Paul Fuchs (paul.fuchs@tum.de)";
  plugin.creator.v1 = (lammpsplugin_factory1 *) &ChemtrainDeployCreator;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
}
