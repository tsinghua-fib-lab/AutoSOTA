#!/usr/bin/env bash

/usr/bin/env

jupytext --output ../docs/examples/execute_water.ipynb CG_water_difftre.md

cd ../docs/examples && papermill execute_water.ipynb CG_water_difftre.ipynb --progress-bar --request-save-on-cell-execute

cp CG_water_difftre.ipynb ../../examples/CG_water_difftre.ipynb
