#!/usr/bin/env bash

set -euo pipefail

if command -v module >/dev/null 2>&1; then
  module load python
  module load pytorch/2.8.0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

if [ -d "$ROOT_DIR/.venv" ]; then
  source "$ROOT_DIR/.venv/bin/activate"
else
  echo "Warning: Virtual environment not found at $ROOT_DIR/.venv"
  echo "Run 'bash setup.sh' to create it"
fi

if ! python -c "import pyvista" >/dev/null 2>&1; then
  echo "Installing missing plotting dependency: pyvista"
  python -m pip install pyvista
fi

echo "Running plane and sphere plotting scripts"
python plotting/plotting_smileyface_plane.py
python plotting/plotting_smileyface_sphere.py

echo "Running varied-sigma diffusion plotting scripts"
python plotting/plotting_smileyface_plane_varied_sigmas.py
python plotting/plotting_smileyface_sphere_varied_sigmas.py
# python plotting/plotting_bunny_varied_sigmas.py

echo "Running plane and sphere NF plotting scripts"
python plotting/plotting_nf_smileface_plane.py
python plotting/plotting_nf_smileyface_sphere.py

echo "Running varied-sigma NF plotting scripts"
python plotting/plotting_nf_smileyface_plane_varied_sigmas.py
python plotting/plotting_nf_smileyface_sphere_varied_sigmas.py

# echo "Running bunny plotting script"
# python plotting/plotting_bunny.py

echo "Building combined plane/sphere metrics table"
python plotting/build_plane_sphere_metrics_table.py

echo "Writing main.tex table snippets for 3D tasks"
python plotting/build_main_tex_tables.py --only static,plane_sphere,mesh --strict
