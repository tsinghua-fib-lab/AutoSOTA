#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
REQ_FILE="${REQUIREMENTS_FILE:-requirements.txt}"

# Avoid accidental leakage from a user's personal site-packages; keep the
# environment definition explicit and reproducible.
export PYTHONNOUSERSITE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1

# If running on NERSC (or any site using Environment Modules), allow loading
# site modules before creating/activating the venv. Set NERSC_MODULES to
# override the default list. Detection prefers an existing `module` command
# or the presence of the NERSC_HOST env var or hostname containing 'nersc'.
if ! command -v module >/dev/null 2>&1; then
  if [ -f /usr/share/Modules/init/bash ]; then
    # try to initialize environment-modules on some systems
    # shellcheck disable=SC1091
    source /usr/share/Modules/init/bash || true
  elif [ -f /etc/profile.d/modules.sh ]; then
    # fallback common location
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh || true
  fi
fi

if command -v module >/dev/null 2>&1; then
  if [ -n "${NERSC_HOST:-}" ] || hostname | grep -qi nersc; then
    NERSC_MODULES="${NERSC_MODULES:-python pytorch/2.8.0}"
    echo "Detected NERSC-like environment — loading modules: $NERSC_MODULES"
    for m in $NERSC_MODULES; do
      module load "$m" || true
    done
    # update PYTHON_BIN if the module provides python3
    if command -v python3 >/dev/null 2>&1; then
      PYTHON_BIN="$(command -v python3)"
    fi
    NERSC_DETECTED=1

    # Default to NERSC-specific dependency list when available.
    if [ -z "${REQUIREMENTS_FILE:-}" ] && [ -f "requirements.nersc.txt" ]; then
      REQ_FILE="requirements.nersc.txt"
    fi
  fi
fi
if [ "${NERSC_DETECTED:-0}" = "1" ]; then
  echo "Running setup with loaded NERSC modules"

  # Ensure we use the module-provided python executable.
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
  if [ -z "$PYTHON_BIN" ]; then
    echo "Could not find python after loading modules." >&2
    exit 1
  fi

  # If an existing venv was created with a different Python minor version,
  # recreate it so wheel resolution matches module Python compatibility.
  if [ -x ".venv/bin/python" ]; then
    VENV_VER="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
    MOD_VER="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
    if [ "$VENV_VER" != "$MOD_VER" ]; then
      echo "Existing .venv uses Python $VENV_VER but modules provide Python $MOD_VER; recreating .venv"
      rm -rf .venv
    fi
  fi

  if [ ! -d ".venv" ]; then
    echo "Creating virtualenv with --system-site-packages to reuse module-installed packages"
    "$PYTHON_BIN" -m venv --system-site-packages .venv
  fi
fi

# Non-NERSC path: create venv without system site packages
if [ "${NERSC_DETECTED:-0}" != "1" ] && [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip --disable-pip-version-check

if [ ! -f "$REQ_FILE" ]; then
  echo "Requirements file not found: $REQ_FILE" >&2
  exit 1
fi

echo "Installing dependencies from: $REQ_FILE"
python -m pip install --no-input --disable-pip-version-check -r "$REQ_FILE"

snapshot_dir=".reproducibility"
mkdir -p "$snapshot_dir"

{
  echo "# Setup snapshot"
  echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname 2>/dev/null || true)"
  echo "cwd=$ROOT_DIR"
  echo "selected_requirements=$REQ_FILE"
  echo "python_bin=$PYTHON_BIN"
  echo "venv_python=$(command -v python)"
  echo "python_version=$(python --version 2>&1)"
  echo "pip_version=$(python -m pip --version 2>&1)"
  echo "python_executable=$(python -c 'import sys; print(sys.executable)')"
  echo
  echo "# module list"
  if command -v module >/dev/null 2>&1; then
    module list 2>&1 || true
  else
    echo "module_command=unavailable"
  fi
  echo
  echo "# pip freeze --local"
  python -m pip freeze --local | sort
} > "$snapshot_dir/setup-snapshot.txt"

python -m pip freeze --local | sort > "$snapshot_dir/requirements.local.txt"

cp "$REQ_FILE" "$snapshot_dir/selected-requirements.txt"

echo "Setup complete. Activate with: source .venv/bin/activate"
