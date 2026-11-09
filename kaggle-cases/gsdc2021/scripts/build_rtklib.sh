#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$task_root"
release_path="$task_root/upstream/demo5_b34e_phone/demo5_b34e_phone"
repo_path="$task_root/upstream/RTKLIB_smartphone"
cp "$release_path/src/rtkpos.c" "$repo_path/src/rtkpos.c"
cp "$release_path/src/rtklib.h" "$repo_path/src/rtklib.h"
# The released smartphone C source requires C99. IERS is not enabled in the
# original build options, so the unused Fortran archive is unnecessary.
make -B -j4 -C "$repo_path/app/consapp/rnx2rtkp/gcc" \
  'CFLAGS=-Wall -O3 -std=gnu99 -Wno-unused-but-set-variable -I../../../../src -DTRACE -DENAGLO -DENAQZS -DENAGAL -DENACMP -DENAIRN -DNFREQ=3' \
  'LDLIBS=-lm -lrt'
