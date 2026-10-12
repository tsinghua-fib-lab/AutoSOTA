#!/bin/sh
set -eu
PG_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
exec sh "$PG_ROOT/proteingym/baselines/S2F_ESM1v_PSSM/scripts/score_all.sh"
