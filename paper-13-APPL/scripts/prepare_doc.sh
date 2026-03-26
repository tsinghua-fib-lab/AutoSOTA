#!/usr/bin/env bash

rm -r docs/reference && python scripts/gen_ref_nav.py && echo "Generated reference navigation"
