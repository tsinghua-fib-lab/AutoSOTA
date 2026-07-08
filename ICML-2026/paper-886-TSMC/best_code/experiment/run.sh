#!/usr/bin/env bash
# Helper script to run sub-scripts from the `pwd` without cd hopping.

content=$(cat "$1")

shift

eval "$content"
