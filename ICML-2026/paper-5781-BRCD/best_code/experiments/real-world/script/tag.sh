#!/bin/bash

message=$1
# assert message is not empty
if [ -z "$message" ]; then
  echo "Usage: . script/tag.sh <message>"
fi

git tag -a $(python setup.py --version) -m $message