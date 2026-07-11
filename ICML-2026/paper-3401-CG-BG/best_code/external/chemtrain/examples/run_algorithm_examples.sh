#! /bin/bash

cd ../docs/algorithms || exit

for file in *.md
do
    output=$(sed 's/\.md/\.ipynb/g' <<< $file)
    jupytext --output ../../examples/$output $file --execute
done
