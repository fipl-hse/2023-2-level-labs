#!/bin/bash
source config/common.sh

set -x

echo -e '\n'
echo 'Running pymarkdownlnt check...'

configure_script

FAILED=0
LABS=$(get_labs)

for LAB_NAME in $LABS; do
 echo "Running pymarkdownlnt for lab ${LAB_NAME}"

 for filename in ${LAB_NAME}/*.md; do
   echo "Running pymarkdownlnt for ${filename}"
   python -m pymarkdown --config config/stage_1_style_tests/pymarkdownlnt.json scan ${filename}

    check_if_failed
  done
done

echo "Running pymarkdownlnt for README.md"
#python -m pymarkdown --config config/stage_1_style_tests/pymarkdownlnt.json scan README.md docs/**/*.md

check_if_failed

echo "Pymarkdownlnt check passed."

