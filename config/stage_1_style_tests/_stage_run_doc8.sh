#!/bin/bash
source config/common.sh

set -x

echo -e '\n'
echo 'Running doc8 check...'

configure_script

#python -m doc8 --config config/stage_1_style_tests/.doc8 README.rst docs/**/*.rst

FAILED=0
LABS=$(get_labs)

for LAB_NAME in $LABS; do
  echo -e '\n'
  echo "Running doc8 for lab ${LAB_NAME}"

  filename=${LAB_NAME}/*.rst
  echo "Running doc8 for ${filename}"
  python -m doc8 --config config/stage_1_style_tests/.doc8 ${filename}
  check_if_failed
done

check_if_failed

echo "Doc8 check passed."
