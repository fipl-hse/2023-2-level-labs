#!/bin/bash
source config/common.sh

set -ex

echo -e '\n'
echo 'Running flake8 check...'

configure_script

python -m flake8 --config ./config/stage_1_style_tests/.flake8 config seminars

FAILED=0
LABS=$(get_labs)

for LAB_NAME in $LABS; do
  echo "Running flake8 for lab ${LAB_NAME}"
  TARGET_SCORE=$(get_score ${LAB_NAME})

  if [[ ${TARGET_SCORE} -gt 5 ]]; then
    echo "Running flake8 checks for marks 6, 8 and 10"
    python -m flake8 --config ./config/stage_1_style_tests/.flake8 ${LAB_NAME}
  fi

  echo "Checking flake8 for lab ${LAB_NAME}"
  check_if_failed
done

if [[ ${FAILED} -eq 1 ]]; then
  echo "Flake8 check failed."
  exit ${FAILED}
fi

echo "Flake8 check passed."
