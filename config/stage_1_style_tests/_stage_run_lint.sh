#!/bin/bash
source config/common.sh

set -x

echo -e '\n'
echo 'Running lint check...'

configure_script

FAILED=0

lint_output=$(python -m pylint --exit-zero --rcfile config/stage_1_style_tests/.pylintrc config)

python config/stage_1_style_tests/lint_level.py \
  --lint-output "${lint_output}" \
  --target-score "10"

check_if_failed

LABS=$(get_labs)

for LAB_NAME in $LABS; do
  echo "Running lint for lab ${LAB_NAME}"
  TARGET_SCORE=$(get_score ${LAB_NAME})

  check_skip "$1" "$2" "$LAB_NAME"

  if [[ ${LAB_NAME} == 'lab_6_pipeline' ]]; then
    export PYTHONPATH=${PYTHONPATH}:lab_6_pipeline/universal_dependencies
  fi

  IGNORE_OPTION=""
  if [ "$REPOSITORY_TYPE" == "public" ]; then
    IGNORE_OPTION="--ignore ${LAB_NAME}/tests"
  fi

  lint_output=$(python -m pylint --exit-zero --rcfile config/stage_1_style_tests/.pylintrc ${LAB_NAME} ${IGNORE_OPTION})

  python config/stage_1_style_tests/lint_level.py \
    --lint-output "${lint_output}" \
    --target-score "${TARGET_SCORE}"

  check_if_failed
done

if [[ ${FAILED} -eq 1 ]]; then
  echo "Lint check failed."
  exit ${FAILED}
fi

echo "Lint check passed."
