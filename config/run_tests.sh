#!/bin/bash

set -x

source config/common.sh

configure_script

FAILED=0
LABS=$(get_labs)

echo "Current scope: $LABS"

for LAB_NAME in $LABS; do
  echo "Running tests for lab ${LAB_NAME}"

	bash config/_stage_run_pytest.sh "$PR_NAME" "$PR_AUTHOR" -l "$LAB_NAME" -m "$LAB_NAME"

  if [[ $? -ne 0 ]]; then
      FAILED=1
  fi

done

if [[ ${FAILED} -eq 1 ]]; then
	echo "Tests failed."
	exit 1
fi

echo "Tests passed."
