#!/bin/bash
source config/common.sh

set -x

echo -e '\n'
echo 'Running stubgen check...'

configure_script

FAILED=0
LABS=$(get_labs)

for LAB_NAME in $LABS; do
  echo "Checking stubgen for lab: ${LAB_NAME}"
  python ./config/generate_stubs/run_generator.py \
        --source_code_path ${LAB_NAME}/main.py \
        --target_code_path ./build/stubs/${LAB_NAME}/main.py

  check_if_failed

  python ./config/generate_stubs/run_generator.py \
        --source_code_path ${LAB_NAME}/start.py \
        --target_code_path ./build/stubs/${LAB_NAME}/start.py

  check_if_failed

done

if [[ ${FAILED} -eq 1 ]]; then
	echo "Stubgen check failed."
	exit ${FAILED}
fi

echo "Stubgen check passed."
