#!/bin/bash
source config/common.sh

set -x

echo -e '\n'
echo 'Running docstring style check...'

configure_script

FAILED=0
LABS=$(get_labs)

for LAB_NAME in $LABS; do
 echo -e '\n'
 echo "Running docstring style check for lab ${LAB_NAME}"

 filename=${LAB_NAME}/main.py
 echo "Running darglint for ${filename}"
 darglint --docstring-style google --strictness full --enable DAR104 ${filename}
 check_if_failed

 echo "Running pydocstyle for ${filename}"
 python -m pydocstyle --config ./config/stage_1_style_tests/.pydocstyle ${filename}
 check_if_failed
done

if [[ ${FAILED} -eq 1 ]]; then
  echo "Docstring style check failed."
  exit ${FAILED}
fi

echo "Docstring style check passed."
