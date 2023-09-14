#!/bin/bash
source config/common.sh

set -ex

source venv/bin/activate

echo "Running start.py checks..."

source venv/bin/activate

export PYTHONPATH="$(pwd):${PYTHONPATH}"

LABS=$(get_labs)
WAS_FAILED=0

for LAB_NAME in $LABS; do
	echo "Running start.py checks for lab ${LAB_NAME}"
  IS_ADMIN=$(python config/is_admin.py --pr_name "$1")
	if [ "$REPOSITORY_TYPE" == "public" ] && [ "$IS_ADMIN" == 'YES' ] ; then
	  echo '[skip-lab] option was enabled, skipping check...'
	  continue
	fi
  cd ${LAB_NAME}
	if ! python start.py;  then
    	WAS_FAILED=1
	fi

	if [[ $WAS_FAILED -eq 1 ]]; then
    echo "start.py fails while running"
    echo "Check for start.py file for lab ${LAB_NAME} failed."
    exit 1
  fi

  echo "Check calling lab ${LAB_NAME} passed"

  cd ..
  START_PY_FILE=$(cat ${LAB_NAME}/start.py)
  python config/check_start_content.py --start_py_content "$START_PY_FILE"
done
