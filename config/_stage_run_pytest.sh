set -x
source config/common.sh

configure_script

PR_NAME="$1"
PR_AUTHOR="$2"
shift 2

while getopts "l:m:" opt; do
    case $opt in
        l) LAB_PATH=$OPTARG;;
        m) PYTEST_LABEL=$OPTARG;;
    esac
done

if [[ "$LAB_PATH" ]]; then
  check_skip "$PR_NAME" "$PR_AUTHOR" "$LAB_PATH"

  TARGET_SCORE=$(get_score "$LAB_PATH")

  if [[ "$LAB_PATH" == "lab_5_scrapper" ]]; then
    python config/config_param_changer.py --config_path="lab_5_scrapper/scrapper_config.json"
    echo "Changed config params"
  fi
fi

if [[ "$LAB_PATH" ]]; then
  LABEL="mark${TARGET_SCORE} and ${PYTEST_LABEL}"
else
  LABEL="${PYTEST_LABEL}"
fi

if [[ "$LAB_PATH" == "lab_5_scrapper" ]]; then
  python -m pytest -m "${LABEL}" --capture=no --ignore=lab_6_pipeline
elif [[ "$LAB_PATH" == "lab_6_pipeline" && ${TARGET_SCORE} == 10 ]]; then
  python -m pytest -m "${LABEL}" --capture=no
else
  python -m pytest -m "${LABEL}" --capture=no --ignore lab_6_pipeline/tests/s4_pos_frequency_pipeline_test.py --ignore lab_6_pipeline/tests/s3_6_advanced_pipeline_test.py
fi

ret=$?
if [ "$ret" = 5 ]; then
  echo "No tests collected.  Exiting with 0 (instead of 5)."
  exit 0
fi

echo "Pytest results (should be 0): $ret"

exit "$ret"
