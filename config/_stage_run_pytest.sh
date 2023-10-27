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
fi

if [[ "$LAB_PATH" ]]; then
  LABEL="mark${TARGET_SCORE} and ${PYTEST_LABEL}"
else
  LABEL="${PYTEST_LABEL}"
fi

python -m pytest -m "${LABEL}"
ret=$?

if [ "$ret" = 5 ]; then
  echo "No tests collected.  Exiting with 0 (instead of 5)."
  exit 0
fi

echo "Pytest results (should be 0): $ret"

exit "$ret"
