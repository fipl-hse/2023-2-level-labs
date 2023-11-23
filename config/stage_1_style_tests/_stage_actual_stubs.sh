source config/common.sh

echo "Check stubs relevance"

configure_script

python config/stage_1_style_tests/check_actual_stubs.py
check_if_failed
