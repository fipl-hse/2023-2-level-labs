set -ex
source config/common.sh

echo -e '\n'

echo "Check docstring"

configure_script

python config/stage_1_style_tests/check_docstrings.py
check_if_failed
