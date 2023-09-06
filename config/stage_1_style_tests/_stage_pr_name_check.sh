set -ex
source config/common.sh

echo -e '\n'

echo "Pull Request Name is $1"

configure_script

python config/stage_1_style_tests/pr_name_check.py --pr-name="$1" --pr-author="$2"
