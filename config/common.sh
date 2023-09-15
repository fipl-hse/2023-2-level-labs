get_score() {
  export TARGET_SCORE=$(head -2 $1/target_score.txt | tail -1)
  echo ${TARGET_SCORE}
}

get_labs() {
  jq -r '.labs[].name' config/project_config.json

}

configure_script() {
  if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    source venv/bin/activate
    export PYTHONPATH=$(pwd):$PYTHONPATH
    which python
    python -m pip list
  elif [[ "$OSTYPE" == "msys" ]]; then
    source venv/Scripts/activate
    export PYTHONPATH=$(pwd)
  fi
}

check_skip () {
  python config/skip_check.py --pr_name "$1" --pr_author "$2" --lab_path "$3"
  if [ $? -eq 0 ]; then
    echo 'skip check due to special conditions...' && exit 0
  fi
}

check_if_failed() {
  if [[ $? -ne 0 ]]; then
    echo "Check failed."
    exit 1
  else
    echo "Check passed."
  fi
}

