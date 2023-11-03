set -x

echo $1
if [ $1 == "smoke" ]; then
  DIRS_TO_CHECK=("config" "seminars")
else
  DIRS_TO_CHECK=("lab_1_classify_by_unigrams" "lab_2_tokenize_by_bpe" "lab_3_generate_by_ngrams" "config" "seminars")
fi

python -m pylint --exit-zero --rcfile config/stage_1_style_tests/.pylintrc "${DIRS_TO_CHECK[@]}"

mypy "${DIRS_TO_CHECK[@]}"

python -m flake8 "${DIRS_TO_CHECK[@]}"
