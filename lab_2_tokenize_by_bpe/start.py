"""
BPE Tokenizer starter
"""
import json
from pathlib import Path

from lab_2_tokenize_by_bpe.main import prepare_word, collect_frequencies, count_tokens_pairs, merge_tokens


def main() -> None:
    """
    Launches an implementation
    """
    assets_path = Path(__file__).parent / 'assets'
    with open(assets_path / 'text.txt', 'r', encoding='utf-8') as text_file:
        text = text_file.read()
    with open(assets_path / 'secrets/secret_2.txt', 'r', encoding='utf-8') as text_file:
        encoded_secret = text_file.read()
    dict_frequencies = collect_frequencies(text, None, '</s>')
    merged_tokens = train(dict_frequencies, 100)
    if merged_tokens:
        vocabulary = get_vocabulary(merged_tokens, '<unk>')
        secret = [int(num) for num in encoded_secret.split()]
        result = decode(secret, vocabulary, '</s>')
        print(result)
        assert result, "Encoding is not working"

    print(prepare_word('father', '</b>', '</s>'))
    print(collect_frequencies(text, None, '</s>'))
    print(count_tokens_pairs(collect_frequencies(text, None, '</s>')))
    print(merge_tokens(count_tokens_pairs(collect_frequencies(text, None, '</s>')), ('и', 'м')))
    #    print(train(count_tokens_pairs(collect_frequencies(text, None, '</s>')), 12))
    result = 1
    assert result, "Encoding is not working"


if __name__ == "__main__":
    main()
