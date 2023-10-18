"""
BPE Tokenizer starter
"""
from pathlib import Path

from lab_2_tokenize_by_bpe.main import prepare_word, collect_frequencies, count_tokens_pairs


def main() -> None:
    """
    Launches an implementation
    """
    assets_path = Path(__file__).parent / 'assets'
    with open(assets_path / 'text.txt', 'r', encoding='utf-8') as text_file:
        text = text_file.read()

    print(prepare_word("it's", '</b>', '</s>'))
    print(collect_frequencies(text, None, '</s>'))
    print(count_tokens_pairs(collect_frequencies("It's far, farther, farthest and old, older, oldest", None, '</s>')))

    result = count_tokens_pairs
    assert result, "Encoding is not working"


if __name__ == "__main__":
    main()
