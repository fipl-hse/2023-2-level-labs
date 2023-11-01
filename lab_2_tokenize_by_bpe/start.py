"""
BPE Tokenizer starter
"""
from pathlib import Path

import lab_2_tokenize_by_bpe.main as main_py


def main() -> None:
    """
    Launches an implementation
    """
    assets_path = Path(__file__).parent / 'assets'
    with open(assets_path / 'text.txt', 'r', encoding='utf-8') as text_file:
        text = text_file.read()

    word_frequencies = main_py.collect_frequencies(text, None, '</s>')
    if isinstance(word_frequencies, dict):
        vocabulary = main_py.get_vocabulary(word_frequencies, '<unk>')

    with open(assets_path / 'secrets' / 'secret_1.txt', 'r', encoding='utf-8') as secret_file:
        secret = secret_file.read()

    result = main_py.decode(secret.split(' '), vocabulary, '</s>')
    assert result, "Encoding is not working"


if __name__ == "__main__":
    main()
