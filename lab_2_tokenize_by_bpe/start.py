"""
BPE Tokenizer starter
"""
from pathlib import Path

from lab_2_tokenize_by_bpe.main import (collect_frequencies, decode, get_vocabulary, train)


def main() -> None:
    """
    Launches an implementation
    """
    assets_path = Path(__file__).parent / 'assets'
    with open(assets_path / 'text.txt', 'r', encoding='utf-8') as text_file:
        text = text_file.read()
    text_frequencies = (collect_frequencies(text, None, '</s>'))
    text_frequencies = (train(text_frequencies, 100))
    if text_frequencies:
        with open(assets_path / 'secrets/secret_1.txt', 'r', encoding='utf-8') as file:
            secret = file.read()
        vocabulary = get_vocabulary(text_frequencies, '<unk>')
        secret_list = [int(num) for num in secret.split(' ')]
    result = decode(secret_list, vocabulary, '</s>')
    print(result)
    assert result, "Encoding is not working"


if __name__ == "__main__":
    main()
