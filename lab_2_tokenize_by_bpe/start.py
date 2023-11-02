"""
BPE Tokenizer starter
"""
from pathlib import Path

import lab_2_tokenize_by_bpe.main as main_bpe


def main() -> None:
    """
    Launches an implementation
    """
    assets_path = Path(__file__).parent / 'assets'
    with open(assets_path / 'text.txt', 'r', encoding='utf-8') as text_file:
        text = text_file.read()
    text_frequencies = (main_bpe.collect_frequencies(text, None, '</s>'))
    text_frequencies = (main_bpe.train(text_frequencies, 100))
    if text_frequencies:
        with open(assets_path / 'secrets/secret_1.txt', 'r', encoding='utf-8') as file:
            secret = file.read()
        vocabulary = main_bpe.get_vocabulary(text_frequencies, '<unk>')
        secret_list = [int(num) for num in secret.split(' ')]
    result = main_bpe.decode(secret_list, vocabulary, '</s>')
    print(result)
    assert result, "Encoding is not working"


if __name__ == "__main__":
    main()
