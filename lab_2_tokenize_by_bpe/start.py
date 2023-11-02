"""
BPE Tokenizer starter
"""

from pathlib import Path
from lab_2_tokenize_by_bpe.main import collect_frequencies, decode, get_vocabulary, train


def main() -> None:
    """
    Launches an implementation
    """
    assets_path = Path(__file__).parent / 'assets'
    with open(assets_path / 'text.txt', 'r', encoding='utf-8') as text_file:
        text = text_file.read()
    with open(assets_path / 'secrets' / 'secret_3.txt', 'r', encoding='utf-8') as text_file:
        encoded_text = text_file.read().split()
        encoded_list = [int(el) for el in encoded_text]
        word_frequencies = train(collect_frequencies(text, None, '</s>'), 100)
        if word_frequencies:
            vocabulary = get_vocabulary(word_frequencies, '<unk>')
            result = decode(encoded_list, vocabulary, '</s>')
            print(result)
            assert result, "Encoding is not working"


if __name__ == "__main__":
    main()
