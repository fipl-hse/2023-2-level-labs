"""
BPE Tokenizer starter
"""
from pathlib import Path
from lab_2_tokenize_by_bpe.main import (collect_frequencies,
                                        train, get_vocabulary,
                                        decode)


def main() -> None:
    """
    Launches an implementation
    """
    assets_path = Path(__file__).parent / 'assets'
    with open(assets_path / 'text.txt', 'r', encoding='utf-8') as text_file:
        text = text_file.read()

    with open(assets_path / 'secrets/secret_3.txt', 'r', encoding='utf-8') as secret_file:
        encoded_text = secret_file.read()

    word_frequencies = collect_frequencies(text, None, '</s>')
    word_frequencies = train(word_frequencies, 100)
    if word_frequencies:
        vocabulary = get_vocabulary(word_frequencies, '<unk>')
        encoded_text = [int(num) for num in encoded_text.split()]
        result = decode(encoded_text, vocabulary, '</s>')
        print(result)
        assert result, "Encoding is not working"


if __name__ == "__main__":
    main()