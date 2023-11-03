"""
BPE Tokenizer starter
"""

from pathlib import Path
from main import collect_frequencies, get_vocabulary, train


def main() -> None:
    """
    Launches an implementation
    """
    assets_path = Path(__file__).parent / 'assets'
    with open(assets_path / 'text.txt', 'r', encoding='utf-8') as text_file:
        text = text_file.read()

    result = None
    assert result, 'Encoding is not working'
    text_frequencies = collect_frequencies(text, None, '</s>')
    train_text = train(text_frequencies, 100)
    if train_text:
        with open(assets_path / 'secrets' / 'secret_1.txt', 'r', encoding='utf-8') as text_file:
            secret_text = text_file.read()
        vocabulary = get_vocabulary(train_text, '<unk>')
        decoding_list = []
        for number in secret_text.split(' '):
            decoding_list.append(int(number))



if __name__ == "__main__":
    main()
