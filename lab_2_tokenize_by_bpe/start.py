"""
BPE Tokenizer starter
"""
from pathlib import Path

from lab_2_tokenize_by_bpe.main import collect_frequencies, train, decode, get_vocabulary


def main() -> None:
    """
    Launches an implementation
    """
    assets_path = Path(__file__).parent / 'assets'
    with open(assets_path / 'text.txt', 'r', encoding='utf-8') as text_file:
        text = text_file.read()
    with open(assets_path / 'secrets'/'secret_3.txt', 'r', encoding='utf-8') as text_file:
        encoded_text = text_file.read()
    dict_train = train(collect_frequencies(text, start_of_word=None, end_of_word='</s>'), 100)
    if not dict_train:
        return
    vocabulary = get_vocabulary(dict_train, '<unk>')
    encoded_list = [int(coded) for coded in encoded_text.split()]
    result = decode(encoded_list, vocabulary, '</s>')
    print(result)


if __name__ == "__main__":
    main()
