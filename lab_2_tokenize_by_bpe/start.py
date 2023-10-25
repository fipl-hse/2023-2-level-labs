"""
BPE Tokenizer starter
"""
from pathlib import Path

from lab_2_tokenize_by_bpe.main import collect_frequencies, train, get_vocabulary, decode


def main() -> None:
    """
    Launches an implementation
    """
    assets_path = Path(__file__).parent / 'assets'
    with open(assets_path / 'text.txt', 'r', encoding='utf-8') as text_file:
        text = text_file.read()
    with open(assets_path / 'secrets' / 'secret_1.txt', 'r', encoding='utf-8') as text_file:
        encoded_text = text_file.read()
    trained_dict = train(collect_frequencies(text, start_of_word=None, end_of_word='</s>'), 100)
    if not trained_dict:
        return
    vocab = get_vocabulary(trained_dict, '<unk>')
    encoded_list = [int(code) for code in encoded_text.split()]
    result = decode(encoded_list, vocab, '</s>')
    print(result)
    assert result, "Encoding is not working"


if __name__ == "__main__":
    main()
