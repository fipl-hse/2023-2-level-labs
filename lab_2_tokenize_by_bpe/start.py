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

    with open(assets_path / 'secrets' / 'secret_1.txt', 'r', encoding='utf-8') as text_file:
        encoded_text = text_file.read().split()

    to_encode = []
    for i in encoded_text:
        to_encode.append(int(i))

    #for mark 4:
    dict_of_freq = collect_frequencies(text, None, "</s>")
    print(dict_of_freq)

    #for mark 6:
    trained_dict = train(dict_of_freq, 100)
    print(trained_dict)

    #for mark 8:
    if not trained_dict:
        return
    dict_of_tokens = get_vocabulary(trained_dict, '<unk>')
    decoded_text = decode(to_encode, dict_of_tokens, "</s>")
    print(decoded_text)

    result = decoded_text
    assert result, "Encoding is not working"


if __name__ == "__main__":
    main()
