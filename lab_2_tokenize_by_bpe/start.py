"""
BPE Tokenizer starter
"""
from pathlib import Path

from lab_2_tokenize_by_bpe.main import collect_frequencies, decode, get_vocabulary, train

# import json


def main() -> None:
    """
    Launches an implementation
    """
    assets_path = Path(__file__).parent / 'assets'
    # Mark6 results
    with open(assets_path / 'text.txt', 'r', encoding='utf-8') as text_file:
        text = text_file.read()
    text_freq = collect_frequencies(text, None, '</s>')
    merged_freq = train(text_freq, 100)
    if merged_freq:
        with open(assets_path / 'secrets/secret_4.txt', 'r', encoding='utf-8') as text_file:
            secret_text = text_file.read()
        secret_vocab = get_vocabulary(merged_freq, '<unk>')
        if secret_vocab:
            print(secret_vocab)

            secret_list = [int(num) for num in secret_text.split()]
            decoded_secret = decode(secret_list, secret_vocab, '</s>')

            print(decoded_secret)
            result = decoded_secret
            assert result, "Encoding is not working"

    # with open(assets_path / 'vocab.json', 'r', encoding='utf-8') as file:
    #     vocab = json.load(file)
    #
    # corrected_vocab = {}
    # for key, value in vocab.items():
    #     if '\u2581' in key:
    #         corrected_vocab[' ' + key[1:]] = value
    #     else:
    #         corrected_vocab[key] = value


if __name__ == "__main__":
    main()
