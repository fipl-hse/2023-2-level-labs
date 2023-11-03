"""
BPE Tokenizer starter
"""
import json
from pathlib import Path

from lab_2_tokenize_by_bpe.main import (calculate_bleu, collect_frequencies, decode, encode,
                                        get_vocabulary, train)


def main() -> None:
    """
    Launches an implementation
    """
    assets_path = Path(__file__).parent / 'assets'
    with open(assets_path / 'text.txt', 'r', encoding='utf-8') as text_file:
        text = text_file.read()
    with open(assets_path / 'secrets/secret_2.txt', 'r', encoding='utf-8') as text_file:
        encoded_secret = text_file.read()
    dict_frequencies = collect_frequencies(text, None, '</s>')
    merged_tokens = train(dict_frequencies, 100)
    if merged_tokens:
        vocabulary = get_vocabulary(merged_tokens, '<unk>')
        secret = [int(num) for num in encoded_secret.split()]
        result = decode(secret, vocabulary, '</s>')
        print(result)
        assert result, "Encoding is not working"

    with open(assets_path / 'for_translation_ru_raw.txt', 'r', encoding='utf-8') as file:
        predicted = file.read()
    with open(assets_path / 'vocab.json', 'r', encoding='utf-8') as file:
        vocabulary = json.load(file)
    with open(assets_path / 'for_translation_ru_encoded.txt', 'r', encoding='utf-8') as file:
        actual = file.read()

    if [int(token) for token in actual.split()] == encode(
            predicted, vocabulary, '\u2581', None, '<unk>'):
        print("Encoding is successful!")

    with open(assets_path / 'for_translation_en_encoded.txt', 'r', encoding='utf-8') as file:
        encoded_en = file.read()
    with open(assets_path / 'for_translation_en_raw.txt', 'r', encoding='utf-8') as file:
        decoded_en = file.read()

    decoded = decode([int(num) for num in encoded_en.split()], vocabulary, None)
    decoded = decoded.replace('\u2581', ' ')

    print(calculate_bleu(decoded, decoded_en))


if __name__ == "__main__":
    main()
