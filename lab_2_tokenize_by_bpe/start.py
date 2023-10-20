"""
BPE Tokenizer starter
"""
import json
from pathlib import Path

from lab_2_tokenize_by_bpe.main import collect_frequencies, decode, encode, get_vocabulary, train


def main() -> None:
    """
    Launches an implementation
    """
    assets_path = Path(__file__).parent / 'assets'
    # Mark6 results
    with open(assets_path / 'text.txt', 'r', encoding='utf-8') as file:
        text = file.read()
    text_freq = collect_frequencies(text, None, '</s>')
    merged_freq = train(text_freq, 100)
    # Secrets
    if merged_freq:
        with open(assets_path / 'secrets/secret_5.txt', 'r', encoding='utf-8') as file:
            secret_text = file.read()
        vocabulary = get_vocabulary(merged_freq, '<unk>')
        if vocabulary:
            print(vocabulary)

            secret_list = [int(num) for num in secret_text.split()]
            decoded_secret = decode(secret_list, vocabulary, '</s>')

            decoded_secret = decoded_secret.replace('ев', 'ел')
            decoded_secret = decoded_secret.replace('до', 'ев')
            print(decoded_secret)
            result = decoded_secret
            assert result, "Encoding is not working"

    # Step 14
    with open(assets_path / 'for_translation_ru_raw.txt', 'r', encoding='utf-8') as file:
        text_predicted = file.read()
    with open(assets_path / 'vocab.json', 'r', encoding='utf-8') as file:
        vocabulary = json.load(file)
    with open(assets_path / 'for_translation_ru_encoded.txt', 'r', encoding='utf-8') as file:
        encoded_actual = file.read()

    encoded_pred = encode(text_predicted, vocabulary, '\u2581', None, '<unk>')
    if encoded_pred:
        for letter_pred, letter_actual in zip(encoded_pred, encoded_actual.split()):
            if letter_pred != int(letter_actual):
                print(letter_pred, letter_actual)
    # _Произошло : 1003, (53, 11759, 1492 / 1483, 6737), 818
    # _Космонавтом : 5468, 230, (5699, 30218, 46 / 13283, 222, 6896)
    # _Гагарин. : 7150, (1382, 8324, 141 / 7744, 2190), 3


if __name__ == "__main__":
    main()
