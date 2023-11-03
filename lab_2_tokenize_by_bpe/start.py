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
    # Mark6 results
    with open(assets_path / 'text.txt', 'r', encoding='utf-8') as file:
        text = file.read()
    text_freq = collect_frequencies(text, None, '</s>')
    text_freq = train(text_freq, 100)
    # Secrets
    if text_freq:
        with open(assets_path / 'secrets/secret_5.txt', 'r', encoding='utf-8') as file:
            secret_text = file.read()
        vocabulary = get_vocabulary(text_freq, '<unk>')
        if vocabulary:
            print(vocabulary)

            secret_list = [int(num) for num in secret_text.split()]
            decoded = decode(secret_list, vocabulary, '</s>')

            print(decoded)
            result = decoded
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
        for pred, actual in zip(encoded_pred, encoded_actual.split()):
            if pred != actual:
                # print(pred, actual)
                pass

    # It's correct, but there are three words that can be made with fewer n-grams or differently
    # _Произошло : 1003, (53, 11759, 1492 / 1483, 6737), 818
    # _Космонавтом : 5468, 230, (5699, 30218, 46 / 13283, 222, 6896)
    # _Гагарин. : 7150, (1382, 8324, 141 / 7744, 2190), 3

    print("\u2581\u041f\u0440\u043e", "\u0438", "\u0437\u043e", "\u0448", "\u043b\u043e")
    print("\u2581\u041f\u0440\u043e", "\u0438\u0437", "\u043e\u0448", "\u043b\u043e")
    print("\u2581\u041a\u043e", "\u0441", "\u043c\u043e\u043d",
            "\u0430\u0432\u0442\u043e", "\u043c")
    print("\u2581\u041a\u043e", "\u0441", "\u043c\u043e\u043d\u0430",
            "\u0432", "\u0442\u043e\u043c")
    print("\u2581\u0413\u0430", "\u0433", "\u0430\u0440\u0438", "\u043d", ".")
    print("\u2581\u0413\u0430", "\u0433\u0430\u0440", "\u0438\u043d", ".")

    with open(assets_path / 'for_translation_en_encoded.txt', 'r', encoding='utf-8') as file:
        encoded = file.read()
    decoded = decode([int(num) for num in encoded.split()], vocabulary, None)
    decoded = decoded.replace('\u2581', ' ')
    # decoded = decoded[6:-4]  # tokens of start and end of text
    print(decoded)
    with open(assets_path / 'for_translation_en_raw.txt', 'r', encoding='utf-8') as file:
        actual = file.read()
    print(calculate_bleu(decoded, actual))


if __name__ == "__main__":
    main()
