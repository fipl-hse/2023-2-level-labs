"""
BPE Tokenizer starter
"""
from pathlib import Path

from lab_2_tokenize_by_bpe.main import (calculate_bleu, collect_frequencies, decode, encode,
                                        get_vocabulary, load_vocabulary, train)


def main() -> None:
    """
    Launches an implementation
    """
    assets_path = Path(__file__).parent / 'assets'
    with open(assets_path / 'text.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    with open(assets_path / 'secrets/secret_3.txt', 'r', encoding='utf-8') as file:
        encoded = [int(num) for num in file.read().split()]

    word_frequencies = collect_frequencies(text, None, '</s>')
    word_frequencies = train(word_frequencies, 100)

    if word_frequencies:
        vocabulary = get_vocabulary(word_frequencies, '<unk>')
        result = decode(encoded, vocabulary, '</s>')
        print(result)
        assert result, "Encoding is not working"

    with open(assets_path / 'for_translation_ru_raw.txt', 'r', encoding='utf-8') as file:
        predicted = file.read()

    vocabulary = load_vocabulary('assets/vocab.json')
    if not vocabulary:
        return None

    with open(assets_path / 'for_translation_ru_encoded.txt', 'r', encoding='utf-8') as file:
        reference = file.read().split()

    predicted_ru = encode(predicted, vocabulary, '\u2581', None, '<unk>')
    if not predicted_ru:
        return None

    for pred_token, actual_token in zip(predicted_ru, reference):
        if pred_token != actual_token:
            print(pred_token, actual_token)

    with open(assets_path / 'for_translation_en_encoded.txt', 'r', encoding='utf-8') as file:
        encoded = [int(num) for num in file.read().split()]
    with open(assets_path / 'for_translation_en_raw.txt', 'r', encoding='utf-8') as file:
        reference_en = file.read()

    decoded_en = decode(encoded, vocabulary, None)
    if not decoded_en:
        return None

    decoded_en = decoded_en.replace('\u2581', ' ')
    bleu = calculate_bleu(decoded_en, reference_en)
    if not bleu:
        return None
    print(bleu)
    return None


if __name__ == "__main__":
    main()
