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
        encoded_text = file.read()

    word_frequencies = collect_frequencies(text, None, '</s>')
    word_frequencies = train(word_frequencies, 100)

    if word_frequencies:
        vocabulary = get_vocabulary(word_frequencies, '<unk>')
        encoded_text = [int(num) for num in encoded_text.split()]
        result = decode(encoded_text, vocabulary, '</s>')
        print(result)
        assert result, "Encoding is not working"

    with open(assets_path / 'for_translation_ru_raw.txt', 'r', encoding='utf-8') as file:
        predicted_ru = file.read()

    vocabulary = load_vocabulary('assets/vocab.json')
    if not vocabulary:
        return None

    with open(assets_path / 'for_translation_ru_encoded.txt', 'r', encoding='utf-8') as file:
        ref_ru = file.read()

    predicted_ru = encode(predicted_ru, vocabulary, '\u2581', None, '<unk>')
    if not predicted_ru:
        return None

    ref_ru = ref_ru.split()
    for pred_token, actual_token in zip(predicted_ru, ref_ru):
        if pred_token != actual_token:
            print(pred_token, actual_token)

    with open(assets_path / 'for_translation_en_encoded.txt', 'r', encoding='utf-8') as file:
        encoded_en = file.read().split()
    with open(assets_path / 'for_translation_en_raw.txt', 'r', encoding='utf-8') as file:
        reference_en = file.read()

    encoded_en = [int(i) for i in encoded_en]
    decoded_en = decode(encoded_en, vocabulary, None)
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
