"""
BPE Tokenizer starter
"""
from pathlib import Path

from lab_2_tokenize_by_bpe.main import (
    collect_frequencies, decode, get_vocabulary, train,
    load_vocabulary, encode, collect_ngrams,
    calculate_precision, geo_mean, calculate_bleu
)


def main() -> None:
    """
    Launches an implementation
    """
    assets_path = Path(__file__).parent / 'assets'
    with open(assets_path / 'text.txt', 'r', encoding='utf-8') as text_file:
        text = text_file.read()

    with open(assets_path / 'secrets/secret_3.txt', 'r', encoding='utf-8') as secret_file:
        encoded_text = secret_file.read()

    word_frequencies = collect_frequencies(text, None, '</s>')
    word_frequencies = train(word_frequencies, 100)
    if word_frequencies:
        vocabulary = get_vocabulary(word_frequencies, '<unk>')
        encoded_text_list = [int(num) for num in encoded_text.split()]
        result = decode(encoded_text_list, vocabulary, '</s>')
        print(result)
        assert result, "Encoding is not working"

    with open(assets_path / 'for_translation_ru_raw.txt', 'r', encoding='utf-8') as file:
        predicted_ru = file.read()
    vocabulary_adv = load_vocabulary('assets/vocab.json')
    if not vocabulary_adv:
        return None
    with open(assets_path / 'for_translation_ru_encoded.txt', 'r', encoding='utf-8') as file:
        reference_ru = file.read()

    predicted_encoded = encode(predicted_ru, vocabulary_adv, '\u2581', None, '<unk>')
    if not predicted_encoded:
        return None

    reference_ru_list = reference_ru.split()
    for pred_token, actual_token in zip(predicted_encoded, reference_ru_list):
        if pred_token != actual_token:
            print(pred_token, actual_token)

    with open(assets_path / 'for_translation_en_encoded.txt', 'r', encoding='utf-8') as file:
        encoded_en = file.read().split()
    with open(assets_path / 'for_translation_en_raw.txt', 'r', encoding='utf-8') as file:
        reference_en = file.read()

    decoded_en = decode(encoded_en, vocabulary_adv, None)
    if not decoded_en:
        return None

    decoded_en = decoded_en.replace('\u2581', ' ')
    bleu = calculate_bleu(decoded_en, reference_en)
    if not bleu:
        return None
    print(bleu)
    return bleu


if __name__ == "__main__":
    main()
