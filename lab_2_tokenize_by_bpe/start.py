"""
BPE Tokenizer starter
"""
from pathlib import Path

from lab_2_tokenize_by_bpe.main import calculate_bleu, decode, encode, load_vocabulary


def main() -> None:
    """
    Launches an implementation
    """
    assets_path = Path(__file__).parent / 'assets'

    with open(assets_path/'for_translation_ru_raw.txt', 'r', encoding='utf-8') as file:
        ru_raw_text = file.read()
    with open(assets_path/'for_translation_ru_encoded.txt','r', encoding='utf-8') as file:
        translation_ru = file.read()
    with open(assets_path/'for_translation_en_encoded.txt', 'r', encoding='utf-8') as file:
        en_encoded_text = file.read()
    with open(assets_path/'for_translation_en_raw.txt', 'r', encoding='utf-8') as file:
        translation_en = file.read()
    vocabulary = load_vocabulary('./assets/vocab.json')
    ru_encoding = encode(ru_raw_text, vocabulary, '\u2581', None, '<unk>')
    if ru_encoding is not None:
        for encoding, translation in zip(ru_encoding, translation_ru):
            if encoding == translation:
                print('Everything is ok')
    list_of_indexes = []
    for index in en_encoded_text.split():
        list_of_indexes.append(int(index))
    decoding = decode(list_of_indexes, vocabulary, None)
    decoding = decoding.replace('\u2581', ' ')
    result = calculate_bleu(decoding, translation_en)
    print(result)
    assert result, "Encoding is not working"


if __name__ == "__main__":
    main()
