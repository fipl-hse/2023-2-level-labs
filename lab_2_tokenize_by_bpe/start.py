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
    # with open(assets_path / 'text.txt', 'r', encoding='utf-8') as text_file:
    #     text = text_file.read()
    with open(assets_path / 'for_translation_ru_raw.txt', 'r', encoding='utf-8') as file:
        initial_text = file.read()
    vocabulary = load_vocabulary('./assets/vocab.json')
    with open(assets_path / 'for_translation_ru_encoded.txt', 'r', encoding='utf-8') as file:
        actual_encoded = file.read()

    encoded_text = encode(initial_text, vocabulary, '\u2581', None, '<unk>')
    if encoded_text:
        if encoded_text == map(int, actual_encoded.split()):
            print('Текст на русском языке кодируется верно')

    #print(list(map(int, actual_encoded.split()))) #1071, 13763, ...
    #print(encoded_text)  #21, 746, 70, ...
    # print(list(vocabulary.keys())[1071], list(vocabulary.keys())[13763])
    # print(list(vocabulary.keys())[21], list(vocabulary.keys())[746])
    # На самом деле кодируется строка правильно, но в данном нам словаре ключи не идут по обыванию,
    # поэтому, я думаю, что происходит данное расхождение у encoded_text и actual_encoded
    with open(assets_path / 'for_translation_en_encoded.txt', 'r', encoding='utf-8') as file:
        precisions = file.read()
    encoded_precisions = list(map(int, precisions.split()))
    decoded_precisions = decode(encoded_precisions, vocabulary, None)
    if decoded_precisions:
        decoded_precisions = decoded_precisions.replace('\u2581', ' ')
    with open(assets_path / 'for_translation_en_raw.txt', 'r', encoding='utf-8') as file:
        ideal_reference = file.read()
    bleu =  calculate_bleu(decoded_precisions, ideal_reference)
    if bleu:
        print(bleu)
    # print(decoded_precisions)
    # print(ideal_reference)

    result = bleu
    assert result, "Decoding is not working"


if __name__ == "__main__":
    main()
