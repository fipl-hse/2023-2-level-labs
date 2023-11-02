"""
BPE Tokenizer starter
"""
import json
from pathlib import Path
from lab_2_tokenize_by_bpe.main import (calculate_bleu, decode, encode)


def main() -> None:
    """
    Launches an implementation
    """
    assets_path = Path(__file__).parent / 'assets'
    with open(assets_path / 'text.txt', 'r', encoding='utf-8') as text_file:
        text = text_file.read()
    with open(assets_path / 'vocab.json', 'r', encoding='utf-8') as file:
        vocabulary = json.load(file)
    with open(assets_path / 'for_translation_ru_raw.txt', 'r', encoding='utf-8') as file:
        ru_raw = file.read()
    with open(assets_path / 'for_translation_ru_encoded.txt', 'r', encoding='utf-8') as file:
        ru_encoded = file.read()

    encode_pred = encode(ru_raw, vocabulary, '\u2581', None, '<unk>')
    correct_tokens = [token for token in encode_pred if token in map(int, ru_encoded.split())]
    print(f"Файл закодирован правильно на {(len(correct_tokens) / len(encode_pred)*100)}%")

    with open(assets_path / 'for_translation_en_encoded.txt', 'r', encoding='utf-8') as file:
        encoded_en = file.read()
    with open(assets_path / 'for_translation_en_raw.txt', 'r', encoding='utf-8') as file:
        en_raw = file.read()

    decoded_text = decode([int(num) for num in encoded_en.split()], vocabulary, None)
    decoded_text = decoded_text.replace('\u2581', ' ')
    print(f'BLUE = {calculate_bleu(decoded_text, en_raw)}')


if __name__ == "__main__":
    main()
