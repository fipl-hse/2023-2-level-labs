"""
BPE Tokenizer starter
"""
import json
from pathlib import Path

from lab_2_tokenize_by_bpe.main import (calculate_bleu, collect_frequencies,
                                        decode, encode, get_vocabulary, train)


def main() -> None:
    """
    Launches an implementation
    """
    assets_path = Path(__file__).parent / 'assets'
    with open(assets_path / 'text.txt', 'r', encoding='utf-8') as text_file:
        text = text_file.read()
    with open(assets_path / 'secrets/secret_1.txt', 'r', encoding='utf-8') as text_file:
        secret = text_file.read()
    word_frequencies = collect_frequencies(text, None, '</s>')
    massive_merge = train(word_frequencies, 100)
    assert isinstance(massive_merge, dict)
    vocab = get_vocabulary(massive_merge, '<unk>')
    secret_prepared = []
    for num in secret.split():
        secret_prepared.append(int(num))
    result = decode(secret_prepared, vocab, '</s>')
    print(result)
    assert result, "Encoding is not working"
    with open(assets_path / 'for_translation_ru_raw.txt', 'r', encoding='utf-8') as text_file:
        text_for_translation = text_file.read()
    with open(assets_path / 'vocab.json', 'r', encoding='utf-8') as text_file:
        vocabulary = json.load(text_file)
    encoded_text_f_t = encode(text_for_translation, vocabulary, '\u2581', None, '<unk>')
    print(encoded_text_f_t)


if __name__ == "__main__":
    main()
