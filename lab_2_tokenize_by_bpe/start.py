"""
BPE Tokenizer starter
"""
from pathlib import Path
from lab_2_tokenize_by_bpe.main import collect_frequencies, train, get_vocabulary, decode


def main() -> None:
    """
    Launches an implementation
    """
    assets_path = Path(__file__).parent / 'assets'
    with open(assets_path / 'text.txt', 'r', encoding='utf-8') as text_file:
        text = text_file.read()
    with open(assets_path / 'secrets/secret_5.txt', 'r', encoding='utf-8') as text_file:
        secret = text_file.read()
    word_frequencies = collect_frequencies(text, None, '</s>')
    massive_merge = train(word_frequencies, 100)
    vocab = get_vocabulary(massive_merge, '<unk>')
    secret_prepared = []
    for num in secret.split():
        secret_prepared.append(int(num))
    result = decode(secret_prepared, vocab, '<unk>')
    print(result)
    assert result, "Encoding is not working"


if __name__ == "__main__":
    main()
