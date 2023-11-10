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

    with open(assets_path / 'secrets/secret_4.txt', 'r', encoding='utf-8') as text_file:
        encoded = text_file.read()
    freq_dict = collect_frequencies(text, None, '</s>')
    merged_tokens = train(freq_dict, 100)

    if merged_tokens:
        vocab = get_vocabulary(merged_tokens, '<unk>')
        secret = [int(num) for num in encoded.split()]
        result = decode(secret, vocab, '</s>')
        print(result)

        assert result, "Encoding is not working"


if __name__ == "__main__":
    main()
