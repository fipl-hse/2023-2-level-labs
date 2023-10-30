"""
BPE Tokenizer starter
"""
from pathlib import Path
from lab_2_tokenize_by_bpe.main import collect_frequencies, decode, get_vocabulary, train


def main() -> None:
    """
    Launches an implementation
    """
    assets_path = Path(__file__).parent / 'assets'
    with open(assets_path / 'text.txt', 'r', encoding='utf-8') as text_file:
        text = text_file.read()

    with open(assets_path/ 'secrets/secret_3.txt', 'r', encoding='utf-8') as text_file:
        secret = text_file.read()
    dict_of_freqs = collect_frequencies(text, '<s>')
    merged_tokens = train(dict_of_freqs, 100)
    if merged_tokens:
        vocabulary = get_vocabulary(merged_tokens, '<unk>')
        list_of_nums = [int(num) for num in secret.split()]
        transcript = decode(list_of_nums, vocabulary, '</s>')
        print(transcript)
    result = None
    assert result, "Encoding is not working"


if __name__ == "__main__":
    main()
