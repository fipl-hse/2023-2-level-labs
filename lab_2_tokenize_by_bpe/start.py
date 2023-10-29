"""
BPE Tokenizer starter
"""
from pathlib import Path


def main() -> None:
    """
    Launches an implementation
    """
    assets_path = Path(__file__).parent / 'assets'
    with open(assets_path / 'text.txt', 'r', encoding='utf-8') as text_file:
        text = text_file.read()

    dict_of_freqs = collect_frequencies(text, '<s>')
    merging_tokens = train(dict_of_freqs, 100)

    result = None
    assert result, "Encoding is not working"


if __name__ == "__main__":
    main()
