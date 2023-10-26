"""
BPE Tokenizer starter
"""
from pathlib import Path
from lab_2_tokenize_by_bpe.main import collect_frequencies


def main() -> None:
    """
    Launches an implementation
    """
    assets_path = Path(__file__).parent / 'assets'
    with open(assets_path / 'text.txt', 'r', encoding='utf-8') as text_file:
        text = text_file.read()

    result = None
    assert result, "Encoding is not working"
    frequency_dictionary = collect_frequencies(text, None, '</s>')
    print(frequency_dictionary)


if __name__ == "__main__":
    main()
