"""
BPE Tokenizer starter
"""
from pathlib import Path
import lab_2_tokenize_by_bpe.main


def main() -> None:
    """
    Launches an implementation
    """
    assets_path = Path(__file__).parent / 'assets'
    with open(assets_path / 'text.txt', 'r', encoding='utf-8') as text_file:
        text = text_file.read()

    result = lab_2_tokenize_by_bpe.main.collect_frequencies(text, None, '</s>')
    #assert result, "Encoding is not working"


if __name__ == "__main__":
    main()
