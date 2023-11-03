"""
BPE Tokenizer starter
"""
from pathlib import Path

from main import collect_frequencies, count_tokens_pairs, train


def main() -> None:
    """
    Launches an implementation
    """
    assets_path = Path(__file__).parent / 'assets'
    with open(assets_path / 'text.txt', 'r', encoding='utf-8') as text_file:
        text = text_file.read()

    result1 = collect_frequencies(text, None, '</s>')
    print(result1)
    result2 = count_tokens_pairs(result1)
    print(result2)
    print(train(result2, 100))
    assert result2, "Encoding is not working"


if __name__ == "__main__":
    main()
