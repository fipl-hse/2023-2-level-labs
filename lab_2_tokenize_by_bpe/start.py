"""
BPE Tokenizer starter
"""
from pathlib import Path
from main import collect_frequencies, count_tokens_pairs, merge_tokens, train


def main() -> None:
    """
    Launches an implementation
    """
    assets_path = Path(__file__).parent / 'assets'
    with open(assets_path / 'text.txt', 'r', encoding='utf-8') as text_file:
        text = text_file.read()
    word_frequencies = collect_frequencies(text, None, '</s>')
    result = train(word_frequencies, 100)
    print(result)
    assert result, "Encoding is not working"


if __name__ == "__main__":
    main()
