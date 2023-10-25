"""
BPE Tokenizer starter
"""
from pathlib import Path
from main import train
from main import collect_frequencies


def main() -> None:
    """
    Launches an implementation
    """
    assets_path = Path(__file__).parent / 'assets'
    with open(assets_path / 'text.txt', 'r', encoding='utf-8') as text_file:
        text = text_file.read()

    result = train(collect_frequencies(text, None, '</s>'), 100)
    print(result)
    assert result, "Encoding is not working"


if __name__ == "__main__":
    main()
