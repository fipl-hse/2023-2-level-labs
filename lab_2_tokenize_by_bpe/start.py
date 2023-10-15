"""
BPE Tokenizer starter
"""
from pathlib import Path
from main import collect_frequencies

def main() -> None:
    """
    Launches an implementation
    """
    assets_path = Path(__file__).parent / 'assets'
    with open(assets_path / 'text.txt', 'r', encoding='utf-8') as text_file:
        text = text_file.read()
    print(collect_frequencies(text, None, '</s>'))


if __name__ == "__main__":
    main()
