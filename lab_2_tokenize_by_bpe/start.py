"""
BPE Tokenizer starter
"""
from pathlib import Path
from lab_2_tokenize_by_bpe.main import train, collect_frequencies


def main() -> None:
    """
    Launches an implementation
    """
    assets_path = Path(__file__).parent / 'assets'
    with open(assets_path / 'text.txt', 'r', encoding='utf-8') as text_file:
        text = text_file.read()
    result = train(collect_frequencies(
            'Вез корабль карамель, наскочил корабль на мель, '
            'матросы две недели карамель на мели ели.', None, '</s>'), 30)
    print(result)
    assert result, "Encoding is not working"


if __name__ == "__main__":
    main()
