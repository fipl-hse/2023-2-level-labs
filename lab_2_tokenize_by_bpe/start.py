"""
BPE Tokenizer starter
"""
from pathlib import Path

import lab_2_tokenize_by_bpe.main as main_bpe


def main() -> None:
    """
    Launches an implementation
    """
    assets_path = Path(__file__).parent / 'assets'
    with open(assets_path / 'text.txt', 'r', encoding='utf-8') as text_file:
        text = text_file.read()
    result = (main_bpe.collect_frequencies(text, None, '</s>'))
    train_result = (main_bpe.train(result, 100))

    if result is None:
        assert result, "Encoding is not working"
    else:
        print(result)
        print(train_result)


if __name__ == "__main__":
    main()
