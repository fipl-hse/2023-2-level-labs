"""
BPE Tokenizer starter
"""
from pathlib import Path

from lab_2_tokenize_by_bpe.main import collect_frequencies, train


def main() -> None:
    """
    Launches an implementation
    """
    assets_path = Path(__file__).parent / 'assets'
    with open(assets_path / 'text.txt', 'r', encoding='utf-8') as text_file:
        text = text_file.read()

    #for mark 4:
    dict_of_freq = collect_frequencies(text, None, "</s>")
    print(dict_of_freq)

    #for mark 6:
    trained_dict = train(dict_of_freq, 100)
    print(trained_dict)

    result = trained_dict
    assert result, "Encoding is not working"


if __name__ == "__main__":
    main()
