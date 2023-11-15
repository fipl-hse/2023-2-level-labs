"""
Generation by NGrams starter
"""

import lab_3_generate_by_ngrams.main as main_py


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
        text_processor = main_py.TextProcessor('_')
        encoded_text = text_processor.encode(text)
        decoded_text = text_processor.decode(encoded_text)
    result = decoded_text
    assert result


if __name__ == "__main__":
    main()
