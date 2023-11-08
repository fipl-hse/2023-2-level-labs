"""
Generation by NGrams starter
"""
from lab_3_generate_by_ngrams.main import TextProcessor


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    split_text = TextProcessor('_')
    encoding = split_text.encode(text)
    result = split_text.decode(encoding)
    print('Encoding of the text:', encoding)
    print('Decoding of the text:', result)
    assert result

if __name__ == "__main__":
    main()
