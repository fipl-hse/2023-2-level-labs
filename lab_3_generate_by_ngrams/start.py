"""
Generation by NGrams starter
"""
from main import TextProcessor

def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/Harry_Potter.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    result = TextProcessor('_')
    encoded_corpus = result.encode(text)
    print(result.decode(encoded_corpus))
    assert result


if __name__ == "__main__":
    main()
