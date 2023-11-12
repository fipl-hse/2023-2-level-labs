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
    processor = TextProcessor(end_of_word_token='_')
    encoded = processor.encode(text)
    if not encoded:
        return None
    result = processor.decode(encoded)
    print(result)
    assert result


if __name__ == "__main__":
    main()
